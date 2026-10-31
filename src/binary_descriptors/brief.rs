//! Structs and functions for finding and computing BRIEF descriptors as
//! described in [Calonder, et. al. (2010)].
///
/// [Calonder, et. al. (2010)]: https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
use image::{GenericImageView, GrayImage, Luma};
use rand_distr::{Distribution, Normal};

use crate::{corners::Corner, definitions::Image, integral_image::integral_image, point::Point};

use super::{
    BinaryDescriptor,
    constants::{BRIEF_PATCH_DIAMETER, BRIEF_PATCH_RADIUS},
};

/// BRIEF descriptor as described in [Calonder, et. al. (2010)].
///
/// [Calonder, et. al. (2010)]: https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
#[derive(Clone, PartialEq)]
pub struct BriefDescriptor {
    /// Results of the pairwise pixel intensity tests that comprise this BRIEF
    /// descriptor.
    pub bits: Vec<u128>,
    /// Pixel location and corner score of the keypoint associated with this
    /// BRIEF descriptor.
    pub corner: Corner,
}

impl BinaryDescriptor for BriefDescriptor {
    fn get_size(&self) -> u32 {
        (self.bits.len() * 128) as u32
    }

    fn hamming_distance(&self, other: &Self) -> u32 {
        assert_eq!(self.get_size(), other.get_size());
        self.bits
            .iter()
            .zip(other.bits.iter())
            .fold(0, |acc, x| acc + (x.0 ^ x.1).count_ones())
    }

    fn get_bit_subset(&self, bits: &[u32]) -> u128 {
        assert!(
            bits.len() <= 128,
            "Can't extract more than 128 bits (found {})",
            bits.len()
        );
        let mut subset = 0;
        for b in bits {
            subset <<= 1;
            // isolate the bit at index b
            subset += (self.bits[(b / 128) as usize] >> (b % 128)) % 2;
        }
        subset
    }

    fn position(&self) -> Point<u32> {
        self.corner.into()
    }
}

/// Collection of two points that a BRIEF descriptor uses to generate its bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestPair {
    /// The first point in the pair.
    pub p0: Point<u32>,
    /// The second point in the pair.
    pub p1: Point<u32>,
}

fn local_pixel_average(integral_image: &Image<Luma<u32>>, x: u32, y: u32, radius: u32) -> u8 {
    if radius == 0 {
        return 0;
    }
    let y_min = y.saturating_sub(radius);
    let x_min = x.saturating_sub(radius);
    let y_max = u32::min(
        y.saturating_add(radius).saturating_add(1),
        integral_image.height().saturating_sub(1),
    );
    let x_max = u32::min(
        x.saturating_add(radius).saturating_add(1),
        integral_image.width().saturating_sub(1),
    );

    if y_max <= y_min || x_max <= x_min {
        return 0;
    }
    let pixel_area = (y_max - y_min) * (x_max - x_min);

    // UNSAFETY JUSTIFICATION
    //
    // Benefit
    //
    // Removing the unsafe pixel accesses in this function increases the
    // runtimes for bench_brief_fixed_test_pairs_1000_keypoints,
    // bench_brief_random_test_pairs_1000_keypoints, and
    // bench_rotated_brief_1000_keypoints by about 40%, 30%, and 30%,
    // respectively.
    //
    // Correctness
    //
    // The values of x_min, x_max, y_min, and y_max are all bounded between zero
    // and the integral image dimensions by the checks at the top of this
    // function.
    let (bottom_right, top_left, top_right, bottom_left) = unsafe {
        (
            integral_image.unsafe_get_pixel(x_max, y_max)[0],
            integral_image.unsafe_get_pixel(x_min, y_min)[0],
            integral_image.unsafe_get_pixel(x_max, y_min)[0],
            integral_image.unsafe_get_pixel(x_min, y_max)[0],
        )
    };
    let total_intensity = bottom_right + top_left - top_right - bottom_left;
    (total_intensity / pixel_area) as u8
}

pub(crate) fn brief_impl(
    integral_image: &Image<Luma<u32>>,
    keypoints: &[Point<u32>],
    test_pairs: &[TestPair],
    length: usize,
) -> Result<Vec<BriefDescriptor>, String> {
    if !length.is_multiple_of(128) {
        return Err(format!(
            "BRIEF descriptor length must be a multiple of 128 bits (found {})",
            length
        ));
    }

    if length != test_pairs.len() {
        return Err(format!(
            "BRIEF descriptor length must be equal to the number of test pairs ({} != {})",
            length,
            test_pairs.len()
        ));
    }

    for (i, tp) in test_pairs.iter().enumerate() {
        if tp.p0.x >= BRIEF_PATCH_DIAMETER
            || tp.p0.y >= BRIEF_PATCH_DIAMETER
            || tp.p1.x >= BRIEF_PATCH_DIAMETER
            || tp.p1.y >= BRIEF_PATCH_DIAMETER
        {
            return Err(format!(
                "Test pair {} has coordinates outside the {}x{} patch: p0=({}, {}), p1=({}, {})",
                i, BRIEF_PATCH_DIAMETER, BRIEF_PATCH_DIAMETER, tp.p0.x, tp.p0.y, tp.p1.x, tp.p1.y
            ));
        }
    }

    let mut descriptors: Vec<BriefDescriptor> = Vec::with_capacity(keypoints.len());
    // the integral image has an extra row and column of zeros at the top-left
    let (width, height) = (integral_image.width() - 1, integral_image.height() - 1);
    // minimum number of pixels between a keypoint and each image edge
    let margin = BRIEF_PATCH_RADIUS + 1;

    for keypoint in keypoints {
        // if the keypoint is too close to the edge, return an error
        if keypoint.x < margin
            || keypoint.x.saturating_add(margin) >= width
            || keypoint.y < margin
            || keypoint.y.saturating_add(margin) >= height
        {
            return Err(format!(
                "Found keypoint less than {} px from image edge: ({}, {})",
                margin, keypoint.x, keypoint.y
            ));
        }

        let patch_top_left = Point {
            x: keypoint.x - BRIEF_PATCH_RADIUS,
            y: keypoint.y - BRIEF_PATCH_RADIUS,
        };

        let mut descriptor = BriefDescriptor {
            bits: Vec::with_capacity(length / 128),
            corner: Corner {
                x: keypoint.x,
                y: keypoint.y,
                score: 0.,
            },
        };
        let mut descriptor_chunk = 0u128;
        // for each test pair, compare the pixels within the patch at those points
        for (idx, test_pair) in test_pairs.iter().enumerate() {
            // if we've entered a new chunk, then save the previous one
            if idx != 0 && idx % 128 == 0 {
                descriptor.bits.push(descriptor_chunk);
                descriptor_chunk = 0;
            }

            let p0 = Point {
                x: test_pair.p0.x + patch_top_left.x,
                y: test_pair.p0.y + patch_top_left.y,
            };
            let p1 = Point {
                x: test_pair.p1.x + patch_top_left.x,
                y: test_pair.p1.y + patch_top_left.y,
            };

            // if p0 < p1, then record true for this test; otherwise, record false
            descriptor_chunk <<= 1;
            descriptor_chunk += (local_pixel_average(integral_image, p0.x, p0.y, 2)
                < local_pixel_average(integral_image, p1.x, p1.y, 2))
                as u128;
        }
        // save the final chunk too
        descriptor.bits.push(descriptor_chunk);
        descriptors.push(descriptor);
    }

    Ok(descriptors)
}

/// Generates BRIEF descriptors for small patches around keypoints in an image.
///
/// Returns a tuple containing a vector of `BriefDescriptor` and a vector of
/// `TestPair`. All returned descriptors are based on the same `TestPair` set.
/// Patches are 31x31 pixels centred on the keypoints, and there must be at
/// least 16 pixels between each keypoint and every image edge (i.e.
/// `16 <= x < width - 16` and `16 <= y < height - 16`). If any keypoints are
/// too close to an edge, returns an error.
///
/// `length` must be a multiple of 128 bits. Returns an error otherwise.
///
/// If `override_test_pairs` is `Some`, then those test pairs are used, and none
/// are generated. Use this when you already have test pairs from another run
/// and want to compare the descriptors later.
///
/// If `override_test_pairs` is `None`, then `TestPair`s are generated according
/// to an isotropic Gaussian.
///
/// [Calonder, et. al. (2010)] used Gaussian smoothing to decrease the effects of noise in the
/// patches. This is slow, even with a box filter approximation. For maximum
/// performance, the average intensities of sub-patches of radius 5 around the
/// test points are computed and used instead of the intensities of the test
/// points themselves. This is much faster because the averages come from
/// integral images. Calonder suggests that this approach may be faster, and
/// [Rublee et. al. (2012)][rublee] use this approach to quickly compute ORB
/// descriptors.
///
/// [rublee]: http://www.gwylab.com/download/ORB_2012.pdf
/// [Calonder, et. al. (2010)]: https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
pub fn brief(
    image: &GrayImage,
    keypoints: &[Point<u32>],
    length: usize,
    override_test_pairs: Option<&Vec<TestPair>>,
) -> Result<(Vec<BriefDescriptor>, Vec<TestPair>), String> {
    // if we have test pairs already, use them; otherwise, generate some
    let test_pairs = if let Some(t) = override_test_pairs {
        t.clone()
    } else {
        // generate a set of test pairs within a 31x31 grid with a Gaussian bias (sigma = 6.6)
        let test_pair_distribution = Normal::new(BRIEF_PATCH_RADIUS as f32 + 1.0, 6.6).unwrap();
        let mut rng = rand::rng();
        let mut test_pairs: Vec<TestPair> = Vec::with_capacity(length);
        while test_pairs.len() < length {
            let (x0, y0, x1, y1) = (
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
            );
            if x0 < BRIEF_PATCH_DIAMETER
                && y0 < BRIEF_PATCH_DIAMETER
                && x1 < BRIEF_PATCH_DIAMETER
                && y1 < BRIEF_PATCH_DIAMETER
            {
                test_pairs.push(TestPair {
                    p0: Point::new(x0, y0),
                    p1: Point::new(x1, y1),
                });
            }
        }
        test_pairs.clone()
    };

    let integral_image = integral_image(image);

    let descriptors = brief_impl(&integral_image, keypoints, &test_pairs, length)?;

    // return the descriptors for all the keypoints and the test pairs used
    Ok((descriptors, test_pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hamming_distance() {
        let d1 = BriefDescriptor {
            bits: vec![
                0xe41749023c71b74df05b57165519a180,
                0x9c06c422620e05d01105618cb3a2dcf1,
            ],
            corner: Corner::new(0, 0, 0.),
        };
        let d2 = BriefDescriptor {
            bits: vec![
                0x8af22bb0596edc267c6f72cf425ebe1a,
                0xc1f6291520e474e8fa114e15420413d1,
            ],
            corner: Corner::new(0, 0, 0.),
        };
        assert_eq!(d1.hamming_distance(&d2), 134);
    }

    #[test]
    fn test_get_bit_subset() {
        let d = BriefDescriptor {
            bits: vec![
                0xdbe3de5bd950adf3d730034f9e4a55f7,
                0xf275f00f6243892a18ffefd0499996ee,
            ],
            corner: Corner::new(0, 0, 0.),
        };
        let bits = vec![
            226, 38, 212, 210, 60, 205, 68, 184, 47, 105, 152, 169, 11, 39, 76, 217, 183, 113, 189,
            251, 37, 181, 62, 28, 148, 92, 251, 77, 222, 148, 56, 142,
        ];
        assert_eq!(d.get_bit_subset(&bits), 0b11001010011100011100011111011110);
    }

    #[test]
    fn test_local_pixel_average() {
        let image = gray_image!(
            186, 106,  86,  22, 191,  10, 204, 217;
             37, 188,  82,  28,  99, 110, 166, 202;
             36,  97, 176,  54, 141,  42,  44,  40;
            248, 163, 218, 204, 117, 121, 151, 135;
            138, 100,  77, 115,  93, 246, 204, 163;
            123,   1, 104,  97,  67, 208,   0, 116;
              5, 237, 254, 171, 172, 165,  50,  39;
             92,  31, 238,  88,  44,  67, 140, 255
        );
        let integral_image: Image<Luma<u32>> = integral_image(&image);
        assert_eq!(local_pixel_average(&integral_image, 3, 3, 2), 117);
    }

    // `brief()` accepts user-supplied `TestPair`s with public
    // `u32` fields and adds their coordinates into the patch top-left
    // without checking them, in `brief_impl`
    // (src/binary_descriptors/brief.rs:172-179):
    //
    //     let p0 = Point {
    //         x: test_pair.p0.x + patch_top_left.x + 1,
    //         y: test_pair.p0.y + patch_top_left.y + 1,
    //     };
    //     ...
    //     local_pixel_average(integral_image, p0.x, p0.y, 2)
    //
    // In release the additions wrap. `local_pixel_average`
    // (src/binary_descriptors/brief.rs:70-109) then does:
    //
    //     let y_min = y.saturating_sub(radius);
    //     let x_min = x.saturating_sub(radius);
    //     let y_max = u32::min(y + radius + 1, integral_image.height() - 1);
    //     let x_max = u32::min(x + radius + 1, integral_image.width() - 1);
    //     let pixel_area = (y_max - y_min) * (x_max - x_min);
    //     if pixel_area == 0 { return 0; }
    //     let (..) = unsafe {
    //         (
    //             integral_image.unsafe_get_pixel(x_max, y_max)[0],
    //             integral_image.unsafe_get_pixel(x_min, y_min)[0],
    //             ...
    //         )
    //     };
    //
    // Pick `test_pair.p0 = (u32::MAX - 5, u32::MAX - 5)` and a valid
    // keypoint at (16, 16) in a 35 × 35 image (BRIEF_PATCH_RADIUS = 15, so
    // any keypoint with `BRIEF_PATCH_RADIUS < x < width - BRIEF_PATCH_RADIUS`
    // is accepted). `patch_top_left = (1, 1)`. Then
    //     p0.x = (u32::MAX - 5) + 1 + 1 = u32::MAX - 3   (no wrap)
    //     p0.y = u32::MAX - 3
    // In `local_pixel_average(integral_image, u32::MAX - 3, u32::MAX - 3, 2)`:
    //     x_min = (u32::MAX - 3) - 2 = u32::MAX - 5      (huge)
    //     x_max = min((u32::MAX - 3) + 2 + 1, 35) = min(0, 35) = 0
    //                                                    (wraps in release)
    // With `x_max = 0` and `x_min = u32::MAX - 5`, the subtraction
    // `x_max - x_min` wraps to 6, `pixel_area = 6 * 6 = 36`, the early
    // `pixel_area == 0` return is bypassed, and `unsafe_get_pixel(x_min,
    // y_min)` reads memory far past the end of the integral image's
    // backing buffer.
    //
    // Requires `--release` to disable the overflow check on the additions.
    // -----------------------------------------------------------------------

    #[test]
    fn brief_with_adversarial_test_pair_reads_oob() {
        // 35 × 35 image is the smallest that satisfies BRIEF's "keypoint
        // must be at least 17 pixels from any edge" check.
        let img = GrayImage::from_pixel(35, 35, Luma([0u8]));

        // A perfectly valid keypoint, well away from the edge.
        let keypoints = vec![Point::new(17u32, 17u32)];

        // 128 attacker-controlled test pairs. Adding `u32::MAX - 5` to the
        // patch top-left + 1 = 2 gives `u32::MAX - 3`, which then drives
        // `local_pixel_average` past the end of the 36 × 36 integral image.
        let mut pairs = Vec::with_capacity(128);
        for _ in 0..128 {
            pairs.push(TestPair {
                p0: Point::new(u32::MAX - 5, u32::MAX - 5),
                p1: Point::new(0, 0),
            });
        }

        // Public, safe entry point — must reject the out-of-range test pairs.
        let result = brief(&img, &keypoints, 128, Some(&pairs));
        assert!(result.is_err());
    }

    fn pair(x0: u32, y0: u32, x1: u32, y1: u32) -> TestPair {
        TestPair {
            p0: Point::new(x0, y0),
            p1: Point::new(x1, y1),
        }
    }

    /// Image whose intensity increases from left to right.
    fn horizontal_gradient(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, _| Luma([(x * 6) as u8]))
    }

    /// Test pairs comparing two pixels of equal intensity, so every bit is 0.
    fn neutral_pairs(n: usize) -> Vec<TestPair> {
        vec![pair(15, 15, 15, 15); n]
    }

    #[test]
    fn test_get_size() {
        let mut d = BriefDescriptor {
            bits: vec![0],
            corner: Corner::new(0, 0, 0.),
        };
        assert_eq!(d.get_size(), 128);
        d.bits = vec![0; 4];
        assert_eq!(d.get_size(), 512);
    }

    #[test]
    fn test_position() {
        let d = BriefDescriptor {
            bits: vec![0],
            corner: Corner::new(7, 11, 0.5),
        };
        assert_eq!(d.position(), Point::new(7, 11));
    }

    #[test]
    fn test_hamming_distance_identical_and_opposite() {
        let d1 = BriefDescriptor {
            bits: vec![0x1234, u128::MAX],
            corner: Corner::new(0, 0, 0.),
        };
        let d2 = BriefDescriptor {
            bits: vec![!0x1234, 0],
            corner: Corner::new(0, 0, 0.),
        };
        assert_eq!(d1.hamming_distance(&d1), 0);
        assert_eq!(d1.hamming_distance(&d2), 256);
        assert_eq!(d2.hamming_distance(&d1), 256);
    }

    #[test]
    #[should_panic]
    fn test_hamming_distance_panics_on_unequal_sizes() {
        let d1 = BriefDescriptor {
            bits: vec![0],
            corner: Corner::new(0, 0, 0.),
        };
        let d2 = BriefDescriptor {
            bits: vec![0, 0],
            corner: Corner::new(0, 0, 0.),
        };
        d1.hamming_distance(&d2);
    }

    #[test]
    fn test_get_bit_subset_order_and_chunks() {
        let d = BriefDescriptor {
            // bit 0 and bit 127 of chunk 0 set; bit 1 of chunk 1 (index 129) set
            bits: vec![1 | (1 << 127), 0b10],
            corner: Corner::new(0, 0, 0.),
        };
        assert_eq!(d.get_bit_subset(&[]), 0);
        // first requested bit ends up as the most significant bit of the result
        assert_eq!(d.get_bit_subset(&[0, 1]), 0b10);
        assert_eq!(d.get_bit_subset(&[1, 0]), 0b01);
        assert_eq!(d.get_bit_subset(&[127, 128, 129]), 0b101);
        // repeated indices are allowed
        assert_eq!(d.get_bit_subset(&[129, 129, 2]), 0b110);
    }

    #[test]
    fn test_get_bit_subset_max_length() {
        let d = BriefDescriptor {
            bits: vec![u128::MAX],
            corner: Corner::new(0, 0, 0.),
        };
        let bits: Vec<u32> = (0..128).collect();
        assert_eq!(d.get_bit_subset(&bits), u128::MAX);
    }

    #[test]
    #[should_panic(expected = "Can't extract more than 128 bits")]
    fn test_get_bit_subset_panics_on_too_many_bits() {
        let d = BriefDescriptor {
            bits: vec![0],
            corner: Corner::new(0, 0, 0.),
        };
        d.get_bit_subset(&[0; 129]);
    }

    #[test]
    fn test_local_pixel_average_zero_radius() {
        let image = GrayImage::from_pixel(5, 5, Luma([200]));
        let integral_image: Image<Luma<u32>> = integral_image(&image);
        assert_eq!(local_pixel_average(&integral_image, 2, 2, 0), 0);
    }

    #[test]
    fn test_local_pixel_average_clipped_at_borders() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        );
        let integral_image: Image<Luma<u32>> = integral_image(&image);
        // window clipped to the top-left 2x2 block: (1 + 2 + 4 + 5) / 4
        assert_eq!(local_pixel_average(&integral_image, 0, 0, 1), 3);
        // window clipped to the bottom-right 2x2 block: (5 + 6 + 8 + 9) / 4
        assert_eq!(local_pixel_average(&integral_image, 2, 2, 1), 7);
        // window covering the whole image: 45 / 9
        assert_eq!(local_pixel_average(&integral_image, 1, 1, 10), 5);
    }

    #[test]
    fn test_local_pixel_average_uniform_image() {
        let image = GrayImage::from_pixel(10, 10, Luma([77]));
        let integral_image: Image<Luma<u32>> = integral_image(&image);
        for (x, y) in [(0, 0), (5, 5), (9, 9), (100, 100)] {
            let avg = local_pixel_average(&integral_image, x, y, 2);
            assert!(
                avg == 77 || avg == 0,
                "unexpected average {avg} at ({x}, {y})"
            );
        }
        assert_eq!(local_pixel_average(&integral_image, 5, 5, 2), 77);
    }

    #[test]
    fn test_brief_rejects_length_not_multiple_of_128() {
        let image = GrayImage::new(40, 40);
        let keypoints = [Point::new(20, 20)];
        for length in [1, 64, 129, 255] {
            let result = brief(&image, &keypoints, length, Some(&neutral_pairs(length)));
            assert!(result.err().unwrap().contains("multiple of 128"));
        }
    }

    #[test]
    fn test_brief_rejects_mismatched_test_pairs() {
        let image = GrayImage::new(40, 40);
        let keypoints = [Point::new(20, 20)];
        for n in [0, 127, 129, 256] {
            let result = brief(&image, &keypoints, 128, Some(&neutral_pairs(n)));
            assert!(result.err().unwrap().contains("number of test pairs"));
        }
    }

    #[test]
    fn test_brief_rejects_test_pair_outside_patch() {
        let image = GrayImage::new(40, 40);
        let keypoints = [Point::new(20, 20)];
        let d = BRIEF_PATCH_DIAMETER;
        for bad in [
            pair(d, 0, 0, 0),
            pair(0, d, 0, 0),
            pair(0, 0, d, 0),
            pair(0, 0, 0, d),
        ] {
            let mut pairs = neutral_pairs(128);
            pairs[42] = bad;
            let err = brief(&image, &keypoints, 128, Some(&pairs)).err().unwrap();
            assert!(err.contains("Test pair 42"), "{err}");
        }
        // the largest in-patch coordinate is accepted
        let mut pairs = neutral_pairs(128);
        pairs[0] = pair(d - 1, d - 1, d - 1, d - 1);
        assert!(brief(&image, &keypoints, 128, Some(&pairs)).is_ok());
    }

    #[test]
    fn test_brief_keypoint_edge_distance() {
        let pairs = neutral_pairs(128);
        // keypoints need 16 px to every edge, so a 32x32 image admits none
        let image = GrayImage::new(32, 32);
        assert!(brief(&image, &[Point::new(16, 16)], 128, Some(&pairs)).is_err());

        // 33x33 is the smallest image that admits a keypoint, at (16, 16)
        let image = GrayImage::new(33, 33);
        assert!(brief(&image, &[Point::new(16, 16)], 128, Some(&pairs)).is_ok());

        for kp in [
            Point::new(15, 16),
            Point::new(16, 15),
            Point::new(17, 16),
            Point::new(16, 17),
            Point::new(0, 0),
            Point::new(100, 100),
            Point::new(u32::MAX, 16),
            Point::new(16, u32::MAX),
        ] {
            let err = brief(&image, &[kp], 128, Some(&pairs)).err().unwrap();
            assert!(err.contains("image edge"), "{err}");
        }

        // a single bad keypoint fails the whole call
        let image = GrayImage::new(64, 64);
        let keypoints = [Point::new(30, 30), Point::new(5, 30)];
        assert!(brief(&image, &keypoints, 128, Some(&pairs)).is_err());
    }

    #[test]
    fn test_brief_generated_test_pairs() {
        let image = horizontal_gradient(64, 64);
        let keypoints = [Point::new(20, 20), Point::new(40, 30)];
        for length in [128, 256, 512] {
            let (descriptors, pairs) = brief(&image, &keypoints, length, None).unwrap();
            assert_eq!(pairs.len(), length);
            for tp in &pairs {
                for p in [tp.p0, tp.p1] {
                    assert!(p.x < BRIEF_PATCH_DIAMETER && p.y < BRIEF_PATCH_DIAMETER);
                }
            }
            assert_eq!(descriptors.len(), keypoints.len());
            for (d, kp) in descriptors.iter().zip(keypoints) {
                assert_eq!(d.get_size() as usize, length);
                assert_eq!(d.position(), kp);
            }

            // reusing the returned test pairs reproduces the descriptors
            let (again, same_pairs) = brief(&image, &keypoints, length, Some(&pairs)).unwrap();
            assert!(again == descriptors);
            assert_eq!(same_pairs, pairs);
        }
    }

    #[test]
    fn test_brief_no_keypoints() {
        let image = GrayImage::new(40, 40);
        let (descriptors, pairs) = brief(&image, &[], 128, None).unwrap();
        assert!(descriptors.is_empty());
        assert_eq!(pairs.len(), 128);
    }

    #[test]
    fn test_brief_uniform_image_has_no_set_bits() {
        let image = GrayImage::from_pixel(50, 50, Luma([128]));
        let keypoints = [Point::new(20, 20), Point::new(25, 30)];
        let (descriptors, _) = brief(&image, &keypoints, 256, None).unwrap();
        for d in descriptors {
            assert_eq!(d.bits, vec![0, 0]);
        }
    }

    #[test]
    fn test_brief_intensity_comparison() {
        let image = horizontal_gradient(40, 40);
        let keypoints = [Point::new(20, 20)];

        // p0 darker than p1 sets a bit
        let mut pairs = neutral_pairs(128);
        pairs[3] = pair(5, 15, 25, 15);
        let (d, _) = brief(&image, &keypoints, 128, Some(&pairs)).unwrap();
        assert_eq!(d[0].bits[0].count_ones(), 1);

        // p0 brighter than p1 does not
        pairs[3] = pair(25, 15, 5, 15);
        let (d, _) = brief(&image, &keypoints, 128, Some(&pairs)).unwrap();
        assert_eq!(d[0].bits[0], 0);
    }

    #[test]
    fn test_brief_every_test_pair_sets_a_distinct_bit() {
        let image = horizontal_gradient(40, 40);
        let keypoints = [Point::new(20, 20)];
        let length = 256;
        let mut seen = std::collections::HashSet::new();
        for i in 0..length {
            let mut pairs = neutral_pairs(length);
            pairs[i] = pair(5, 15, 25, 15);
            let (d, _) = brief(&image, &keypoints, length, Some(&pairs)).unwrap();
            let set_bits: Vec<u32> = (0..length as u32)
                .filter(|&b| d[0].get_bit_subset(&[b]) == 1)
                .collect();
            assert_eq!(set_bits.len(), 1, "test pair {i} set bits {set_bits:?}");
            assert!(seen.insert(set_bits[0]), "bit {} reused", set_bits[0]);
        }
    }

    #[test]
    fn test_brief_patch_is_centred_on_keypoint() {
        // Intensity grows with the distance from column 20 and row 20, so the
        // image is symmetric around the keypoint (20, 20). Test points that
        // mirror each other through the patch centre must compare equal.
        let image = GrayImage::from_fn(41, 41, |x, y| {
            Luma([(x.abs_diff(20) * 6 + y.abs_diff(20) * 2) as u8])
        });
        let keypoints = [Point::new(20, 20)];
        let c = BRIEF_PATCH_RADIUS;
        for d in [3, 6, 10] {
            for p in [
                pair(c - d, c, c + d, c),
                pair(c + d, c, c - d, c),
                pair(c, c - d, c, c + d),
                pair(c, c + d, c, c - d),
            ] {
                let mut pairs = neutral_pairs(128);
                pairs[0] = p;
                let (desc, _) = brief(&image, &keypoints, 128, Some(&pairs)).unwrap();
                assert_eq!(desc[0].bits, vec![0], "{p:?}");
            }
        }
    }

    #[test]
    fn test_brief_translation_invariance() {
        // the same pattern placed at two locations yields identical descriptors
        let pattern = horizontal_gradient(40, 40);
        let mut image = GrayImage::new(120, 60);
        image::imageops::replace(&mut image, &pattern, 0, 0);
        image::imageops::replace(&mut image, &pattern, 70, 10);

        let keypoints = [Point::new(20, 20), Point::new(90, 30)];
        let (d, _) = brief(&image, &keypoints, 256, None).unwrap();
        assert_eq!(d[0].hamming_distance(&d[1]), 0);
        assert!(d[0].bits.iter().any(|&c| c != 0));
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use rand::RngExt;
    use test::{Bencher, black_box};

    #[bench]
    #[ignore]
    fn bench_brief_random_test_pairs_1000_keypoints(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        let mut rng = rand::rng();
        let keypoints = (0..1000)
            .map(|_| {
                Point::new(
                    rng.random_range(24..image.width() - 24),
                    rng.random_range(24..image.height() - 24),
                )
            })
            .collect::<Vec<Point<u32>>>();
        b.iter(|| {
            black_box(brief(&image, &keypoints, 256, None)).unwrap();
        })
    }

    #[bench]
    #[ignore]
    fn bench_brief_fixed_test_pairs_1000_keypoints(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        let mut rng = rand::rng();
        let keypoints = (0..1000)
            .map(|_| {
                Point::new(
                    rng.random_range(24..image.width() - 24),
                    rng.random_range(24..image.height() - 24),
                )
            })
            .collect::<Vec<Point<u32>>>();
        let (_, test_pairs) = brief(&image, &keypoints, 256, None).unwrap();
        b.iter(|| {
            black_box(brief(&image, &keypoints, 256, Some(&test_pairs))).unwrap();
        })
    }
}
