//! Structs and functions for finding and computing BRIEF descriptors as
//! described in [Calonder, et. al. (2010)].
///
/// [Calonder, et. al. (2010)]: https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf
use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use rand_distr::{Distribution, Normal};

use crate::{corners::Corner, integral_image::integral_image, point::Point};

use super::{
    constants::{BRIEF_PATCH_DIAMETER, BRIEF_PATCH_RADIUS},
    BinaryDescriptor,
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

fn local_pixel_average(
    integral_image: &ImageBuffer<Luma<u32>, Vec<u32>>,
    x: u32,
    y: u32,
    radius: u32,
) -> u8 {
    if radius == 0 {
        return 0;
    }
    let y_min = if y < radius { 0 } else { y - radius };
    let x_min = if x < radius { 0 } else { x - radius };
    let y_max = u32::min(y + radius + 1, integral_image.height() - 1);
    let x_max = u32::min(x + radius + 1, integral_image.width() - 1);

    let pixel_area = (y_max - y_min) * (x_max - x_min);
    if pixel_area == 0 {
        return 0;
    }

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
    integral_image: &ImageBuffer<Luma<u32>, Vec<u32>>,
    keypoints: &[Point<u32>],
    test_pairs: &[TestPair],
    length: usize,
) -> Result<Vec<BriefDescriptor>, String> {
    if length % 128 != 0 {
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

    let mut descriptors: Vec<BriefDescriptor> = Vec::with_capacity(keypoints.len());
    let (width, height) = (integral_image.width(), integral_image.height());

    for keypoint in keypoints {
        // if the keypoint is too close to the edge, return an error
        if keypoint.x <= BRIEF_PATCH_RADIUS
            || keypoint.x + BRIEF_PATCH_RADIUS >= width
            || keypoint.y <= BRIEF_PATCH_RADIUS
            || keypoint.y + BRIEF_PATCH_RADIUS >= height
        {
            return Err(format!(
                "Found keypoint within {} px of image edge: ({}, {})",
                BRIEF_PATCH_RADIUS + 1,
                keypoint.x,
                keypoint.y
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
                x: test_pair.p0.x + patch_top_left.x + 1,
                y: test_pair.p0.y + patch_top_left.y + 1,
            };
            let p1 = Point {
                x: test_pair.p1.x + patch_top_left.x + 1,
                y: test_pair.p1.y + patch_top_left.y + 1,
            };

            // if p0 < p1, then record true for this test; otherwise, record false
            descriptor_chunk += (local_pixel_average(integral_image, p0.x, p0.y, 2)
                < local_pixel_average(integral_image, p1.x, p1.y, 2))
                as u128;
            descriptor_chunk <<= 1;
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
/// Patches are 31x31 pixels, so keypoints must be at least 17 pixels from any
/// edge. If any keypoints are too close to an edge, returns an error.
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
        let mut rng = rand::thread_rng();
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
        let integral_image: ImageBuffer<Luma<u32>, Vec<u32>> = integral_image(&image);
        assert_eq!(local_pixel_average(&integral_image, 3, 3, 2), 117);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use rand::Rng;
    use test::{black_box, Bencher};

    #[bench]
    #[ignore]
    fn bench_brief_random_test_pairs_1000_keypoints(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        let mut rng = rand::thread_rng();
        let keypoints = (0..1000)
            .map(|_| {
                Point::new(
                    rng.gen_range(24..image.width() - 24),
                    rng.gen_range(24..image.height() - 24),
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
        let mut rng = rand::thread_rng();
        let keypoints = (0..1000)
            .map(|_| {
                Point::new(
                    rng.gen_range(24..image.width() - 24),
                    rng.gen_range(24..image.height() - 24),
                )
            })
            .collect::<Vec<Point<u32>>>();
        let (_, test_pairs) = brief(&image, &keypoints, 256, None).unwrap();
        b.iter(|| {
            black_box(brief(&image, &keypoints, 256, Some(&test_pairs))).unwrap();
        })
    }
}
