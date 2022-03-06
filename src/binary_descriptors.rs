//! Functions for generating compact binary patch descriptions.

use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use rand_distr::{Distribution, Normal};

use crate::integral_image::{integral_image, sum_image_pixels};

/// A thin wrapper around a vector of bits
#[derive(Debug, Clone)]
pub struct BinaryDescriptor(Vec<u128>);

impl BinaryDescriptor {
    /// Returns the length of the descriptor in bits. Typical values are 128,
    /// 256, and 512.
    pub fn get_size(&self) -> u32 {
        (self.0.len() * 128) as u32
    }
    /// Returns the number of bits that are different between the two descriptors.
    ///
    /// Panics if the two descriptors have unequal lengths. The descriptors
    /// should have been computed using the same set of test pairs, otherwise
    /// comparing them has no meaning.
    pub fn compute_hamming_distance(&self, other: &Self) -> u32 {
        assert_eq!(self.get_size(), other.get_size());
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0, |acc, x| acc + (x.0 ^ x.1).count_ones())
    }
}

/// Collection of two points that a BRIEF descriptor uses to generate its bits.
#[derive(Debug, Clone)]
pub struct TestPair {
    /// The first point in the pair.
    pub p0: (u32, u32),
    /// The second point in the pair.
    pub p1: (u32, u32),
}

/// Generates BRIEF descriptors for small patches around keypoints in an image.
///
/// Returns a tuple containing a vector of `BinaryDescriptor` and a vector of
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
/// Before testing, patches are smoothed with a 9x9 Gaussian approximated by
/// three box filters.
///
/// See [Calonder, et. al. (2010)][https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf]
pub fn brief(
    image: &GrayImage,
    keypoints: &[(u32, u32)],
    length: usize,
    override_test_pairs: Option<&Vec<TestPair>>,
) -> Result<(Vec<BinaryDescriptor>, Vec<TestPair>), String> {
    if length % 128 != 0 {
        return Err(format!(
            "BRIEF descriptor length must be a multiple of 128 bits (found {})",
            length
        ));
    }

    const PATCH_RADIUS: u32 = 15;
    const PATCH_DIAMETER: u32 = PATCH_RADIUS * 2 + 1;
    const SUB_PATCH_RADIUS: u32 = 5;

    let mut descriptors: Vec<BinaryDescriptor> = Vec::with_capacity(keypoints.len());

    // if we have test pairs already, use them; otherwise, generate some
    let test_pairs = if let Some(t) = override_test_pairs {
        t.clone()
    } else {
        // generate a set of test pairs within a 31x31 grid with a Gaussian bias (sigma = 6.6)
        let test_pair_distribution = Normal::new(PATCH_RADIUS as f32 + 1.0, 6.6).unwrap();
        let mut rng = rand::thread_rng();
        let mut test_pairs: Vec<TestPair> = Vec::with_capacity(length);
        while test_pairs.len() < length {
            let (x0, y0, x1, y1) = (
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
                test_pair_distribution.sample(&mut rng) as u32,
            );
            if x0 < PATCH_DIAMETER
                && y0 < PATCH_DIAMETER
                && x1 < PATCH_DIAMETER
                && y1 < PATCH_DIAMETER
            {
                test_pairs.push(TestPair {
                    p0: (x0, y0),
                    p1: (x1, y1),
                });
            }
        }
        test_pairs.clone()
    };

    for keypoint in keypoints {
        // if the keypoint is too close to the edge, return an error
        let (width, height) = (image.width(), image.height());
        if keypoint.0 <= PATCH_RADIUS
            || keypoint.0 + PATCH_RADIUS >= width
            || keypoint.1 <= PATCH_RADIUS
            || keypoint.1 + PATCH_RADIUS >= height
        {
            return Err(format!(
                "Found keypoint within {}px of image edge: ({}, {})",
                PATCH_RADIUS + 1,
                keypoint.0,
                keypoint.1
            ));
        }
        // otherwise, grab a 31x31 patch around the keypoint
        let patch = image.view(
            keypoint.0 - (PATCH_RADIUS + 1),
            keypoint.1 - (PATCH_RADIUS + 1),
            PATCH_DIAMETER,
            PATCH_DIAMETER,
        );

        // the original paper applies a Gaussian blur to the patch, but for
        // speed we will use an integral image to find the average intensities
        // around each test point
        let integral_patch: ImageBuffer<Luma<u32>, Vec<u32>> = integral_image(&patch.to_image());

        let mut descriptor = BinaryDescriptor(Vec::with_capacity(length / 128));
        let mut descriptor_chunk = 0u128;
        // for each test pair, compare the pixels within the patch at those points
        for (idx, test_pair) in test_pairs.iter().enumerate() {
            // if we've entered a new chunk, then save the previous one
            if idx != 0 && idx % 128 == 0 {
                descriptor.0.push(descriptor_chunk);
                descriptor_chunk = 0;
            }

            let (p0, p1) = (test_pair.p0, test_pair.p1);

            // check bounds for the sub-patch around the test point
            let p0_left = if SUB_PATCH_RADIUS >= p0.0 {
                0
            } else {
                p0.0 - SUB_PATCH_RADIUS
            };
            let p0_right = if p0.0 + SUB_PATCH_RADIUS >= PATCH_DIAMETER {
                PATCH_DIAMETER - 1
            } else {
                p0.0 + SUB_PATCH_RADIUS - 1
            };
            let p0_top = if SUB_PATCH_RADIUS >= p0.1 {
                0
            } else {
                p0.1 - SUB_PATCH_RADIUS
            };
            let p0_bottom = if p0.1 + SUB_PATCH_RADIUS >= PATCH_DIAMETER {
                PATCH_DIAMETER - 1
            } else {
                p0.1 + SUB_PATCH_RADIUS - 1
            };
            let p1_left = if SUB_PATCH_RADIUS >= p1.0 {
                0
            } else {
                p1.0 - SUB_PATCH_RADIUS
            };
            let p1_right = if p1.0 + SUB_PATCH_RADIUS >= PATCH_DIAMETER {
                PATCH_DIAMETER - 1
            } else {
                p1.0 + SUB_PATCH_RADIUS - 1
            };
            let p1_top = if SUB_PATCH_RADIUS >= p1.1 {
                0
            } else {
                p1.1 - SUB_PATCH_RADIUS
            };
            let p1_bottom = if p1.1 + SUB_PATCH_RADIUS >= PATCH_DIAMETER {
                PATCH_DIAMETER - 1
            } else {
                p1.1 + SUB_PATCH_RADIUS - 1
            };

            // use the integral image to compute the average intensity for the test point sub-patches
            let p0_total_intensity =
                sum_image_pixels(&integral_patch, p0_left, p0_top, p0_right, p0_bottom);
            let p0_avg_intensity =
                p0_total_intensity[0] / ((p0_bottom - p0_top) * (p0_right - p0_left));
            let p1_total_intensity =
                sum_image_pixels(&integral_patch, p1_left, p1_top, p1_right, p1_bottom);
            let p1_avg_intensity =
                p1_total_intensity[0] / ((p1_bottom - p1_top) * (p1_right - p1_left));

            // if p0 < p1, then record true for this test; otherwise, record false
            descriptor_chunk += (p0_avg_intensity < p1_avg_intensity) as u128;
            descriptor_chunk <<= 1;
        }
        // save the final chunk too
        descriptor.0.push(descriptor_chunk);
        descriptors.push(descriptor);
    }

    // return the descriptors for all the keypoints and the test pairs used
    Ok((descriptors, test_pairs))
}

/// For each descriptor in `d1`, find the descriptor in `d2` with the minimum
/// Hamming distance below `threshold`. If no such descriptor exists in `d2`,
/// the descriptor in `d1` is left unmatched.
///
/// Descriptors in `d2` may be matched with more than one descriptor in `d1`.
///
/// Returns a vector of tuples describing the matched pairs. The first value is
/// an index into `d1`, and the second value is an index into `d2`.
pub fn match_binary_descriptors(
    d1: &[BinaryDescriptor],
    d2: &[BinaryDescriptor],
    threshold: u32,
) -> Vec<(usize, usize)> {
    // let m = 8; // d1[0].get_size() / log_2 (d2.len())
    // let mut substring_tables: Vec<HashMap<BinaryDescriptor, usize>> = vec![HashMap::new(); m];
    // for j in 0..m {
    //     for (i, d2_el) in d2.iter().enumerate() {

    //     }
    // }
    let mut matches = Vec::with_capacity(d2.len());
    // perform linear scan to find the best match
    for (d_a_idx, d_a) in d1.iter().enumerate() {
        let mut best = (u32::MAX, (0usize, 0usize));
        for (d_b_idx, d_b) in d2.iter().enumerate() {
            let distance = d_a.compute_hamming_distance(d_b);
            if distance < best.0 {
                best.0 = distance;
                best.1 = (d_a_idx, d_b_idx);
            }
        }
        if best.0 < threshold {
            matches.push(best.1);
        }
    }
    matches
}
