//! Functions for generating compact binary patch descriptions.

use image::{GenericImageView, GrayImage};
use rand_distr::{Distribution, Normal};

use crate::filter::gaussian_blur_f32;

/// A thin wrapper around a vector of bits
#[derive(Debug)]
pub struct BinaryDescriptor(Vec<bool>);

impl BinaryDescriptor {
    /// Returns the length of the descriptor in bits. Typical values are 128,
    /// 256, and 512.
    pub fn get_size(&self) -> u32 {
        self.0.len() as u32
    }
    /// Returns the number of bits that are different between the two descriptors.
    ///
    /// Panics if the two descriptors have unequal lengths. The descriptors
    /// should have been computed using the same set of test pairs, otherwise
    /// comparing them has no meaning.
    pub fn compute_hamming_distance(&self, other: &Self) -> u32 {
        assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0, |acc, x| acc + (x.0 ^ x.1) as u32)
    }
}

/// Collection of two points that a BRIEF descriptor uses to generate its bits.
#[derive(Debug)]
pub struct TestPair {
    /// The first point in the pair.
    pub p0: (u32, u32),
    /// The second point in the pair.
    pub p1: (u32, u32),
}

/// Generates BRIEF descriptors for small patches around keypoints in an image.
///
/// Returns a tuple containing a vector of `Option<BinaryDescriptor>` and a
/// vector of `TestPair`. All returned descriptors are based on the same
/// `TestPair` set. Patches are 33x33 pixels, so keypoints must be at least 17
/// pixels from any edge. If any keypoints are too close to an edge, their
/// corresponding element in the descriptor vector is `None`.
///
/// If `override_test_pairs` is `Some`, then those test pairs are used, and none
/// are generated. Use this when you already have test pairs from another run
/// and want to compare the descriptors later.
///
/// If `override_test_pairs` is `None`, then `TestPair`s are generated according
/// to an isotropic Gaussian.
///
/// Before testing, patches are smoothed with a 9x9 Gaussian.
///
/// See [Calonder, et. al. (2010)][https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf]
pub fn brief(
    image: &GrayImage,
    keypoints: &[(u32, u32)],
    length: usize,
    override_test_pairs: Option<Vec<TestPair>>,
) -> (Vec<Option<BinaryDescriptor>>, Vec<TestPair>) {
    const PATCH_RADIUS: u32 = 16;
    const PATCH_DIAMETER: u32 = PATCH_RADIUS * 2 + 1;

    let mut descriptors: Vec<Option<BinaryDescriptor>> = Vec::with_capacity(keypoints.len());

    // if we have test pairs already, use them; otherwise, generate some
    let test_pairs = if let Some(t) = override_test_pairs {
        t
    } else {
        // generate a set of test pairs within a 33x33 grid with a Gaussian bias (sigma = 6.6)
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
        test_pairs
    };

    for keypoint in keypoints {
        // if the keypoint is too close to the edge, record None
        let (width, height) = (image.width(), image.height());
        if keypoint.0 <= PATCH_RADIUS
            || keypoint.0 >= width - PATCH_RADIUS
            || keypoint.1 <= PATCH_RADIUS
            || keypoint.1 >= height - PATCH_RADIUS
        {
            descriptors.push(None);
            continue;
        }
        // otherwise, grab a 33x33 patch around the keypoint
        let patch = image.view(
            keypoint.0 - (PATCH_RADIUS + 1),
            keypoint.1 - (PATCH_RADIUS + 1),
            PATCH_DIAMETER,
            PATCH_DIAMETER,
        );
        // apply a Gaussian blur to the patch
        let blurred_patch = gaussian_blur_f32(&patch.to_image(), 4.5);

        let mut descriptor = BinaryDescriptor(Vec::with_capacity(length));
        // for each test pair, compare the pixels within the patch at those points
        for test_pair in &test_pairs {
            // if p0 < p1, then record true for this test; otherwise, record false
            let (p0, p1) = (test_pair.p0, test_pair.p1);
            descriptor.0.push(
                blurred_patch.get_pixel(p0.0, p0.1)[0] < blurred_patch.get_pixel(p1.0, p1.1)[0],
            );
        }
        descriptors.push(Some(descriptor));
    }

    // return the descriptors for all the keypoints and the test pairs used
    (descriptors, test_pairs)
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
    d1: &[Option<BinaryDescriptor>],
    d2: &[Option<BinaryDescriptor>],
    threshold: u32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::with_capacity(d1.len());
    for (d_a_idx, d_a) in d1.iter().enumerate() {
        if d_a.is_none() {
            continue;
        }
        let mut best = (u32::MAX, (0usize, 0usize));
        for (d_b_idx, d_b) in d2.iter().enumerate() {
            if d_b.is_none() {
                continue;
            }
            let distance = d_a
                .as_ref()
                .unwrap()
                .compute_hamming_distance(d_b.as_ref().unwrap());
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
