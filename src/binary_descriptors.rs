//! Functions for generating and comparing compact binary patch descriptions.

use image::{GenericImageView, GrayImage, ImageBuffer, Luma};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash, Hasher},
};

use crate::integral_image::{integral_image, sum_image_pixels};

/// A thin wrapper around a vector of bits
#[derive(Debug, Clone)]
pub struct BinaryDescriptor(Vec<u128>);

impl BinaryDescriptor {
    /// Returns the length of the descriptor in bits. Typical values are 128,
    /// 256, and 512. Will always be a multiple of 128.
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

impl Hash for BinaryDescriptor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for chunk in self.0.iter() {
            state.write_u128(*chunk);
        }
    }
}

#[derive(Clone)]
struct LocalityHasher {
    bits: Vec<u32>,
    bit_index: usize,
    index: u32,
    state: u64,
}

impl LocalityHasher {
    fn new(bits: &[u32]) -> Self {
        // sorting lets us check the bits in order, which is more efficient than
        // having to call self.bits.contains() all the time in Hasher::write()
        let mut sorted_bits = bits.to_vec();
        sorted_bits.sort_unstable();

        LocalityHasher {
            bits: sorted_bits,
            bit_index: 0,
            index: 0,
            state: 0,
        }
    }
}

impl BuildHasher for LocalityHasher {
    type Hasher = LocalityHasher;
    fn build_hasher(&self) -> Self::Hasher {
        self.clone()
    }
}

impl Hasher for LocalityHasher {
    fn finish(&self) -> u64 {
        self.state
    }
    fn write(&mut self, bytes: &[u8]) {
        // extract the bits we want (given by the indices in self.bits) and
        // concatenate them into the hash output
        for b in bytes {
            for i in 0..8 {
                if self.bits[self.bit_index] == self.index + i {
                    self.state <<= 1;
                    // check whether the ith bit of this byte is set
                    // if it is, then increment the hasher state
                    // this value gets shifted left on the next loop
                    self.state += ((*b as u32 & u32::pow(2, i)) >> i) as u64;
                    if self.bit_index < self.bits.len() - 1 {
                        self.bit_index += 1;
                    }
                }
            }
            self.index += 8;
        }
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
/// Calonder used Gaussian smoothing to decrease the effects of noise in the
/// patches. This is slow, even with a box filter approximation. For maximum
/// performance, the average intensities of sub-patches of radius 5 around the
/// test points are computed and used instead of the intensities of the test
/// points themselves. This is much faster because the averages come from
/// integral images. Calonder suggests that this approach may be faster, and
/// [Rublee et. al. (2012)][rublee] use this approach to quickly compute ORB
/// descriptors.
///
/// [rublee]: http://www.gwylab.com/download/ORB_2012.pdf
///
/// See [Calonder et. al. (2010)][https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf]
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
/// Uses [locality-sensitive hashing][lsh] (LSH) for efficient matching. The
/// number of tables is fixed at three, but the hash length is proportional to
/// the log of the size of the largest input array.
///
/// Returns a vector of tuples describing the matched pairs. The first value is
/// an index into `d1`, and the second value is an index into `d2`.
///
/// [lsh]: https://en.wikipedia.org/wiki/Locality_sensitive_hashing#Bit_sampling_for_Hamming_distance
pub fn match_binary_descriptors(
    d1: &[BinaryDescriptor],
    d2: &[BinaryDescriptor],
    threshold: u32,
) -> Vec<(usize, usize)> {
    // early return if either input is empty
    if d1.is_empty() || d2.is_empty() {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();

    // locality-sensitive hashing (LSH)
    // this algorithm is log(d2.len()) but linear in d1.len(), so swap the inputs if needed
    let (queries, database, swapped) = if d1.len() > d2.len() {
        (d2, d1, true)
    } else {
        (d1, d2, false)
    };

    // build l hash tables by selecting k random bits from each descriptor
    let l = 3;
    // k grows as the log of the database size
    // this keeps bucket size roughly constant
    let k = (database.len() as f32).log2() as i32;
    let mut hash_tables = Vec::with_capacity(l);
    for _ in 0..l {
        // choose k random bits (not necessarily unique)
        let bits = (0..k)
            .into_iter()
            .map(|_| rng.gen_range(0, queries[0].get_size()))
            .collect::<Vec<u32>>();

        let mut new_hashmap = HashMap::<u64, Vec<usize>>::with_capacity(database.len());

        // compute the hash of each descriptor in the database and store its index
        // there will be collisions --- we want that to happen
        for (idx, d) in database.iter().enumerate() {
            let mut hasher = LocalityHasher::new(&bits).build_hasher();
            d.hash(&mut hasher);
            let hash = hasher.finish();
            if let Some(v) = new_hashmap.get_mut(&hash) {
                v.push(idx);
            } else {
                new_hashmap.insert(hash, vec![idx]);
            }
        }
        hash_tables.push((bits, new_hashmap));
    }

    // find the hash buckets corresponding to each query descriptor
    // then check all bucket members to find the (probable) best match
    let mut matches = Vec::with_capacity(queries.len());
    for (query_idx, query) in queries.iter().enumerate() {
        // find all buckets for the query descriptor
        let mut candidates = Vec::with_capacity(l);
        for (bits, table) in hash_tables.iter() {
            let mut hasher = LocalityHasher::new(bits).build_hasher();
            query.hash(&mut hasher);
            let query_hash = hasher.finish();
            if let Some(m) = table.get(&query_hash) {
                candidates.append(&mut m.clone());
            }
        }
        // perform linear scan to find the best match
        let mut best = (u32::MAX, 0usize);
        for c in candidates {
            let distance = query.compute_hamming_distance(&database[c]);
            if distance < best.0 {
                best.0 = distance;
                best.1 = c;
            }
        }
        // ignore the match if it's beyond our threshold
        if best.0 < threshold {
            if swapped {
                matches.push((best.1, query_idx));
            } else {
                matches.push((query_idx, best.1));
            }
        }
    }
    matches
}
