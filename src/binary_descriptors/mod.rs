//! Functions for generating and comparing compact binary patch descriptors.

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;

use crate::point::Point;

pub mod brief;
mod constants;

/// A feature descriptor whose value is given by a string of bits.
pub trait BinaryDescriptor {
    /// Returns the length of the descriptor in bits. Typical values are 128,
    /// 256, and 512. Will always be a multiple of 128.
    fn get_size(&self) -> u32;
    /// Returns the number of bits that are different between the two descriptors.
    ///
    /// Panics if the two descriptors have unequal lengths. The descriptors
    /// should have been computed using the same set of test pairs, otherwise
    /// comparing them has no meaning.
    fn hamming_distance(&self, other: &Self) -> u32;
    /// Given a set of bit indices, returns those bits from the descriptor as a
    /// single concatenated value.
    ///
    /// Panics if `bits.len() > 128`.
    fn get_bit_subset(&self, bits: &[u32]) -> u128;
    /// Returns the pixel location of this binary descriptor in its associated
    /// image.
    fn position(&self) -> Point<u32>;
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
/// Returns a vector of references describing the matched pairs. The first
/// reference is to a descriptor in `d1`, and the second reference is to an
/// index into `d2`.
///
/// [lsh]:
///     https://en.wikipedia.org/wiki/Locality_sensitive_hashing#Bit_sampling_for_Hamming_distance
pub fn match_binary_descriptors<'a, T: BinaryDescriptor>(
    d1: &'a [T],
    d2: &'a [T],
    threshold: u32,
    seed: Option<u64>,
) -> Vec<(&'a T, &'a T)> {
    // early return if either input is empty
    if d1.is_empty() || d2.is_empty() {
        return Vec::new();
    }

    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s)
    } else {
        StdRng::from_entropy()
    };

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
            .map(|_| rng.gen_range(0..queries[0].get_size()))
            .collect::<Vec<u32>>();

        let mut new_hashmap = HashMap::<u128, Vec<&T>>::with_capacity(database.len());

        // compute the hash of each descriptor in the database and store its index
        // there will be collisions --- we want that to happen
        for d in database.iter() {
            let hash = d.get_bit_subset(&bits);
            if let Some(v) = new_hashmap.get_mut(&hash) {
                v.push(d);
            } else {
                new_hashmap.insert(hash, vec![d]);
            }
        }
        hash_tables.push((bits, new_hashmap));
    }

    // find the hash buckets corresponding to each query descriptor
    // then check all bucket members to find the (probable) best match
    let mut matches = Vec::with_capacity(queries.len());
    for query in queries.iter() {
        // find all buckets for the query descriptor
        let mut candidates = Vec::with_capacity(l);
        for (bits, table) in hash_tables.iter() {
            let query_hash = query.get_bit_subset(bits);
            if let Some(m) = table.get(&query_hash) {
                for new_candidate in m.clone() {
                    candidates.push(new_candidate);
                }
            }
        }
        // perform linear scan to find the best match
        let mut best_score = u32::MAX;
        let mut best_candidate = None;
        for c in candidates {
            let distance = query.hamming_distance(c);
            if distance < best_score {
                best_score = distance;
                best_candidate = Some(c);
            }
        }
        // ignore the match if it's beyond our threshold
        if best_score < threshold {
            if swapped {
                matches.push((best_candidate.unwrap(), query));
            } else {
                matches.push((query, best_candidate.unwrap()));
            }
        }
    }
    matches
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::{binary_descriptors::brief::brief, utils::gray_bench_image};
    use test::{black_box, Bencher};

    #[bench]
    #[ignore]
    fn bench_matcher_1000_keypoints_each(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        let mut rng = rand::thread_rng();
        let keypoints = (0..1000)
            .map(|_| {
                Point::new(
                    rng.gen_range(20..image.width() - 20),
                    rng.gen_range(20..image.height() - 20),
                )
            })
            .collect::<Vec<Point<u32>>>();
        let (first_descriptors, test_pairs) = brief(&image, &keypoints, 256, None).unwrap();
        let (second_descriptors, _) = brief(&image, &keypoints, 256, Some(&test_pairs)).unwrap();
        b.iter(|| {
            black_box(match_binary_descriptors(
                &first_descriptors,
                &second_descriptors,
                24,
                Some(0xc0),
            ));
        });
    }
}
