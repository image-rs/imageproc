use super::Bits64;
use crate::definitions::Image;
use image::{imageops, Luma};

/// Stores the result of [`average_hash`].
/// Implements [`Hash`] trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AverageHash(Bits64);

impl AverageHash {
    /// Compute the [hamming distance] between hashes.
    ///
    /// [hamming distance]: https://en.wikipedia.org/wiki/Hamming_distance
    pub fn hamming_distance(self, AverageHash(other): AverageHash) -> u32 {
        self.0.hamming_distance(other)
    }
}

/// Compute the [average hash] of a grayscale image.
pub fn average_hash(img: &Image<Luma<f32>>) -> AverageHash {
    const HASH_SIZE: u32 = 8;
    let resized = imageops::resize(img, HASH_SIZE, HASH_SIZE, imageops::FilterType::Lanczos3);
    let num_pixels = (HASH_SIZE * HASH_SIZE) as usize;
    debug_assert_eq!(num_pixels, resized.len() as usize);

    let sum: f32 = resized.pixels().map(|p| p[0]).sum();
    let mean = sum / num_pixels as f32;
    let bits = resized.pixels().map(|p| p[0] > mean);
    AverageHash(Bits64::new(bits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average_hash_self_distance_zero() {
        let img = gray_image!(type: f32,
            0., 1., 2.;
            3., 4., 5.;
            6., 7., 8.
        );
        let hash = average_hash(&img);
        assert_eq!(0, hash.hamming_distance(hash));
    }

    #[test]
    fn test_average_hash_distance_non_zero() {
        let img1 = gray_image!(type: f32,
            0., 1., 2.;
            3., 4., 5.;
            6., 7., 255.
        );
        let mut img2 = img1.clone();
        *img2.get_pixel_mut(2, 2) = Luma([0.]);

        let hash1 = average_hash(&img1);
        let hash2 = average_hash(&img2);

        assert_eq!(hash1.hamming_distance(hash2), hash2.hamming_distance(hash1));
        assert!(hash1.hamming_distance(hash2) > 0);
    }
}
