use super::{signals, Bits64};
use crate::definitions::Image;
use image::{imageops, math::Rect, Luma};
use std::borrow::Cow;

/// Stores the result of the [`phash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash(Bits64);

impl PHash {
    /// Compute the [hamming distance] between hashes.
    ///
    /// [hamming distance]: https://en.wikipedia.org/wiki/Hamming_distance
    pub fn hamming_distance(self, PHash(other): PHash) -> u32 {
        self.0.hamming_distance(other)
    }
}

/// Compute the [pHash] using [DCT].
///
/// # Example
///
/// ```
/// use imageproc::imghash;
///
/// # fn main() {
/// let img1 = image::open("first.png").to_luma32f();
/// let img2 = image::open("second.png").to_luma32f();
/// let hash1 = imghash::phash(&img1);
/// let hash2 = imghash::phash(&img2);
/// let dist = hash1.hamming_distance(hash2);
/// dbg!(dist);
/// # }
/// ```
///
/// [pHash]: phash.org/docs/pubs/thesis_zauner.pdf
/// [DCT]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
pub fn phash(img: &Image<Luma<f32>>) -> PHash {
    const N: u32 = 8;
    const HASH_FACTOR: u32 = 4;
    let img = imageops::resize(
        img,
        HASH_FACTOR * N,
        HASH_FACTOR * N,
        imageops::FilterType::Lanczos3,
    );
    let dct = signals::dct2(Cow::Owned(img));
    let topleft = Rect {
        x: 1,
        y: 1,
        width: N,
        height: N,
    };
    let topleft_dct = crate::compose::crop(&dct, topleft);
    debug_assert_eq!(topleft_dct.dimensions(), (N, N));
    assert_eq!(topleft_dct.len(), (N * N) as usize);
    let mean =
        topleft_dct.iter().copied().reduce(|a, b| a + b).unwrap() / (topleft_dct.len() as f32);
    let bits = topleft_dct.iter().map(|&x| x > mean);
    PHash(Bits64::new(bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_phash() {
        let img1 = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.
        );
        let mut img2 = img1.clone();
        *img2.get_pixel_mut(0, 0) = Luma([0f32]);
        let mut img3 = img2.clone();
        *img3.get_pixel_mut(0, 1) = Luma([0f32]);

        let hash1 = phash(&img1);
        let hash2 = phash(&img2);
        let hash3 = phash(&img3);

        assert_eq!(0, hash1.hamming_distance(hash1));
        assert_eq!(0, hash2.hamming_distance(hash2));
        assert_eq!(0, hash3.hamming_distance(hash3));

        assert_eq!(hash1.hamming_distance(hash2), hash2.hamming_distance(hash1));

        assert!(hash1.hamming_distance(hash2) > 0);
        assert!(hash1.hamming_distance(hash3) > 0);
        assert!(hash2.hamming_distance(hash3) > 0);

        assert!(hash1.hamming_distance(hash2) < hash1.hamming_distance(hash3));
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::proptest_utils::arbitrary_image;
    use proptest::prelude::*;

    const N: usize = 100;

    proptest! {
        #[test]
        fn proptest_phash(img in arbitrary_image(0..N, 0..N)) {
            let hash = phash(&img);
            assert_eq!(0, hash.hamming_distance(hash));
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::luma32f_bench_image;
    use test::{black_box, Bencher};

    const N: u32 = 600;

    #[bench]
    fn bench_phash(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        b.iter(|| {
            let img = black_box(&img);
            black_box(phash(img));
        });
    }
}
