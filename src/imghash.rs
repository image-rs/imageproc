//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing
use crate::definitions::Image;
use image::{imageops, math::Rect, Luma};
use std::sync::Arc;

/// Stores the result of the [`phash`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash(Bits64);

impl PHash {
    /// Compute the [hamming distance].
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
/// # fn main() {
/// let img1 = image::open!("first.png").to_luma32f();
/// let img2 = image::open!("second.png").to_luma32f();
/// let hash1 = phash(&img1);
/// let hash2 = phash(&img2);
/// let dist = hash1.hamming_distance(hash2);
/// dbg!(dist);
/// # }
/// ```
///
/// [pHash]: https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
/// [DCT]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
pub fn phash(img: &Image<Luma<f32>>) -> PHash {
    const HASH_SIZE: u32 = 8;
    const HASH_FACTOR: u32 = 4;
    const N: u32 = HASH_FACTOR * HASH_SIZE;

    let img = imageops::resize(img, N, N, imageops::FilterType::Lanczos3);
    let dct = {
        let mut ctx = DctCtx::plan(img.width() as usize);
        let dct = transpose_square(ctx.dct_of_image(&transpose_square(img)));
        ctx.dct_of_image(&dct)
    };
    let topleft = Rect {
        x: 0,
        y: 0,
        width: HASH_SIZE,
        height: HASH_SIZE,
    };
    let topleft_dct = crate::compose::crop(&dct, topleft);

    const SKIP: usize = 1;
    let mean = topleft_dct
        .iter()
        .skip(SKIP)
        .copied()
        .reduce(|a, b| a + b)
        .unwrap()
        / ((topleft_dct.len() - SKIP) as f32);

    let it = topleft_dct.iter().map(|&x| x > mean);
    PHash(Bits64::new(it))
}

struct DctCtx {
    arena: Vec<f32>,
    planner: Arc<dyn rustdct::TransformType2And3<f32>>,
}

impl DctCtx {
    fn plan(width: usize) -> Self {
        let planner = rustdct::DctPlanner::new().plan_dct2(width);
        let arena = vec![0f32; width + planner.get_scratch_len()];
        Self { planner, arena }
    }
    fn dct_of_image(&mut self, img: &Image<Luma<f32>>) -> Image<Luma<f32>> {
        let width = usize::try_from(img.width()).unwrap();
        let height = usize::try_from(img.height()).unwrap();

        let (scratch_space, dct_buf) = self.arena.split_at_mut(self.planner.get_scratch_len());
        assert_eq!(dct_buf.len(), width);

        let mut dct = |inout: &mut [f32]| {
            debug_assert_eq!(inout.len(), width);
            self.planner.process_dct2_with_scratch(inout, scratch_space);
        };

        let mut img_buf = Vec::with_capacity(width * height);
        for row in img.rows() {
            let row = row.into_iter().map(|p| p.0[0]);
            for (dst, src) in dct_buf.iter_mut().zip(row) {
                *dst = src;
            }
            dct(dct_buf);
            img_buf.extend_from_slice(dct_buf);
        }
        debug_assert_eq!(img_buf.len(), width * height);
        Image::from_vec(img.width(), img.height(), img_buf).unwrap()
    }
}

fn transpose_square(img: Image<Luma<f32>>) -> Image<Luma<f32>> {
    assert_eq!(img.width(), img.height());
    let n = usize::try_from(img.width()).unwrap();
    let at = |row, col| {
        debug_assert!(row < n);
        debug_assert!(col < n);
        row + n * col
    };

    let mut data = img.into_vec();
    let buf = data.as_mut_slice();
    assert_eq!(buf.len(), n * n);
    for row in 0..n {
        for col in row..n {
            let a = buf[at(row, col)];
            let b = buf[at(col, row)];
            buf[at(row, col)] = b;
            buf[at(col, row)] = a;
        }
    }
    Image::from_vec(n as u32, n as u32, data).unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Bits64(u64);

impl Bits64 {
    fn hamming_distance(self, other: Bits64) -> u32 {
        self.xor(other).0.count_ones()
    }
    fn new(v: impl IntoIterator<Item = bool>) -> Self {
        const N: usize = 64;
        let mut bits = Self::zero();
        let mut n = 0;
        for bit in v {
            if bit {
                bits.set_nth_bit(n);
            } else {
                bits.clear_nth_bit(n);
            };
            n += 1;
        }
        assert_eq!(n, N);
        bits
    }
    fn zero() -> Self {
        Self(0)
    }
    fn set_nth_bit(&mut self, n: usize) {
        debug_assert!(n < 64);
        self.0 |= 1 << n;
    }
    fn clear_nth_bit(&mut self, n: usize) {
        debug_assert!(n < 64);
        self.0 &= !(1 << n);
    }
    fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bits64_ops() {
        let mut bits = Bits64::zero();
        bits.set_nth_bit(0);
        assert_eq!(bits, Bits64(1));
        bits.set_nth_bit(1);
        assert_eq!(bits, Bits64(1 + 2));
        bits.clear_nth_bit(0);
        assert_eq!(bits, Bits64(2));
        bits.clear_nth_bit(1);
        assert_eq!(bits, Bits64::zero());
    }
    #[test]
    fn test_bits64_new() {
        const N: usize = 64;

        let mut v = [false; N];
        v[0] = true;
        let it = v.iter().copied();
        assert_eq!(Bits64::new(it), Bits64(1));
        v[1] = true;
        let it = v.iter().copied();
        assert_eq!(Bits64::new(it), Bits64(1 + 2));
    }
    #[test]
    #[should_panic]
    fn test_bits64_new_fail() {
        const N: usize = 64;
        let it = (1..N).map(|x| x % 2 == 0);
        let _bits = Bits64::new(it);
    }

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

    #[test]
    fn test_transpose() {
        #[allow(non_snake_case)]
        let T = |img: &Image<Luma<f32>>| transpose_square(img.clone());

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.;
            7., 8., 9.
        );
        let img_t = gray_image!(type: f32,
            1., 4., 7.;
            2., 5., 8.;
            3., 6., 9.
        );
        assert_pixels_eq!(T(&img), img_t);
        assert_pixels_eq!(T(&T(&img)), img);
    }
}
