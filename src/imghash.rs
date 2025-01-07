//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing
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
    let dct = dct(Cow::Owned(img));
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

fn dct(img: Cow<Image<Luma<f32>>>) -> Image<Luma<f32>> {
    #[allow(non_snake_case)]
    let T = |img: Cow<Image<_>>| -> Image<_> {
        let (w, h) = img.as_ref().dimensions();
        if w == h {
            let mut img = img.into_owned();
            transpose_inplace(&mut img);
            return img;
        }
        transpose(img.as_ref())
    };

    let mut planner = rustdct::DctPlanner::new();
    let rows_ctx = planner.plan_dct2(img.width() as usize);
    let cols_ctx = planner.plan_dct2(img.height() as usize);

    let scratch_len = rows_ctx.get_scratch_len().max(cols_ctx.get_scratch_len());
    let cap = scratch_len + rows_ctx.len().max(cols_ctx.len());
    let mut arena = vec![0f32; cap];

    let dct = dct_of_rows(&T(img), cols_ctx.as_ref(), &mut arena);
    dct_of_rows(&T(Cow::Owned(dct)), rows_ctx.as_ref(), &mut arena)
}

fn dct_of_rows(
    img: &Image<Luma<f32>>,
    ctx: &dyn rustdct::TransformType2And3<f32>,
    arena: &mut [f32],
) -> Image<Luma<f32>> {
    let width = usize::try_from(img.width()).unwrap();
    let height = usize::try_from(img.height()).unwrap();
    assert_eq!(width, ctx.len());

    let arena_len = ctx.len() + ctx.get_scratch_len();
    assert!(arena_len <= arena.len());
    let (dct_buf, scratch_space) = arena[..arena_len].split_at_mut(ctx.len());

    let mut dct = |inout: &mut [f32]| {
        debug_assert_eq!(inout.len(), ctx.len());
        ctx.process_dct2_with_scratch(inout, scratch_space);
    };

    let mut img_buf = Vec::with_capacity(width * height);
    for row in img.rows() {
        let row = row.into_iter().map(|p| p.0[0]);
        debug_assert_eq!(row.len(), dct_buf.len());
        for (dst, src) in dct_buf.iter_mut().zip(row) {
            *dst = src;
        }
        dct(dct_buf);
        img_buf.extend_from_slice(dct_buf);
    }
    debug_assert_eq!(img_buf.len(), width * height);
    Image::from_vec(img.width(), img.height(), img_buf).unwrap()
}

fn transpose_inplace(img: &mut Image<Luma<f32>>) {
    assert_eq!(
        img.width(),
        img.height(),
        "inplace transposition supported only for square images."
    );
    let n = usize::try_from(img.width()).unwrap();
    let at = |row, col| {
        debug_assert!(row < n);
        debug_assert!(col < n);
        row + n * col
    };
    let buf = img.as_mut();
    assert_eq!(buf.len(), n * n);
    for row in 0..n {
        for col in row..n {
            let a = buf[at(row, col)];
            let b = buf[at(col, row)];
            buf[at(row, col)] = b;
            buf[at(col, row)] = a;
        }
    }
}

fn transpose(img: &Image<Luma<f32>>) -> Image<Luma<f32>> {
    let nwidth = img.height();
    let nheight = img.width();
    Image::from_fn(nwidth, nheight, |x, y| *img.get_pixel(y, x))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Bits64(u64);

impl Bits64 {
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
    fn hamming_distance(self, other: Bits64) -> u32 {
        self.xor(other).0.count_ones()
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
        bits.set_nth_bit(2);
        assert_eq!(bits, Bits64(4));
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
    fn test_transpose_inplace() {
        #[allow(non_snake_case)]
        let T = |img: &Image<_>| {
            let mut img = img.clone();
            transpose_inplace(&mut img);
            img
        };

        let img = gray_image!(type: f32, 1.);
        assert_eq!(T(&img), img);

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

    #[test]
    fn test_transpose() {
        #[allow(non_snake_case)]
        let T = |img: &Image<_>| transpose(img);

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.
        );
        let img_t = gray_image!(type: f32,
            1., 4.;
            2., 5.;
            3., 6.
        );
        assert_pixels_eq!(T(&img), img_t);
        assert_pixels_eq!(T(&T(&img)), img);
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
        fn proptest_transpose(img in arbitrary_image(0..N, 0..N)) {
            let img_t = transpose(&img);
            assert_eq!(img.width(), img_t.height());
            assert_eq!(img.height(), img_t.width());
            assert_pixels_eq!(img, transpose(&img_t));
        }
        #[test]
        fn proptest_transpose_inplace(img in arbitrary_image(0..N, 0..N)) {
            #[allow(non_snake_case)]
            let T = |img: &Image<_>| {
                let mut img = img.clone();
                transpose_inplace(&mut img);
                img
            };
            if img.width() != img.height() {
                return Ok(());
            }
            assert_pixels_eq!(T(&img), transpose(&img));
            assert_pixels_eq!(T(&T(&img)), img);
        }
        #[test]
        fn proptest_dct_of_rows(img in arbitrary_image(0..N, 0..N)) {
            let ctx = rustdct::DctPlanner::new().plan_dct2(img.width() as usize);
            let mut arena = vec![0f32; ctx.len() + ctx.get_scratch_len()];
            let dct = dct_of_rows(&img, ctx.as_ref(), arena.as_mut());
            assert_eq!(dct.dimensions(), img.dimensions());
        }
        #[test]
        fn proptest_dct(img in arbitrary_image(0..N, 0..N)) {
            let dct = dct(Cow::Borrowed(&img));
            assert_eq!(dct.dimensions(), img.dimensions());
        }
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
    fn bench_transpose(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        b.iter(|| {
            black_box(transpose(&img));
        });
    }
    #[bench]
    fn bench_transpose_inplace(b: &mut Bencher) {
        let mut img = luma32f_bench_image(N, N);
        b.iter(|| {
            black_box(&mut img);
            transpose_inplace(&mut img);
        });
    }

    #[bench]
    fn bench_dct_of_rows(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        let ctx = rustdct::DctPlanner::new().plan_dct2(img.width() as usize);
        let mut arena = vec![0f32; ctx.len() + ctx.get_scratch_len()];
        b.iter(|| {
            let dct = dct_of_rows(black_box(&img), ctx.as_ref(), &mut arena);
            black_box(dct);
        });
    }

    #[bench]
    fn bench_dct(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        b.iter(|| {
            let dct = dct(Cow::Borrowed(&img));
            black_box(dct);
        });
    }

    #[bench]
    fn bench_phash(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        b.iter(|| {
            let img = black_box(&img);
            black_box(phash(img));
        });
    }
}
