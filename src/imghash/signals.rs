use crate::definitions::Image;
use image::Luma;
use std::borrow::Cow;

/// Computes the 2 dimensional [DCT].
///
/// [DCT]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
pub(super) fn dct2d(img: Cow<Image<Luma<f32>>>) -> Image<Luma<f32>> {
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
    let cap = {
        let rows_cap = rows_ctx.len() + rows_ctx.get_scratch_len();
        let cols_cap = cols_ctx.len() + cols_ctx.get_scratch_len();
        rows_cap.max(cols_cap)
    };
    let mut arena = vec![0f32; cap];

    let dct = dct1d(&T(img), cols_ctx.as_ref(), &mut arena);
    dct1d(&T(Cow::Owned(dct)), rows_ctx.as_ref(), &mut arena)
}

/// Computes the 1 dimensional [DCT] for each row.
///
/// [DCT]: https://en.wikipedia.org/wiki/Discrete_cosine_transform
// TODO: compute inplace
fn dct1d(
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
        debug_assert_eq!(scratch_space.len(), ctx.get_scratch_len());
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

#[cfg(test)]
mod tests {
    use super::*;

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
    #[test]
    fn test_dct1d() {
        let mut arena = vec![0f32; 1_000];
        let mut planner = rustdct::DctPlanner::new();
        #[allow(non_snake_case)]
        let mut DCT = |img: &Image<_>| -> Image<_> {
            let ctx = planner.plan_dct2(img.width() as usize);
            dct1d(img, ctx.as_ref(), arena.as_mut())
        };
        let img = gray_image!(type: f32,
            1., 2., 3.
        );
        let expected = gray_image!(type: f32,
            6.0, -1.7320508, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.
        );
        let expected = gray_image!(type: f32,
            6.0, -1.7320508, 0.0;
            15.0, -1.7320508, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.;
            7., 8., 9.
        );
        let expected = gray_image!(type: f32,
            6.0, -1.7320508, 0.0;
            15.0, -1.7320508, 0.0;
            24.0, -1.7320508, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);
    }
    #[test]
    fn test_dct2d() {
        #[allow(non_snake_case)]
        let DCT = |img: &Image<_>| dct2d(Cow::Borrowed(img));

        let img = gray_image!(type: f32,
            1., 2., 3.
        );
        let expected = gray_image!(type: f32,
            6.0, -1.7320508, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.
        );
        let expected = gray_image!(type: f32,
            21.0, -3.464_101_6, 0.0;
            -6.3639607, 0.0, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);

        let img = gray_image!(type: f32,
            1., 2., 3.;
            4., 5., 6.;
            7., 8., 9.
        );
        let expected = gray_image!(type: f32,
            45.0, -5.196_152, 0.0;
            -15.588_457, 0.0, 0.0;
            0.0, 0.0, 0.0
        );
        assert_pixels_eq!(DCT(&img), expected);
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
        fn proptest_dct1d(img in arbitrary_image(0..N, 0..N)) {
            let ctx = rustdct::DctPlanner::new().plan_dct2(img.width() as usize);
            let mut arena = vec![0f32; ctx.len() + ctx.get_scratch_len()];
            let dct = dct1d(&img, ctx.as_ref(), arena.as_mut());
            assert_eq!(dct.dimensions(), img.dimensions());
        }
        #[test]
        fn proptest_dct2d(img in arbitrary_image(0..N, 0..N)) {
            let dct = dct2d(Cow::Borrowed(&img));
            assert_eq!(dct.dimensions(), img.dimensions());
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
    fn bench_dct1d(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        let ctx = rustdct::DctPlanner::new().plan_dct2(img.width() as usize);
        let mut arena = vec![0f32; ctx.len() + ctx.get_scratch_len()];
        b.iter(|| {
            let dct = dct1d(black_box(&img), ctx.as_ref(), &mut arena);
            black_box(dct);
        });
    }

    #[bench]
    fn bench_dct2d(b: &mut Bencher) {
        let img = luma32f_bench_image(N, N);
        b.iter(|| {
            let dct = dct2d(black_box(Cow::Borrowed(&img)));
            black_box(dct);
        });
    }
}
