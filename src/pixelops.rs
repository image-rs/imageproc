//! Pixel manipulations.

use crate::definitions::Clamp;
use crate::math::cast;
use conv::ValueInto;
use image::Pixel;

/// Adds pixels with the given weights. Results are clamped to prevent arithmetical overflows.
///
/// # Examples
/// ```
/// # extern crate image;
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::pixelops::weighted_sum;
///
/// let left = Rgb([10u8, 20u8, 30u8]);
/// let right = Rgb([100u8, 80u8, 60u8]);
///
/// let sum = weighted_sum(left, right, 0.7, 0.3);
/// assert_eq!(sum, Rgb([37, 38, 39]));
/// # }
/// ```
pub fn weighted_sum<P: Pixel>(left: P, right: P, left_weight: f32, right_weight: f32) -> P
where
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    left.map2(&right, |p, q| {
        weighted_channel_sum(p, q, left_weight, right_weight)
    })
}

/// Equivalent to `weighted_sum(left, right, left_weight, 1 - left_weight)`.
///
/// # Examples
/// ```
/// # extern crate image;
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::pixelops::interpolate;
///
/// let left = Rgb([10u8, 20u8, 30u8]);
/// let right = Rgb([100u8, 80u8, 60u8]);
///
/// let sum = interpolate(left, right, 0.7);
/// assert_eq!(sum, Rgb([37, 38, 39]));
/// # }
/// ```
pub fn interpolate<P: Pixel>(left: P, right: P, left_weight: f32) -> P
where
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    weighted_sum(left, right, left_weight, 1.0 - left_weight)
}

#[inline(always)]
fn weighted_channel_sum<C>(left: C, right: C, left_weight: f32, right_weight: f32) -> C
where
    C: ValueInto<f32> + Clamp<f32>,
{
    Clamp::clamp(cast(left) * left_weight + cast(right) * right_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};
    use test::{black_box, Bencher};

    #[test]
    fn test_weighted_channel_sum() {
        // Midpoint
        assert_eq!(weighted_channel_sum(10u8, 20u8, 0.5, 0.5), 15u8);
        // Mainly left
        assert_eq!(weighted_channel_sum(10u8, 20u8, 0.9, 0.1), 11u8);
        // Clamped
        assert_eq!(weighted_channel_sum(150u8, 150u8, 1.8, 0.8), 255u8);
    }

    #[bench]
    fn bench_weighted_sum_rgb(b: &mut Bencher) {
        b.iter(|| {
            let left = black_box(Rgb([10u8, 20u8, 33u8]));
            let right = black_box(Rgb([80u8, 70u8, 60u8]));
            let left_weight = black_box(0.3);
            let right_weight = black_box(0.7);
            black_box(weighted_sum(left, right, left_weight, right_weight));
        })
    }

    #[bench]
    fn bench_weighted_sum_gray(b: &mut Bencher) {
        b.iter(|| {
            let left = black_box(Luma([10u8]));
            let right = black_box(Luma([80u8]));
            let left_weight = black_box(0.3);
            let right_weight = black_box(0.7);
            black_box(weighted_sum(left, right, left_weight, right_weight));
        })
    }

    #[bench]
    fn bench_interpolate_rgb(b: &mut Bencher) {
        b.iter(|| {
            let left = black_box(Rgb([10u8, 20u8, 33u8]));
            let right = black_box(Rgb([80u8, 70u8, 60u8]));
            let left_weight = black_box(0.3);
            black_box(interpolate(left, right, left_weight));
        })
    }

    #[bench]
    fn bench_interpolate_gray(b: &mut Bencher) {
        b.iter(|| {
            let left = black_box(Luma([10u8]));
            let right = black_box(Luma([80u8]));
            let left_weight = black_box(0.3);
            black_box(interpolate(left, right, left_weight));
        })
    }
}
