//! Statistical properties of images.

use crate::definitions::Image;
use crate::math::cast;
use conv::ValueInto;
use image::{GenericImageView, GrayImage, Pixel, Primitive};
use num::Bounded;

/// A set of per-channel histograms from an image with 8 bits per channel.
pub struct ChannelHistogram {
    /// Per-channel histograms.
    pub channels: Vec<[u32; 256]>,
}

/// Returns a vector of per-channel histograms.
pub fn histogram<P>(image: &Image<P>) -> ChannelHistogram
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let mut hist = vec![[0u32; 256]; P::CHANNEL_COUNT as usize];

    for pix in image.pixels() {
        for (i, c) in pix.channels().iter().enumerate() {
            hist[i][*c as usize] += 1;
        }
    }

    ChannelHistogram { channels: hist }
}

/// A set of per-channel cumulative histograms from an image with 8 bits per channel.
pub struct CumulativeChannelHistogram {
    /// Per-channel cumulative histograms.
    pub channels: Vec<[u32; 256]>,
}

/// Returns per-channel cumulative histograms.
pub fn cumulative_histogram<P>(image: &Image<P>) -> CumulativeChannelHistogram
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let mut hist = histogram(image);

    for c in 0..hist.channels.len() {
        for i in 1..hist.channels[c].len() {
            hist.channels[c][i] += hist.channels[c][i - 1];
        }
    }

    CumulativeChannelHistogram {
        channels: hist.channels,
    }
}

/// Returns the `p`th percentile of the pixel intensities in an image.
///
/// We define the `p`th percentile intensity to be the least `x` such
/// that at least `p`% of image pixels have intensity less than or
/// equal to `x`.
///
/// # Panics
/// If `p > 100`.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::stats::percentile;
///
/// let image = gray_image!(
///     1, 2, 3, 4, 5;
///     6, 7, 8, 9, 10);
///
/// // The 0th percentile is always 0
/// assert_eq!(percentile(&image, 0), 0);
///
/// // Exactly 10% of pixels have intensity <= 1.
/// assert_eq!(percentile(&image, 10), 1);
///
/// // Fewer than 15% of pixels have intensity <=1, so the 15th percentile is 2.
/// assert_eq!(percentile(&image, 15), 2);
///
/// // All pixels have intensity <= 10.
/// assert_eq!(percentile(&image, 100), 10);
/// # }
/// ```
pub fn percentile(image: &GrayImage, p: u8) -> u8 {
    assert!(p <= 100, "requested percentile must be <= 100");

    let cum_hist = cumulative_histogram(&image).channels[0];
    let total = cum_hist[255] as u64;

    for i in 0..256 {
        if 100 * cum_hist[i] as u64 / total >= p as u64 {
            return i as u8;
        }
    }

    unreachable!();
}

/// Returns the square root of the mean of the squares of differences
/// between all subpixels in left and right. All channels are considered
/// equally. If you do not want this (e.g. if using RGBA) then change
/// image formats first.
pub fn root_mean_squared_error<I, J, P>(left: &I, right: &J) -> f64
where
    I: GenericImageView<Pixel = P>,
    J: GenericImageView<Pixel = P>,
    P: Pixel,
    P::Subpixel: ValueInto<f64>,
{
    mean_squared_error(left, right).sqrt()
}

/// Returns the peak signal to noise ratio for a clean image and its noisy
/// aproximation. All channels are considered equally. If you do not want this
/// (e.g. if using RGBA) then change image formats first.
/// See also [peak signal-to-noise ratio (wikipedia)](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio).
pub fn peak_signal_to_noise_ratio<I, J, P>(original: &I, noisy: &J) -> f64
where
    I: GenericImageView<Pixel = P>,
    J: GenericImageView<Pixel = P>,
    P: Pixel,
    P::Subpixel: ValueInto<f64> + Primitive,
{
    let max: f64 = cast(<P::Subpixel as Bounded>::max_value());
    let mse = mean_squared_error(original, noisy);
    20f64 * max.log(10f64) - 10f64 * mse.log(10f64)
}

fn mean_squared_error<I, J, P>(left: &I, right: &J) -> f64
where
    I: GenericImageView<Pixel = P>,
    J: GenericImageView<Pixel = P>,
    P: Pixel,
    P::Subpixel: ValueInto<f64>,
{
    assert_dimensions_match!(left, right);
    let mut sum_squared_diffs = 0f64;
    for (p, q) in left.pixels().zip(right.pixels()) {
        for (c, d) in p.2.channels().iter().zip(q.2.channels().iter()) {
            let fc: f64 = cast(*c);
            let fd: f64 = cast(*d);
            let diff = fc - fd;
            sum_squared_diffs += diff * diff;
        }
    }
    let count = (left.width() * left.height() * P::CHANNEL_COUNT as u32) as f64;
    sum_squared_diffs / count
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};
    use test::Bencher;

    #[test]
    fn test_cumulative_histogram() {
        let image = gray_image!(1u8, 2u8, 3u8, 2u8, 1u8);
        let hist = cumulative_histogram(&image).channels[0];

        assert_eq!(hist[0..4], [0, 2, 4, 5]);
        assert!(hist.iter().skip(4).all(|x| *x == 5));
    }

    #[test]
    fn test_cumulative_histogram_rgb() {
        let image = rgb_image!(
            [1u8, 10u8, 1u8],
            [2u8, 20u8, 2u8],
            [3u8, 30u8, 3u8],
            [2u8, 20u8, 2u8],
            [1u8, 10u8, 1u8]
        );

        let hist = cumulative_histogram(&image);
        let r = hist.channels[0];
        let g = hist.channels[1];
        let b = hist.channels[2];

        assert_eq!(r[0..4], [0, 2, 4, 5]);
        assert!(r.iter().skip(4).all(|x| *x == 5));

        assert_eq!(g[0..10], [0; 10]);
        assert_eq!(g[10..20], [2; 10]);
        assert_eq!(g[20..30], [4; 10]);
        assert_eq!(g[30], 5);
        assert!(b.iter().skip(30).all(|x| *x == 5));

        assert_eq!(b[0..4], [0, 2, 4, 5]);
        assert!(b.iter().skip(4).all(|x| *x == 5));
    }

    #[test]
    fn test_histogram() {
        let image = gray_image!(1u8, 2u8, 3u8, 2u8, 1u8);
        let hist = histogram(&image).channels[0];

        assert_eq!(hist[0..4], [0, 2, 2, 1]);
    }

    #[test]
    fn test_histogram_rgb() {
        let image = rgb_image!(
            [1u8, 10u8, 1u8],
            [2u8, 20u8, 2u8],
            [3u8, 30u8, 3u8],
            [2u8, 20u8, 2u8],
            [1u8, 10u8, 1u8]
        );

        let hist = histogram(&image);
        let r = hist.channels[0];
        let g = hist.channels[1];
        let b = hist.channels[2];

        assert_eq!(r[0..4], [0, 2, 2, 1]);
        assert!(r.iter().skip(4).all(|x| *x == 0));

        assert_eq!(g[0..10], [0; 10]);
        assert_eq!(g[10], 2);
        assert_eq!(g[11..20], [0; 9]);
        assert_eq!(g[20], 2);
        assert_eq!(g[21..30], [0; 9]);
        assert_eq!(g[30], 1);
        assert!(b.iter().skip(30).all(|x| *x == 0));

        assert_eq!(b[0..4], [0, 2, 2, 1]);
        assert!(b.iter().skip(4).all(|x| *x == 0));
    }

    #[test]
    fn test_root_mean_squared_error_grayscale() {
        let left = gray_image!(
            1, 2, 3;
            4, 5, 6);

        let right = gray_image!(
            8, 4, 7;
            6, 9, 1);

        let rms = root_mean_squared_error(&left, &right);
        let expected = (114f64 / 6f64).sqrt();
        assert_eq!(rms, expected);
    }

    #[test]
    fn test_root_mean_squared_error_rgb() {
        let left = rgb_image!([1, 2, 3], [4, 5, 6]);
        let right = rgb_image!([8, 4, 7], [6, 9, 1]);
        let rms = root_mean_squared_error(&left, &right);
        let expected = (114f64 / 6f64).sqrt();
        assert_eq!(rms, expected);
    }

    #[test]
    #[should_panic]
    fn test_root_mean_squares_rejects_mismatched_dimensions() {
        let left = gray_image!(1, 2);
        let right = gray_image!(8; 4);
        let _ = root_mean_squared_error(&left, &right);
    }

    fn left_image_rgb(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| Rgb([x as u8, y as u8, (x + y) as u8]))
    }

    fn right_image_rgb(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| Rgb([(x + y) as u8, x as u8, y as u8]))
    }

    #[bench]
    fn bench_root_mean_squared_error_rgb(b: &mut Bencher) {
        let left = left_image_rgb(50, 50);
        let right = right_image_rgb(50, 50);

        b.iter(|| {
            let error = root_mean_squared_error(&left, &right);
            test::black_box(error);
        });
    }

    fn left_image_gray(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, _| Luma([x as u8]))
    }

    fn right_image_gray(width: u32, height: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |_, y| Luma([y as u8]))
    }

    #[bench]
    fn bench_root_mean_squared_error_gray(b: &mut Bencher) {
        let left = left_image_gray(50, 50);
        let right = right_image_gray(50, 50);

        b.iter(|| {
            let error = root_mean_squared_error(&left, &right);
            test::black_box(error);
        });
    }
}
