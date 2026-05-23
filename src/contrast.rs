//! Functions for manipulating the contrast of images.

use std::cmp::min;

use image::{GrayImage, Luma, Pixel};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::definitions::{HasBlack, HasWhite, Image};
use crate::filter::gaussian_blur_f32;
use crate::integral_image::{integral_image, sum_image_pixels};
use crate::map::{map_pixels, map_pixels_mut};
use crate::stats::{cumulative_histogram, histogram};

/// Specifies the adaptive thresholding algorithm to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AdaptiveThresholdType {
    /// The threshold value is the mean of the neighborhood area minus delta.
    Mean,
    /// The threshold value is a Gaussian-weighted sum of the neighborhood values minus delta.
    Gaussian,
}

/// Applies an adaptive threshold to a grayscale image.
///
/// For each pixel, a threshold is calculated based on the surrounding pixels in a `(2 * block_radius + 1)`
/// square block. The thresholding algorithm is determined by `adaptive_type`.
///
/// If the pixel's brightness is greater than or equal to `threshold - delta`, it is set to 255 (white),
/// otherwise it is set to 0 (black).
///
/// * `Mean`: The threshold is the mean of the pixel values in the block.
/// * `Gaussian`: The threshold is a Gaussian-weighted sum of the pixel values in the block.
///
/// # Panics
///
/// If `block_radius` is zero.
pub fn adaptive_threshold(
    image: &GrayImage,
    block_radius: u32,
    delta: i32,
    adaptive_type: AdaptiveThresholdType,
) -> GrayImage {
    assert!(block_radius > 0, "block_radius must be positive");
    let ksize = 2 * block_radius + 1;

    match adaptive_type {
        AdaptiveThresholdType::Mean => adaptive_threshold_mean(image, block_radius, delta),
        AdaptiveThresholdType::Gaussian => adaptive_threshold_gaussian(image, ksize, delta),
    }
}

/// Applies mean adaptive thresholding using an integral image for high performance.
///
/// The threshold for a pixel is the mean of the pixel values within the
/// `(2 * block_radius + 1) x (2 * block_radius + 1)` neighborhood. A pixel is set to 255
/// if its value is `>= mean - delta`, and 0 otherwise. The local mean is calculated
/// efficiently by using a pre-computed integral image.
fn adaptive_threshold_mean(image: &GrayImage, block_radius: u32, delta: i32) -> GrayImage {
    let integral = integral_image::<_, u32>(image);
    let mut out = GrayImage::from_pixel(image.width(), image.height(), Luma::black());

    for y in 0..image.height() {
        for x in 0..image.width() {
            // Traverse all neighbors in (2 * block_radius + 1) x (2 * block_radius + 1)
            let y_low = y.saturating_sub(block_radius);
            let y_high = min(image.height() - 1, y + block_radius);
            let x_low = x.saturating_sub(block_radius);
            let x_high = min(image.width() - 1, x + block_radius);

            // Number of pixels in the block, adjusted for edge cases.
            let w = (y_high - y_low + 1) * (x_high - x_low + 1);
            let mean = sum_image_pixels(&integral, x_low, y_low, x_high, y_high)[0] / w;

            if image.get_pixel(x, y)[0] as i32 >= mean as i32 - delta {
                out.put_pixel(x, y, Luma::white());
            }
        }
    }
    out
}

/// Applies Gaussian adaptive thresholding by blurring the image.
///
/// The threshold for a pixel is the Gaussian-weighted sum of the pixel values
/// in a `ksize x ksize` neighborhood. This is calculated by applying a Gaussian
/// blur. A pixel is set to 255 if its value is `>= weighted_sum - delta`, and 0
/// otherwise. The sigma for the Gaussian kernel is derived from `ksize` to
/// match the behavior of libraries like OpenCV.
fn adaptive_threshold_gaussian(image: &GrayImage, ksize: u32, delta: i32) -> GrayImage {
    // The formula for sigma is derived from OpenCV's [`getGaussianKernel()`](https://github.com/opencv/opencv/blob/dac243bd265e79af2315ce04fac2a0a5bdf47efe/modules/imgproc/include/opencv2/imgproc.hpp#L1453-L1454).
    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    let sigma = 0.3 * ((ksize as f32 - 1.0) * 0.5 - 1.0) + 0.8;

    let float_image = map_pixels(image, |p| Luma([p[0] as f32]));
    let blurred = gaussian_blur_f32(&float_image, sigma);

    let mut out = GrayImage::from_pixel(image.width(), image.height(), Luma::black());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let threshold = blurred.get_pixel(x, y)[0] as i32;
            if image.get_pixel(x, y)[0] as i32 >= threshold - delta {
                out.put_pixel(x, y, Luma::white());
            }
        }
    }
    out
}

/// Returns the [Otsu threshold level] of an 8bpp image.
///
/// [Otsu threshold level]: https://en.wikipedia.org/wiki/Otsu%27s_method
pub fn otsu_level(image: &GrayImage) -> u8 {
    let hist = histogram(image);
    let (width, height) = image.dimensions();
    let total_weight = width * height;

    // Sum of all pixel intensities, to use when calculating means.
    let total_pixel_sum = hist.channels[0]
        .iter()
        .enumerate()
        .fold(0f64, |sum, (t, h)| sum + (t as u32 * h) as f64);

    // Sum of all pixel intensities in the background class.
    let mut background_pixel_sum = 0f64;

    // The weight of a class (background or foreground) is
    // the number of pixels which belong to that class at
    // the current threshold.
    let mut background_weight = 0u32;
    let mut foreground_weight;

    let mut largest_variance = 0f64;
    let mut best_threshold = 0u8;

    for (threshold, hist_count) in hist.channels[0].iter().enumerate() {
        background_weight += hist_count;
        if background_weight == 0 {
            continue;
        };

        foreground_weight = total_weight - background_weight;
        if foreground_weight == 0 {
            break;
        };

        background_pixel_sum += (threshold as u32 * hist_count) as f64;
        let foreground_pixel_sum = total_pixel_sum - background_pixel_sum;

        let background_mean = background_pixel_sum / (background_weight as f64);
        let foreground_mean = foreground_pixel_sum / (foreground_weight as f64);

        let mean_diff_squared = (background_mean - foreground_mean).powi(2);
        let intra_class_variance =
            (background_weight as f64) * (foreground_weight as f64) * mean_diff_squared;

        if intra_class_variance > largest_variance {
            largest_variance = intra_class_variance;
            best_threshold = threshold as u8;
        }
    }

    best_threshold
}

/// Returns the [Kapur threshold level] of an 8bpp image. This threshold
/// maximizes the entropy of the background and foreground.
///
/// [Kapur threshold level]: https://doi.org/10.1016/0734-189X(85)90125-2
pub fn kapur_level(img: &GrayImage) -> u8 {
    // The implementation looks different to the one you can for example find in
    // ImageMagick, because we are using the simplification of equation (18) in
    // the original article, which allows the computation of the total entropy
    // without having to use nested loops. The names of the variables are taken
    // straight from the article.
    let hist = histogram(img);
    let histogram = &hist.channels[0];
    const N: usize = 256;

    let total_pixels = (img.width() * img.height()) as f64;

    // The p_i in the article. They describe the probability of encountering
    // gray-level i.
    let mut p = [0.0f64; N];
    for i in 0..N {
        p[i] = histogram[i] as f64 / total_pixels;
    }

    // The P_s in the article, which is the probability of encountering
    // gray-level <= s.
    let mut cum_p = [0.0f64; N];
    cum_p[0] = p[0];
    for i in 1..N {
        cum_p[i] = cum_p[i - 1] + p[i];
    }

    // The H_s in the article. These are the entropies attached to the
    // distributions p[0],...,p[s].
    let mut h = [0.0f64; N];
    if p[0] > 0.0 {
        h[0] = -p[0] * p[0].ln();
    }
    for s in 1..N {
        h[s] = if p[s] > 0.0 {
            h[s - 1] - p[s] * p[s].ln()
        } else {
            h[s - 1]
        };
    }

    let mut max_entropy = f64::MIN;
    let mut best_threshold = 0;

    for s in 0..N {
        let pq = cum_p[s] * (1.0 - cum_p[s]);
        if pq <= 0.0 {
            continue;
        }

        // psi_s is the sum of the total entropy of foreground and
        // background at threshold level s. Instead of computing them
        // separately, we use equation (18) of the original article, which
        // simplifies it to this:
        let psi_s = pq.ln() + h[s] / cum_p[s] + (h[255] - h[s]) / (1.0 - cum_p[s]);
        if psi_s > max_entropy {
            max_entropy = psi_s;
            best_threshold = s;
        }
    }

    best_threshold as u8
}

/// Options for how to treat the threshold value in [`threshold`] and [`threshold_mut`].
pub enum ThresholdType {
    /// `dst(x,y) = if src(x,y) > threshold { 255 } else { 0 }`
    Binary,
    /// `dst(x,y) = if src(x,y) > threshold { 0 } else { 255 }`
    BinaryInverted,
    /// `dst(x,y) = if src(x,y) > threshold { threshold } else { src(x,y) }`
    Truncate,
    /// `dst(x,y) = if src(x,y) > threshold { src(x,y) } else { 0 }`
    ToZero,
    /// `dst(x,y) = if src(x,y) > threshold { 0 } else { src(x,y) }`
    ToZeroInverted,
}

/// Applies a threshold to each pixel in a grayscale image. The action taken depends on
/// `threshold_type` - see [`ThresholdType`].
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::contrast::{threshold, ThresholdType};
///
/// let image = gray_image!(
///     10, 80, 20;
///     50, 90, 70);
///
/// // Binary threshold
/// let threshold_binary = gray_image!(
///     0, 255,   0;
///     0, 255, 255);
///
/// assert_pixels_eq!(
///     threshold(&image, 50, ThresholdType::Binary),
///     threshold_binary);
///
/// // Inverted binary threshold
/// let threshold_binary_inverted = gray_image!(
///     255,   0, 255;
///     255,   0,   0);
///
/// assert_pixels_eq!(
///     threshold(&image, 50, ThresholdType::BinaryInverted),
///     threshold_binary_inverted);
///
/// // Truncate
/// let threshold_truncate = gray_image!(
///     10, 50, 20;
///     50, 50, 50);
///
/// assert_pixels_eq!(
///     threshold(&image, 50, ThresholdType::Truncate),
///     threshold_truncate);
///
/// // To zero
/// let threshold_to_zero = gray_image!(
///     10,  0, 20;
///     50,  0,  0);
///
/// assert_pixels_eq!(
///     threshold(&image, 50, ThresholdType::ToZero),
///     threshold_to_zero);
///
/// // To zero inverted
/// let threshold_to_zero_inverted = gray_image!(
///     0, 80,  0;
///     0, 90, 70);
///
/// assert_pixels_eq!(
///     threshold(&image, 50, ThresholdType::ToZeroInverted),
///     threshold_to_zero_inverted);
/// # }
/// ```
pub fn threshold(image: &GrayImage, threshold: u8, threshold_type: ThresholdType) -> GrayImage {
    let mut out = image.clone();
    threshold_mut(&mut out, threshold, threshold_type);
    out
}
#[doc=generate_mut_doc_comment!("threshold")]
pub fn threshold_mut(image: &mut GrayImage, threshold: u8, threshold_type: ThresholdType) {
    match threshold_type {
        ThresholdType::Binary => {
            for p in image.iter_mut() {
                *p = if *p > threshold { 255 } else { 0 };
            }
        }
        ThresholdType::BinaryInverted => {
            for p in image.iter_mut() {
                *p = if *p > threshold { 0 } else { 255 };
            }
        }
        ThresholdType::Truncate => {
            for p in image.iter_mut() {
                *p = if *p > threshold { threshold } else { *p };
            }
        }
        ThresholdType::ToZero => {
            for p in image.iter_mut() {
                *p = if *p > threshold { 0 } else { *p };
            }
        }
        ThresholdType::ToZeroInverted => {
            for p in image.iter_mut() {
                *p = if *p > threshold { *p } else { 0 };
            }
        }
    }
}

/// Equalises the histogram of an 8bpp grayscale image. See also
/// [histogram equalization (wikipedia)](https://en.wikipedia.org/wiki/Histogram_equalization).
pub fn equalize_histogram(image: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    equalize_histogram_mut(&mut out);
    out
}
#[doc=generate_mut_doc_comment!("equalize_histogram")]
pub fn equalize_histogram_mut(image: &mut GrayImage) {
    let hist = cumulative_histogram(image).channels[0];
    let total = hist[255] as f32;

    #[cfg(feature = "rayon")]
    let iter = image.par_iter_mut();
    #[cfg(not(feature = "rayon"))]
    let iter = image.iter_mut();

    iter.for_each(|p| {
        // JUSTIFICATION
        //  Benefit
        //      Using checked indexing here makes this function take 1.1x longer, as measured
        //      by bench_equalize_histogram_mut
        //  Correctness
        //      Each channel of CumulativeChannelHistogram has length 256, and a GrayImage has 8 bits per pixel
        let fraction = unsafe { *hist.get_unchecked(*p as usize) as f32 / total };
        *p = (f32::min(255f32, 255f32 * fraction)) as u8;
    });
}

/// Stretches the contrast in an image, linearly mapping intensities in `(input_lower, input_upper)` to `(output_lower, output_upper)` and saturating
/// values outside this input range.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::contrast::stretch_contrast;
///
/// let image = gray_image!(
///      0,   20,  50;
///     80,  100, 255);
///
/// let lower = 20;
/// let upper = 100;
///
/// // Pixel intensities between 20 and 100 are linearly
/// // scaled so that 20 is mapped to 0 and 100 is mapped to 255.
/// // Pixel intensities less than 20 are sent to 0 and pixel
/// // intensities greater than 100 are sent to 255.
/// let stretched = stretch_contrast(&image, lower, upper, 0u8, 255u8);
///
/// let expected = gray_image!(
///       0,   0,  95;
///     191, 255, 255);
///
/// assert_pixels_eq!(stretched, expected);
/// # }
/// ```
///
/// # Panics
/// If `input_lower >= input_upper` or `output_lower > output_upper`.
pub fn stretch_contrast<P>(
    image: &Image<P>,
    input_lower: u8,
    input_upper: u8,
    output_lower: u8,
    output_upper: u8,
) -> Image<P>
where
    P: Pixel<Subpixel = u8>,
{
    let mut out = image.clone();
    stretch_contrast_mut(
        &mut out,
        input_lower,
        input_upper,
        output_lower,
        output_upper,
    );
    out
}
#[doc=generate_mut_doc_comment!("stretch_contrast")]
pub fn stretch_contrast_mut<P>(
    image: &mut Image<P>,
    input_min: u8,
    input_max: u8,
    output_min: u8,
    output_max: u8,
) where
    P: Pixel<Subpixel = u8>,
{
    assert!(
        input_min < input_max,
        "input_min must be smaller than input_max"
    );
    assert!(
        output_min <= output_max,
        "output_min must be smaller or equal to output_max"
    );

    let input_min: u16 = input_min.into();
    let input_max: u16 = input_max.into();
    let output_min: u16 = output_min.into();
    let output_max: u16 = output_max.into();

    let input_width = input_max - input_min;
    let output_width = output_max - output_min;

    let f = |p: P| {
        p.map_without_alpha(|c| {
            let c = u16::from(c);

            if c <= input_min {
                (output_min) as u8
            } else if c >= input_max {
                (output_max) as u8
            } else {
                ((((c - input_min) * output_width) / input_width) + output_min) as u8
            }
        })
    };

    map_pixels_mut(image, f);
}

/// Adjusts contrast of an 8bpp grayscale image so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram(image: &GrayImage, target: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    match_histogram_mut(&mut out, target);
    out
}
#[doc=generate_mut_doc_comment!("match_histogram")]
pub fn match_histogram_mut(image: &mut GrayImage, target: &GrayImage) {
    let image_histc = cumulative_histogram(image).channels[0];
    let target_histc = cumulative_histogram(target).channels[0];
    let lut = histogram_lut(&image_histc, &target_histc);

    for p in image.iter_mut() {
        *p = lut[*p as usize] as u8;
    }
}

/// `l = histogram_lut(s, t)` is chosen so that `target_histc[l[i]] / sum(target_histc)`
/// is as close as possible to `source_histc[i] / sum(source_histc)`.
fn histogram_lut(source_histc: &[u32; 256], target_histc: &[u32; 256]) -> [usize; 256] {
    let source_total = source_histc[255] as f32;
    let target_total = target_histc[255] as f32;

    let mut lut = [0usize; 256];
    let mut y = 0usize;
    let mut prev_target_fraction = 0f32;

    for s in 0..256 {
        let source_fraction = source_histc[s] as f32 / source_total;
        let mut target_fraction = target_histc[y] as f32 / target_total;

        while source_fraction > target_fraction && y < 255 {
            y += 1;
            prev_target_fraction = target_fraction;
            target_fraction = target_histc[y] as f32 / target_total;
        }

        if y == 0 {
            lut[s] = y;
        } else {
            let prev_dist = f32::abs(prev_target_fraction - source_fraction);
            let dist = f32::abs(target_fraction - source_fraction);
            if prev_dist < dist {
                lut[s] = y - 1;
            } else {
                lut[s] = y;
            }
        }
    }

    lut
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::{HasBlack, HasWhite};
    use image::{GrayImage, Luma};

    #[test]
    fn adaptive_threshold_mean_constant() {
        let image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        let binary = adaptive_threshold(&image, 1, 0, AdaptiveThresholdType::Mean);
        let expected = GrayImage::from_pixel(3, 3, Luma::white());
        assert_pixels_eq!(binary, expected);
    }

    #[test]
    fn adaptive_threshold_gaussian_constant() {
        let image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        let binary = adaptive_threshold(&image, 1, 0, AdaptiveThresholdType::Gaussian);
        let expected = GrayImage::from_pixel(3, 3, Luma::white());
        assert_pixels_eq!(binary, expected);
    }

    #[test]
    fn adaptive_threshold_mean_one_darker_pixel() {
        for y in 0..3 {
            for x in 0..3 {
                let mut image = GrayImage::from_pixel(3, 3, Luma([200u8]));
                image.put_pixel(x, y, Luma([100u8]));
                let binary = adaptive_threshold(&image, 1, 0, AdaptiveThresholdType::Mean);
                // All except the dark pixel have brightness >= their local mean
                let mut expected = GrayImage::from_pixel(3, 3, Luma::white());
                expected.put_pixel(x, y, Luma::black());
                assert_pixels_eq!(binary, expected);
            }
        }
    }

    #[test]
    fn adaptive_threshold_gaussian_specific_case() {
        let image = gray_image!(
            10,  20,  30;
            150, 160, 170;
            230, 240, 250
        );

        let binary = adaptive_threshold(&image, 1, 10, AdaptiveThresholdType::Gaussian);

        // Expected output verified with OpenCV python:
        // ```python
        // import cv2
        // import numpy as np
        // img = np.array([[10, 20, 30], [150, 160, 170], [230, 240, 250]], dtype=np.uint8)
        // th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 10)
        // print(th)
        // [[  0   0   0]
        //  [255 255 255]
        //  [255 255 255]]
        // ```
        let expected = gray_image!(
            0,   0,   0;
            255, 255, 255;
            255, 255, 255
        );

        assert_pixels_eq!(binary, expected);
    }

    #[test]
    fn adaptive_threshold_mean_one_lighter_pixel() {
        for y in 0..5 {
            for x in 0..5 {
                let mut image = GrayImage::from_pixel(5, 5, Luma([100u8]));
                image.put_pixel(x, y, Luma([200u8]));

                let binary = adaptive_threshold(&image, 1, 0, AdaptiveThresholdType::Mean);

                for yb in 0..5 {
                    for xb in 0..5 {
                        let output_intensity = binary.get_pixel(xb, yb)[0];
                        let is_light_pixel = xb == x && yb == y;
                        let local_mean_includes_light_pixel =
                            (yb as i32 - y as i32).abs() <= 1 && (xb as i32 - x as i32).abs() <= 1;

                        if is_light_pixel {
                            assert_eq!(output_intensity, 255);
                        } else if local_mean_includes_light_pixel {
                            assert_eq!(output_intensity, 0);
                        } else {
                            assert_eq!(output_intensity, 255);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_adaptive_thesholding_with_delta() {
        let mut image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        image.put_pixel(2, 2, Luma::black());

        // Test for Mean
        // big delta should make the threshold for the black pixel small enough to be white
        let binary_mean_1 = adaptive_threshold(&image, 1, 100, AdaptiveThresholdType::Mean);
        let expected_white = GrayImage::from_pixel(3, 3, Luma::white());
        assert_pixels_eq!(binary_mean_1, expected_white);

        // smaller delta should make the threshold for the pixel to be black
        let binary_mean_2 = adaptive_threshold(&image, 1, 50, AdaptiveThresholdType::Mean);
        let mut expected_black_corner = GrayImage::from_pixel(3, 3, Luma::white());
        expected_black_corner.put_pixel(2, 2, Luma::black());
        assert_pixels_eq!(binary_mean_2, expected_black_corner);

        // Test for Gaussian
        // as ditto
        let binary_gaussian_1 = adaptive_threshold(&image, 1, 100, AdaptiveThresholdType::Gaussian);
        assert_pixels_eq!(binary_gaussian_1, expected_white);

        let binary_gaussian_2 = adaptive_threshold(&image, 1, 20, AdaptiveThresholdType::Gaussian);
        assert_pixels_eq!(binary_gaussian_2, expected_black_corner);
    }

    #[test]
    fn test_histogram_lut_source_and_target_equal() {
        let mut histc = [0u32; 256];
        for i in 1..histc.len() {
            histc[i] = 2 * i as u32;
        }

        let lut = histogram_lut(&histc, &histc);
        let expected = (0..256).collect::<Vec<_>>();

        assert_eq!(&lut[0..256], &expected[0..256]);
    }

    #[test]
    fn test_histogram_lut_gradient_to_step_contrast() {
        let mut grad_histc = [0u32; 256];
        for i in 0..grad_histc.len() {
            grad_histc[i] = i as u32;
        }

        let mut step_histc = [0u32; 256];
        for i in 30..130 {
            step_histc[i] = 100;
        }
        for i in 130..256 {
            step_histc[i] = 200;
        }

        let lut = histogram_lut(&grad_histc, &step_histc);
        let mut expected = [0usize; 256];

        // No black pixels in either image
        expected[0] = 0;

        for i in 1..64 {
            expected[i] = 29;
        }
        for i in 64..128 {
            expected[i] = 30;
        }
        for i in 128..192 {
            expected[i] = 129;
        }
        for i in 192..256 {
            expected[i] = 130;
        }

        assert_eq!(&lut[0..256], &expected[0..256]);
    }

    fn constant_image(width: u32, height: u32, intensity: u8) -> GrayImage {
        GrayImage::from_pixel(width, height, Luma([intensity]))
    }

    #[test]
    fn test_kapur_constant() {
        assert_eq!(kapur_level(&constant_image(10, 10, 0)), 0);
        assert_eq!(kapur_level(&constant_image(10, 10, 128)), 0);
        assert_eq!(kapur_level(&constant_image(10, 10, 255)), 0);
    }

    #[test]
    fn test_otsu_constant() {
        // Variance is 0 at any threshold, and we
        // only increase the current threshold if we
        // see a strictly greater variance
        assert_eq!(otsu_level(&constant_image(10, 10, 0)), 0);
        assert_eq!(otsu_level(&constant_image(10, 10, 128)), 0);
        assert_eq!(otsu_level(&constant_image(10, 10, 255)), 0);
    }

    #[cfg_attr(miri, ignore = "assert_eq fails")]
    #[test]
    fn test_otsu_level_gradient() {
        let contents = (0u8..26u8).map(|x| x * 10u8).collect();
        let image = GrayImage::from_raw(26, 1, contents).unwrap();
        let level = otsu_level(&image);
        assert_eq!(level, 120);
    }

    #[test]
    fn test_threshold_0_image_0() {
        let expected = 0u8;
        let actual = threshold(&constant_image(10, 10, 0), 0, ThresholdType::Binary);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_0_image_1() {
        let expected = 255u8;
        let actual = threshold(&constant_image(10, 10, 1), 0, ThresholdType::Binary);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_threshold_255_image_255() {
        let expected = 0u8;
        let actual = threshold(&constant_image(10, 10, 255), 255, ThresholdType::Binary);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold() {
        let original_contents = (0u8..26u8).map(|x| x * 10u8).collect();
        let original = GrayImage::from_raw(26, 1, original_contents).unwrap();

        let expected_contents = vec![0u8; 13].into_iter().chain(vec![255u8; 13]).collect();

        let expected = GrayImage::from_raw(26, 1, expected_contents).unwrap();

        let actual = threshold(&original, 125u8, ThresholdType::Binary);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_stretch_contrast() {
        let input = gray_image!(1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 255);
        let expected = gray_image!(10u8, 10, 10, 11, 11, 12, 12, 13, 13, 13, 52, 120);
        assert_pixels_eq!(stretch_contrast(&input, 1, 255, 10, 120), expected);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use test::{Bencher, black_box};

    #[bench]
    fn bench_adaptive_threshold(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        let block_radius = 10;
        b.iter(|| {
            let thresholded =
                adaptive_threshold(&image, block_radius, 0, AdaptiveThresholdType::Mean);
            black_box(thresholded);
        });
    }

    #[bench]
    fn bench_match_histogram(b: &mut Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let matched = match_histogram(&image, &target);
            black_box(matched);
        });
    }

    #[bench]
    fn bench_match_histogram_mut(b: &mut Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let mut image = gray_bench_image(200, 200);
        b.iter(|| {
            match_histogram_mut(&mut image, &target);
        });
    }

    #[bench]
    fn bench_equalize_histogram(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let equalized = equalize_histogram(&image);
            black_box(equalized);
        });
    }

    #[bench]
    fn bench_equalize_histogram_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            equalize_histogram_mut(&mut image);
            black_box(());
        });
    }

    #[bench]
    fn bench_threshold(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let thresholded = threshold(&image, 125, ThresholdType::Binary);
            black_box(thresholded);
        });
    }

    #[bench]
    fn bench_threshold_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            threshold_mut(&mut image, 125, ThresholdType::Binary);
            black_box(());
        });
    }

    #[bench]
    fn bench_otsu_level(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let level = otsu_level(&image);
            black_box(level);
        });
    }

    #[bench]
    fn bench_kapur_level(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let level = kapur_level(&image);
            black_box(level);
        });
    }

    #[bench]
    fn bench_stretch_contrast(b: &mut Bencher) {
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let scaled = stretch_contrast(&image, 0, 255, 0, 255);
            black_box(scaled);
        });
    }

    #[bench]
    fn bench_stretch_contrast_mut(b: &mut Bencher) {
        let mut image = gray_bench_image(200, 200);
        b.iter(|| {
            stretch_contrast_mut(&mut image, 0, 255, 0, 255);
            black_box(());
        });
    }
}
