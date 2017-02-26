//! Functions for manipulating the contrast of images.

use std::cmp::{min, max};
use image::{GrayImage, ImageBuffer, Luma};
use definitions::{HasBlack, HasWhite};
use integralimage::{integral_image, sum_image_pixels};
use rayon::prelude::*;

/// Applies an adaptive threshold to an image.
///
/// This algorithm compares each pixel's brightness with the average brightness of the pixels
/// in the (2 * `block_radius` + 1) square block centered on it. If the pixel if at least as bright
/// as the threshold then it will have a value of 255 in the output image, otherwise 0.
pub fn adaptive_threshold(image: &GrayImage, block_radius: u32) -> GrayImage {
     assert!(block_radius > 0);
     let integral = integral_image(image);
     let mut out = ImageBuffer::from_pixel(image.width(), image.height(), Luma::black());
     for y in 0..image.height() {
         for x in 0..image.width() {
             let current_pixel = image.get_pixel(x, y);
             // Traverse all neighbors in (2 * block_radius + 1) x (2 * block_radius + 1)
             let (y_low, y_high) = (max(0, y as i32 - (block_radius as i32)) as u32,
                                    min(image.height() - 1, y + block_radius));
             let (x_low, x_high) = (max(0, x as i32 - (block_radius as i32)) as u32,
                                    min(image.width() - 1, x + block_radius));

             // Number of pixels in the block, adjusted for edge cases.
             let w = (y_high - y_low + 1) * (x_high - x_low + 1);
             let mean = sum_image_pixels(&integral, x_low, y_low, x_high, y_high) / w;

             if current_pixel[0] as u32 >= mean as u32 {
                 out.put_pixel(x, y, Luma::white());
             }
         }
     }
     out
}

/// Returns the Otsu threshold level of an 8bpp image.
/// This threshold will optimally binarize an image that
/// contains two classes of pixels which have distributions
/// with equal variances. For details see:
/// Xu, X., et al. Pattern recognition letters 32.7 (2011)
pub fn otsu_level(image: &GrayImage) -> u8 {
    let hist = histogram(image);
    let levels = hist.len();
    let mut histc = [0i32; 256];
    let mut meanc = [0.0f64; 256];
    let mut pixel_count = hist[0] as f64;

    histc[0] = hist[0];

    for i in 1..levels {
        pixel_count += hist[i] as f64;
        histc[i] = histc[i - 1] + hist[i];
        meanc[i] = meanc[i - 1] + (hist[i] as f64) * (i as f64);
    }

    let mut sigma_max = -1f64;
    let mut otsu = 0f64;
    let mut otsu2 = 0f64;

    for i in 0..levels {
        meanc[i] /= pixel_count;

        let p0 = (histc[i] as f64) / pixel_count;
        let p1 = 1f64 - p0;
        let mu0 = meanc[i] / p0;
        let mu1 = (meanc[levels - 1] / pixel_count - meanc[i]) / p1;

        let sigma = p0 * p1 * (mu0 - mu1).powi(2);
        if sigma >= sigma_max {
            if sigma > sigma_max {
                otsu = i as f64;
                sigma_max = sigma;
            } else {
                otsu2 = i as f64;
            }
        }
    }
    ((otsu + otsu2) / 2.0).ceil() as u8
}

/// Returns a binarized image from an input 8bpp grayscale image
/// obtained by applying the given threshold.
pub fn threshold(image: &GrayImage, thresh: u8) -> GrayImage {
    let mut out = image.clone();
    threshold_mut(&mut out, thresh);
    out
}

/// Mutates given image to form a binarized version produced by applying
/// the given threshold.
pub fn threshold_mut(image: &mut GrayImage, thresh: u8) {
    for p in image.iter_mut() {
        *p = if *p <= thresh { 0 } else { 255 };
    };
}

/// Returns the histogram of grayscale values in an 8bpp
/// grayscale image.
pub fn histogram(image: &GrayImage) -> [i32; 256] {
    let mut hist = [0i32; 256];

    for pix in image.iter() {
        hist[*pix as usize] += 1;
    }

    hist
}

/// Returns the cumulative histogram of grayscale values in an 8bpp
/// grayscale image.
pub fn cumulative_histogram(image: &GrayImage) -> [i32; 256] {
    let mut hist = histogram(image);

    for i in 1..hist.len() {
        hist[i] += hist[i - 1];
    }

    hist
}

/// Equalises the histogram of an 8bpp grayscale image in place. See also
/// [histogram equalization (wikipedia)](https://en.wikipedia.org/wiki/Histogram_equalization).
pub fn equalize_histogram_mut(image: &mut GrayImage) {
    let hist = cumulative_histogram(image);
    let total = hist[255] as f32;

    image.par_iter_mut().for_each(|p| {
        let fraction = hist[*p as usize] as f32 / total;
        *p = (f32::min(255f32, 255f32 * fraction)) as u8;
    });
}

/// Equalises the histogram of an 8bpp grayscale image. See also
/// [histogram equalization (wikipedia)](https://en.wikipedia.org/wiki/Histogram_equalization).
pub fn equalize_histogram(image: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    equalize_histogram_mut(&mut out);
    out
}

/// Adjusts contrast of an 8bpp grayscale image in place so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram_mut(image: &mut GrayImage, target: &GrayImage) {
    let image_histc = cumulative_histogram(image);
    let target_histc = cumulative_histogram(target);
    let lut = histogram_lut(&image_histc, &target_histc);

    for p in image.iter_mut() {
        *p = lut[*p as usize] as u8;
    };
}

/// Adjusts contrast of an 8bpp grayscale image so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram(image: &GrayImage, target: &GrayImage) -> GrayImage {
    let mut out = image.clone();
    match_histogram_mut(&mut out, target);
    out
}

/// `l = histogram_lut(s, t)` is chosen so that `target_histc[l[i]] / sum(target_histc)`
/// is as close as possible to `source_histc[i] / sum(source_histc)`.
fn histogram_lut(source_histc: &[i32; 256], target_histc: &[i32; 256]) -> [usize; 256] {
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
        }
        else {
            let prev_dist = f32::abs(prev_target_fraction - source_fraction);
            let dist = f32::abs(target_fraction - source_fraction);
            if prev_dist < dist {
                lut[s] = y - 1;
            }
            else {
                lut[s] = y;
            }
        }
    }

    lut
}

#[cfg(test)]
mod test {
    use super::*;
    use definitions::{HasBlack, HasWhite};
    use utils::gray_bench_image;
    use image::{GrayImage, ImageBuffer, Luma};
    use test;

    #[test]
    fn adaptive_threshold_constant() {
        let image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        let binary = adaptive_threshold(&image, 1);
        let expected = GrayImage::from_pixel(3, 3, Luma::white());
        assert_pixels_eq!(expected, binary);
    }

    #[test]
    fn adaptive_threshold_one_darker_pixel() {
        for y in 0..3 {
            for x in 0..3 {
                let mut image = GrayImage::from_pixel(3, 3, Luma([200u8]));
                image.put_pixel(x, y, Luma([100u8]));
                let binary = adaptive_threshold(&image, 1);
                // All except the dark pixel have brightness >= their local mean
                let mut expected = GrayImage::from_pixel(3, 3, Luma::white());
                expected.put_pixel(x, y, Luma::black());
                assert_pixels_eq!(binary, expected);
            }
        }
    }

    #[test]
    fn adaptive_threshold_one_lighter_pixel() {
        for y in 0..5 {
            for x in 0..5 {
                let mut image = GrayImage::from_pixel(5, 5, Luma([100u8]));
                image.put_pixel(x, y, Luma([200u8]));

                let binary = adaptive_threshold(&image, 1);

                for yb in 0..5 {
                    for xb in 0..5 {
                        let output_intensity = binary.get_pixel(xb, yb)[0];

                        let is_light_pixel = xb == x && yb == y;

                        let local_mean_includes_light_pixel =
                            (yb as i32 - y as i32).abs() <= 1 &&
                            (xb as i32 - x as i32).abs() <= 1;

                        if is_light_pixel {
                            assert_eq!(output_intensity, 255);
                        }
                        else if local_mean_includes_light_pixel {
                            assert_eq!(output_intensity, 0);
                        }
                        else {
                            assert_eq!(output_intensity, 255);
                        }
                    }
                }
            }
        }
    }

    #[bench]
    fn bench_adaptive_threshold(b: &mut test::Bencher) {
        let image = gray_bench_image(200, 200);
        let block_radius = 10;
        b.iter(|| {
            let thresholded = adaptive_threshold(&image, block_radius);
            test::black_box(thresholded);
        });
    }

    #[bench]
    fn bench_match_histogram(b: &mut test::Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let matched = match_histogram(&image, &target);
            test::black_box(matched);
        });
    }

    #[bench]
    fn bench_match_histogram_mut(b: &mut test::Bencher) {
        let target = GrayImage::from_pixel(200, 200, Luma([150]));
        let mut image = gray_bench_image(200, 200);
        b.iter(|| {
            match_histogram_mut(&mut image, &target);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_cumulative_histogram() {
        let image: GrayImage = ImageBuffer::from_raw(5, 1, vec![
            1u8, 2u8, 3u8, 2u8, 1u8]).unwrap();

        let hist = cumulative_histogram(&image);

        assert_eq!(hist[0], 0);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 4);
        assert_eq!(hist[3], 5);
        assert!(hist.iter().skip(4).all(|x| *x == 5));
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_histogram() {
        let image: GrayImage = ImageBuffer::from_raw(5, 1, vec![
            1u8, 2u8, 3u8, 2u8, 1u8]).unwrap();

        let hist = histogram(&image);

        assert_eq!(hist[0], 0);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 2);
        assert_eq!(hist[3], 1);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_histogram_lut_source_and_target_equal() {
        let mut histc = [0i32; 256];
        for i in 1..histc.len() {
            histc[i] = 2 * i as i32;
        }

        let lut = histogram_lut(&histc, &histc);
        let expected = (0..256).collect::<Vec<_>>();

        assert_eq!(&lut[0..256], &expected[0..256]);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_histogram_lut_gradient_to_step_contrast() {
        let mut grad_histc = [0i32; 256];
        for i in 0..grad_histc.len() {
            grad_histc[i] = i as i32;
        }

        let mut step_histc = [0i32; 256];
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

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_otsu_level() {
        let image: GrayImage = ImageBuffer::from_raw(26, 1,
            vec![0u8, 10u8, 20u8, 30u8, 40u8, 50u8, 60u8, 70u8,
                80u8, 90u8, 100u8, 110u8, 120u8, 130u8, 140u8,
                150u8, 160u8, 170u8, 180u8, 190u8, 200u8,  210u8,
                220u8,  230u8,  240u8,  250u8]).unwrap();
        let level = otsu_level(&image);
        assert_eq!(level, 125);
    }

    #[bench]
    fn bench_otsu_level(b: &mut test::Bencher) {
        let image = gray_bench_image(200, 200);
        b.iter(|| {
            let level = otsu_level(&image);
            test::black_box(level);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_threshold() {
        let original: GrayImage = ImageBuffer::from_raw(26, 1,
            vec![0u8, 10u8, 20u8, 30u8, 40u8, 50u8, 60u8, 70u8,
                80u8, 90u8, 100u8, 110u8, 120u8, 130u8, 140u8,
                150u8, 160u8, 170u8, 180u8, 190u8, 200u8,  210u8,
                220u8,  230u8,  240u8,  250u8]).unwrap();
        let expected: GrayImage = ImageBuffer::from_raw(26, 1,
            vec![0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                0u8, 0u8, 0u8, 0u8, 0u8, 255u8, 255u8,
                255u8, 255u8, 255u8, 255u8, 255u8, 255u8,  255u8,
                255u8,  255u8,  255u8,  255u8]).unwrap();

        let actual = threshold(&original, 125u8);
        assert_pixels_eq!(expected, actual);
    }

    #[bench]
    fn bench_equalize_histogram(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let equalized = equalize_histogram(&image);
            test::black_box(equalized);
        });
    }

    #[bench]
    fn bench_equalize_histogram_mut(b: &mut test::Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            test::black_box(equalize_histogram_mut(&mut image));
        });
    }

    #[bench]
    fn bench_threshold(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let thresholded = threshold(&image, 125);
            test::black_box(thresholded);
        });
    }

    #[bench]
    fn bench_threshold_mut(b: &mut test::Bencher) {
        let mut image = gray_bench_image(500, 500);
        b.iter(|| {
            test::black_box(threshold_mut(&mut image, 125));
        });
    }
}
