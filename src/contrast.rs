//! Functions for manipulating the contrast of images.
use std::cmp::{max, min};
use image::{GenericImage, GrayImage, ImageBuffer, Luma};
use definitions::{HasWhite, HasBlack};
use integralimage::integral_image;

/// Returns the Otsu threshold level of an 8bpp image.
/// This threshold will optimally binarize an image that
/// contains two classes of pixels which have distributions
/// with equal variances. For details see:
/// Xu, X., et al. Pattern recognition letters 32.7 (2011)
pub fn otsu_level<I>(image: &I) -> u8
    where I: GenericImage<Pixel = Luma<u8>>
{
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
pub fn threshold<I>(image: &I, thresh: u8) -> GrayImage
    where I: GenericImage<Pixel = Luma<u8>>
{
    let mut out: GrayImage = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    threshold_mut(&mut out, thresh);
    out
}

/// Mutates given image to form a binarized version produced by applying
/// the given threshold.
pub fn threshold_mut<I>(image: &mut I, thresh: u8)
    where I: GenericImage<Pixel = Luma<u8>>
{
    for y in 0..image.height() {
        for x in 0..image.width() {
            unsafe {
                if image.unsafe_get_pixel(x, y)[0] as u8 <= thresh {
                    image.unsafe_put_pixel(x, y, Luma([0]));
                } else {
                    image.unsafe_put_pixel(x, y, Luma([255]));
                }
            }
        }
    }
}

/// Applies an adaptive threshold to an image.
///
/// This algorithms compares each pixels color with the average color of the `block_size` * `block_size`
/// surrounding pixels (= the local threshold). If the pixel if above the threshold, it will have a
/// value of 255 in the output image, otherwise 0.
pub fn adaptive_threshold<I>(image: &I, block_size: u32) -> GrayImage
    where I: GenericImage<Pixel = Luma<u8>>
{
    let integral = integral_image(image);
    let mut out = ImageBuffer::from_pixel(image.width(), image.height(), Luma::black());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let current_pixel = image.get_pixel(x, y);
            // Traverse all neighbors in blocksize x blocksize
            let (y_low, y_high) = (max(0, y as i32 - (block_size as i32)) as u32, min(image.height(), y + (block_size)));
            let (x_low, x_high) = (max(0, x as i32 - (block_size as i32)) as u32, min(image.width(), x + (block_size)));

            // Number of pixels in the block, adjusted for edge cases.
            let w = (y_high - y_low) * (x_high - x_low);

            // I(A..D)
            let a = integral.get_pixel(x_low, y_low)[0] as i32;
            let b = integral.get_pixel(x_high - 1, y_low)[0] as i32;
            let c = integral.get_pixel(x_high - 1, y_high - 1)[0] as i32;
            let d = integral.get_pixel(x_low, y_high - 1)[0] as i32;

            let mean = (c - b - d + a) / w as i32;

            if current_pixel[0] as u32 >= mean as u32 {
                out.put_pixel(x, y, Luma::white());
            }
        }
    }
    out
}

/// Returns the histogram of grayscale values in an 8bpp
/// grayscale image.
pub fn histogram<I>(image: &I) -> [i32; 256]
    where I: GenericImage<Pixel = Luma<u8>>
{
    let mut hist = [0i32; 256];

    for pix in image.pixels() {
        hist[pix.2[0] as usize] += 1;
    }

    hist
}

/// Returns the cumulative histogram of grayscale values in an 8bpp
/// grayscale image.
pub fn cumulative_histogram<I>(image: &I) -> [i32; 256]
    where I: GenericImage<Pixel = Luma<u8>>
{
    let mut hist = histogram(image);

    for i in 1..hist.len() {
        hist[i] += hist[i - 1];
    }

    hist
}

/// Equalises the histogram of an 8bpp grayscale image in place.
/// https://en.wikipedia.org/wiki/Histogram_equalization
pub fn equalize_histogram_mut<I>(image: &mut I)
    where I: GenericImage<Pixel = Luma<u8>>
{
    let hist = cumulative_histogram(image);
    let total = hist[255] as f32;

    for y in 0..image.height() {
        for x in 0..image.width() {
            let original = unsafe { image.unsafe_get_pixel(x, y)[0] as usize };
            let fraction = hist[original] as f32 / total;
            let out = f32::min(255f32, 255f32 * fraction);
            unsafe {
                image.unsafe_put_pixel(x, y, Luma([out as u8]));
            }
        }
    }
}

/// Equalises the histogram of an 8bpp grayscale image.
/// https://en.wikipedia.org/wiki/Histogram_equalization
pub fn equalize_histogram<I>(image: &I) -> GrayImage
    where I: GenericImage<Pixel = Luma<u8>>
{
    let mut out: GrayImage = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    equalize_histogram_mut(&mut out);
    out
}

/// Adjusts contrast of an 8bpp grayscale image in place so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram_mut<I, J>(image: &mut I, target: &J)
    where I: GenericImage<Pixel = Luma<u8>>,
          J: GenericImage<Pixel = Luma<u8>>
{
    let image_histc = cumulative_histogram(image);
    let target_histc = cumulative_histogram(target);
    let lut = histogram_lut(&image_histc, &target_histc);

    for y in 0..image.height() {
        for x in 0..image.width() {
            unsafe {
                let pix = image.unsafe_get_pixel(x, y)[0] as usize;
                image.unsafe_put_pixel(x, y, Luma([lut[pix] as u8]));
            }
        }
    }
}

/// Adjusts contrast of an 8bpp grayscale image so that its
/// histogram is as close as possible to that of the target image.
pub fn match_histogram<I, J>(image: &I, target: &J) -> GrayImage
    where I: GenericImage<Pixel = Luma<u8>>,
          J: GenericImage<Pixel = Luma<u8>>
{
    let mut out: GrayImage = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    match_histogram_mut(&mut out, target);
    out
}

/// l = histogram_lut(s, t) is chosen so that target_histc[l[i]] / sum(target_histc)
/// is as close as possible to source_histc[i] / sum(source_histc).
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
            y = y + 1;
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
mod test {

    use super::{cumulative_histogram, equalize_histogram, histogram, histogram_lut, otsu_level,
                threshold};
    use utils::gray_bench_image;
    use image::{GrayImage, ImageBuffer};
    use test;

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
}
