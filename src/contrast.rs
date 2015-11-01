//! Functions for manipulating the contrast of images.

use image::{GenericImage, GrayImage, ImageBuffer, Luma};

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
            if image.get_pixel(x, y)[0] as u8 <= thresh {
                image.put_pixel(x, y, Luma([0]));
            } else {
                image.put_pixel(x, y, Luma([255]));
            }
        }
    }
}

/// Returns the histogram of grayscale values in an 8pp
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
            let original = image.get_pixel(x, y)[0] as usize;
            let fraction = hist[original] as f32 / total;
            let out = f32::min(255f32, 255f32 * fraction);
            image.put_pixel(x, y, Luma([out as u8]));
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

#[cfg(test)]
mod test {

    use super::{cumulative_histogram, equalize_histogram, histogram, otsu_level, threshold};
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
    fn test_histogram(){
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
    fn test_otsu_level(){
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
    fn test_threshold(){
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
