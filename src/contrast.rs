//! Functions for manipulating the contrast of images

use image::{
    GenericImage,
    GrayImage,
    ImageBuffer,
    Luma
};

/// Returns the cumulative histogram of grayscale values in an 8bpp
/// grayscale image
fn cumulative_histogram<I: GenericImage<Pixel=Luma<u8>> + 'static>
    (image: &I) -> [i32;256] {

    let mut hist = [0i32;256];

    for pix in image.pixels() {
        hist[pix.2[0] as usize] += 1;
    }

    for i in 1..hist.len() {
        hist[i] += hist[i - 1];
    }

    hist
}

/// Equalises the histogram of an 8bpp grayscale image in place
/// https://en.wikipedia.org/wiki/Histogram_equalization
pub fn equalize_histogram_mut<I: GenericImage<Pixel=Luma<u8>> + 'static>
    (image: &mut I) {

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

/// Equalises the histogram of an 8bpp grayscale image
/// https://en.wikipedia.org/wiki/Histogram_equalization
pub fn equalize_histogram<I: GenericImage<Pixel=Luma<u8>> + 'static>
    (image: &I) -> GrayImage {
    let mut out: GrayImage = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    equalize_histogram_mut(&mut out);
    out
}

#[cfg(test)]
mod test {

    use super::{
        cumulative_histogram,
        equalize_histogram
    };
    use utils::{
        gray_bench_image
    };
    use image::{
        GrayImage,
        ImageBuffer
    };
    use test;

    #[test]
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

    #[bench]
    fn bench_equalize_histogram(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let equalized = equalize_histogram(&image);
            test::black_box(equalized);
            });
    }
}
