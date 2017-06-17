//! Functions for computing gradients of image intensities.

use image::{
    GenericImage,
    GrayImage,
    ImageBuffer,
    Luma
};
use definitions::Image;
use filter::filter3x3;

/// Sobel filter for detecting vertical gradients.
///
/// Used by the [`vertical_sobel`](fn.vertical_sobel.html) function.
pub static VERTICAL_SOBEL: [i32; 9] = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1];

/// Sobel filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_sobel`](fn.horizontal_sobel.html) function.
pub static HORIZONTAL_SOBEL: [i32; 9] = [
     -1, 0, 1,
     -2, 0, 2,
     -1, 0, 1];

/// Prewitt filter for detecting vertical gradients.
///
/// Used by the [`vertical_prewitt`](fn.vertical_prewitt.html) function.
pub static VERTICAL_PREWITT: [i32; 9] = [
    -1, -1, -1,
     0,  0,  0,
     1,  1,  1];

/// Prewitt filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_prewitt`](fn.horizontal_prewitt.html) function.
pub static HORIZONTAL_PREWITT: [i32; 9] = [
     -1, 0, 1,
     -1, 0, 1,
     -1, 0, 1];

/// Convolves an image with the [`HORIZONTAL_SOBEL`](static.HORIZONTAL_SOBEL.html)
/// kernel to detect horizontal gradients.
pub fn horizontal_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &HORIZONTAL_SOBEL)
}

/// Convolves an image with the [`VERTICAL_SOBEL`](static.VERTICAL_SOBEL.html)
/// kernel to detect vertical gradients.
pub fn vertical_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &VERTICAL_SOBEL)
}

/// Convolves an image with the [`HORIZONTAL_PREWITT`](static.HORIZONTAL_PREWITT.html)
/// kernel to detect horizontal gradients.
pub fn horizontal_prewitt(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &HORIZONTAL_PREWITT)
}

/// Convolves an image with the [`VERTICAL_PREWITT`](static.VERTICAL_PREWITT.html)
/// kernel to detect vertical gradients.
pub fn vertical_prewitt(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &VERTICAL_PREWITT)
}

/// Returns the magnitudes of gradients in an image using Sobel filters.
pub fn sobel_gradients(image: &GrayImage) -> Image<Luma<u16>> {
    gradients(image, &HORIZONTAL_SOBEL, &VERTICAL_SOBEL)
}

/// Returns the magnitudes of gradients in an image using Prewitt filters.
pub fn prewitt_gradients(image: &GrayImage) -> Image<Luma<u16>> {
    gradients(image, &HORIZONTAL_PREWITT, &VERTICAL_PREWITT)
}

// TODO: Returns directions as well as magnitudes.
// TODO: Support filtering without allocating a fresh image - filtering functions could
// TODO: take some kind of pixel-sink. This would allow us to compute gradient magnitudes
// TODO: and directions without allocating intermediates for vertical and horizontal gradients.
fn gradients(image: &GrayImage, horizontal_kernel: &[i32; 9], vertical_kernel: &[i32; 9]) -> Image<Luma<u16>> {
    let horizontal: Image<Luma<i16>> = filter3x3(image, horizontal_kernel);
    let vertical: Image<Luma<i16>> = filter3x3(image, vertical_kernel);

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    // This would be more concise using itertools::multizip, but that increased runtime by around 20%
    for y in 0..height {
        for x in 0..width {
            unsafe {
                let h = horizontal.unsafe_get_pixel(x, y)[0];
                let v = vertical.unsafe_get_pixel(x, y)[0];
                let m = gradient_magnitude(h as f32, v as f32);
                out.unsafe_put_pixel(x, y, Luma([m]));
            }
        }
    }

    out
}

fn gradient_magnitude(dx: f32, dy: f32) -> u16 {
    (dx.powi(2) + dy.powi(2)).sqrt() as u16
}

#[cfg(test)]
mod test {
    use super::*;
    use image::{
        ImageBuffer,
        Luma
    };
    use test::{Bencher, black_box};
    use utils::gray_bench_image;

    #[test]
    fn test_gradients_constant_image() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0i16]));
        assert_pixels_eq!(horizontal_sobel(&image), expected);
        assert_pixels_eq!(vertical_sobel(&image), expected);
        assert_pixels_eq!(horizontal_prewitt(&image), expected);
        assert_pixels_eq!(vertical_prewitt(&image), expected);
    }

    #[test]
    fn test_horizontal_sobel_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image_i16!(
            -4i16, -8i16, -4i16;
            -4i16, -8i16, -4i16;
            -4i16, -8i16, -4i16);

        let filtered = horizontal_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_sobel_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image_i16!(
            -4i16, -4i16, -4i16;
            -8i16, -8i16, -8i16;
            -4i16, -4i16, -4i16);

        let filtered = vertical_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_prewitt_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image_i16!(
            -3i16, -6i16, -3i16;
            -3i16, -6i16, -3i16;
            -3i16, -6i16, -3i16);

        let filtered = horizontal_prewitt(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_prewitt_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image_i16!(
            -3i16, -3i16, -3i16;
            -6i16, -6i16, -6i16;
            -3i16, -3i16, -3i16);

        let filtered = vertical_prewitt(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_sobel_gradients(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let gradients = sobel_gradients(&image);
            black_box(gradients);
            });
    }
}
