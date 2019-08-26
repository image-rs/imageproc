//! Functions for computing gradients of image intensities.

use crate::definitions::{HasBlack, Image};
use crate::filter::filter3x3;
use crate::map::{ChannelMap, WithChannel};
use image::{GenericImage, GenericImageView, GrayImage, Luma, Pixel};
use itertools::multizip;

/// Sobel filter for detecting vertical gradients.
///
/// Used by the [`vertical_sobel`](fn.vertical_sobel.html) function.
#[rustfmt::skip]
pub static VERTICAL_SOBEL: [i32; 9] = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1];

/// Sobel filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_sobel`](fn.horizontal_sobel.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_SOBEL: [i32; 9] = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1];

/// Scharr filter for detecting vertical gradients.
///
/// Used by the [`vertical_scharr`](fn.vertical_scharr.html) function.
#[rustfmt::skip]
pub static VERTICAL_SCHARR: [i32; 9] = [
    -3, -10,  -3,
     0,   0,   0,
     3,  10,   3];

/// Scharr filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_scharr`](fn.horizontal_scharr.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_SCHARR: [i32; 9] = [
     -3,  0,   3,
    -10,  0,  10,
     -3,  0,   3];

/// Prewitt filter for detecting vertical gradients.
///
/// Used by the [`vertical_prewitt`](fn.vertical_prewitt.html) function.
#[rustfmt::skip]
pub static VERTICAL_PREWITT: [i32; 9] = [
    -1, -1, -1,
     0,  0,  0,
     1,  1,  1];

/// Prewitt filter for detecting horizontal gradients.
///
/// Used by the [`horizontal_prewitt`](fn.horizontal_prewitt.html) function.
#[rustfmt::skip]
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

/// Convolves an image with the [`HORIZONTAL_SCHARR`](static.HORIZONTAL_SCHARR.html)
/// kernel to detect horizontal gradients.
pub fn horizontal_scharr(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &HORIZONTAL_SCHARR)
}

/// Convolves an image with the [`VERTICAL_SCHARR`](static.VERTICAL_SCHARR.html)
/// kernel to detect vertical gradients.
pub fn vertical_scharr(image: &GrayImage) -> Image<Luma<i16>> {
    filter3x3(image, &VERTICAL_SCHARR)
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
    gradients(image, &HORIZONTAL_SOBEL, &VERTICAL_SOBEL, |p| p)
}

/// Computes per-channel gradients using Sobel filters and calls `f`
/// to compute each output pixel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::gradients::{sobel_gradient_map};
/// use image::Luma;
/// use std::cmp;
///
/// // A shallow horizontal gradient in the red
/// // channel, a steeper vertical gradient in the
/// // blue channel, constant in the green channel.
/// let input = rgb_image!(
///     [0, 5, 0], [1, 5, 0], [2, 5, 0];
///     [0, 5, 2], [1, 5, 2], [2, 5, 2];
///     [0, 5, 4], [1, 5, 4], [2, 5, 4]
/// );
///
/// // Computing independent per-channel gradients.
/// let channel_gradient = rgb_image!(type: u16,
///     [ 4,  0,  8], [ 8,  0,  8], [ 4,  0,  8];
///     [ 4,  0, 16], [ 8,  0, 16], [ 4,  0, 16];
///     [ 4,  0,  8], [ 8,  0,  8], [ 4,  0,  8]
/// );
///
/// assert_pixels_eq!(
///     sobel_gradient_map(&input, |p| p),
///     channel_gradient
/// );
///
/// // Defining the gradient of an RGB image to be the
/// // mean of its per-channel gradients.
/// let mean_gradient = gray_image!(type: u16,
///     4, 5, 4;
///     6, 8, 6;
///     4, 5, 4
/// );
///
/// assert_pixels_eq!(
///     sobel_gradient_map(&input, |p| {
///         let mean = (p[0] + p[1] + p[2]) / 3;
///         Luma([mean])
///     }),
///     mean_gradient
/// );
///
/// // Defining the gradient of an RGB image to be the pixelwise
/// // maximum of its per-channel gradients.
/// let max_gradient = gray_image!(type: u16,
///      8,  8,  8;
///     16, 16, 16;
///      8,  8,  8
/// );
///
/// assert_pixels_eq!(
///     sobel_gradient_map(&input, |p| {
///         let max = cmp::max(cmp::max(p[0], p[1]), p[2]);
///         Luma([max])
///     }),
///     max_gradient
/// );
/// # }
pub fn sobel_gradient_map<P, F, Q>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    Q: Pixel + 'static,
    ChannelMap<P, u16>: HasBlack,
    F: Fn(ChannelMap<P, u16>) -> Q,
{
    gradients(image, &HORIZONTAL_SOBEL, &VERTICAL_SOBEL, f)
}

/// Returns the magnitudes of gradients in an image using Prewitt filters.
pub fn prewitt_gradients(image: &GrayImage) -> Image<Luma<u16>> {
    gradients(image, &HORIZONTAL_PREWITT, &VERTICAL_PREWITT, |p| p)
}

// TODO: Returns directions as well as magnitudes.
// TODO: Support filtering without allocating a fresh image - filtering functions could
// TODO: take some kind of pixel-sink. This would allow us to compute gradient magnitudes
// TODO: and directions without allocating intermediates for vertical and horizontal gradients.
fn gradients<P, F, Q>(
    image: &Image<P>,
    horizontal_kernel: &[i32; 9],
    vertical_kernel: &[i32; 9],
    f: F,
) -> Image<Q>
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    Q: Pixel + 'static,
    ChannelMap<P, u16>: HasBlack,
    F: Fn(ChannelMap<P, u16>) -> Q,
{
    let horizontal: Image<ChannelMap<P, i16>> = filter3x3::<_, _, i16>(image, horizontal_kernel);
    let vertical: Image<ChannelMap<P, i16>> = filter3x3::<_, _, i16>(image, vertical_kernel);

    let (width, height) = image.dimensions();
    let mut out = Image::<Q>::new(width, height);

    // This would be more concise using itertools::multizip over image pixels, but that increased runtime by around 20%
    for y in 0..height {
        for x in 0..width {
            // JUSTIFICATION
            //  Benefit
            //      Using checked indexing here makes this sobel_gradients 1.1x slower,
            //      as measured by bench_sobel_gradients
            //  Correctness
            //      x and y are in bounds for image by construction,
            //      vertical and horizontal are the result of calling filter3x3 on image,
            //      and filter3x3 returns an image of the same size as its input
            let (h, v) = unsafe {
                (
                    horizontal.unsafe_get_pixel(x, y),
                    vertical.unsafe_get_pixel(x, y),
                )
            };
            let mut p = ChannelMap::<P, u16>::black();

            for (h, v, p) in multizip((h.channels(), v.channels(), p.channels_mut())) {
                *p = gradient_magnitude(*h as f32, *v as f32);
            }

            // JUSTIFICATION
            //  Benefit
            //      Using checked indexing here makes this sobel_gradients 1.1x slower,
            //      as measured by bench_sobel_gradients
            //  Correctness
            //      x and y are in bounds for image by construction,
            //      and out has the same dimensions
            unsafe {
                out.unsafe_put_pixel(x, y, f(p));
            }
        }
    }

    out
}

#[inline]
fn gradient_magnitude(dx: f32, dy: f32) -> u16 {
    (dx.powi(2) + dy.powi(2)).sqrt() as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::{ImageBuffer, Luma};
    use test::{black_box, Bencher};

    #[rustfmt::skip::macros(gray_image)]
    #[test]
    fn test_gradients_constant_image() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0i16]));
        assert_pixels_eq!(horizontal_sobel(&image), expected);
        assert_pixels_eq!(vertical_sobel(&image), expected);
        assert_pixels_eq!(horizontal_scharr(&image), expected);
        assert_pixels_eq!(vertical_scharr(&image), expected);
        assert_pixels_eq!(horizontal_prewitt(&image), expected);
        assert_pixels_eq!(vertical_prewitt(&image), expected);
    }

    #[test]
    fn test_horizontal_sobel_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -4, -8, -4;
            -4, -8, -4;
            -4, -8, -4);

        let filtered = horizontal_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_sobel_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -4, -4, -4;
            -8, -8, -8;
            -4, -4, -4);

        let filtered = vertical_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_scharr_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -16, -32, -16;
            -16, -32, -16;
            -16, -32, -16);

        let filtered = horizontal_scharr(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_scharr_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -16, -16, -16;
            -32, -32, -32;
            -16, -16, -16);

        let filtered = vertical_scharr(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_prewitt_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -3, -6, -3;
            -3, -6, -3;
            -3, -6, -3);

        let filtered = horizontal_prewitt(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_prewitt_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -3, -3, -3;
            -6, -6, -6;
            -3, -3, -3);

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
