//! Functions for computing gradients of image intensities.

use crate::definitions::{Clamp, HasBlack, Image};
use crate::filter::{filter, filter_clamped};
use crate::kernel::{self, Kernel};
use crate::map::{ChannelMap, WithChannel};
use image::{GenericImage, GenericImageView, GrayImage, Luma, Pixel};
use itertools::multizip;

/// A special version of `gradient()` function for greyscale images which doesn't require giving a
/// pixel mapping function.
pub fn gradients_greyscale<P, F, Q>(
    image: &GrayImage,
    kernel1: Kernel<i32>,
    kernel2: Kernel<i32>,
) -> Image<Luma<u16>> {
    gradients(image, kernel1, kernel2, |p| p)
}

// TODO: Returns directions as well as magnitudes.
// TODO: Support filtering without allocating a fresh image - filtering functions could
// TODO: take some kind of pixel-sink. This would allow us to compute gradient magnitudes
// TODO: and directions without allocating intermediates for vertical and horizontal gradients.
//
/// Computes per-channel gradients using the given horizontal and vertical kernels and calls `f`
/// to compute each output pixel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::gradients::gradients;
/// use imageproc::kernel::Kernel;
/// use imageproc::kernel;
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
/// let horizontal_kernel = kernel::SOBEL_HORIZONTAL_3X3;
/// let vertical_kernel = kernel::SOBEL_VERTICAL_3X3;
///
/// assert_pixels_eq!(
///     gradients(&input, horizontal_kernel, vertical_kernel, |p| p),
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
///     gradients(&input, horizontal_kernel, vertical_kernel, |p| {
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
///     gradients(&input, horizontal_kernel, vertical_kernel, |p| {
///         let max = cmp::max(cmp::max(p[0], p[1]), p[2]);
///         Luma([max])
///     }),
///     max_gradient
/// );
/// # }
pub fn gradients<P, F, Q>(
    image: &Image<P>,
    kernel1: Kernel<i32>,
    kernel2: Kernel<i32>,
    f: F,
) -> Image<Q>
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16>,
    Q: Pixel,
    ChannelMap<P, u16>: HasBlack,
    F: Fn(ChannelMap<P, u16>) -> Q,
{
    let pass1: Image<ChannelMap<P, i16>> = filter(image, kernel1, <i16 as Clamp<i32>>::clamp);
    let pass2: Image<ChannelMap<P, i16>> = filter(image, kernel2, <i16 as Clamp<i32>>::clamp);

    let (width, height) = image.dimensions();
    let mut out = Image::<Q>::new(width, height);

    // This would be more concise using itertools::multizip over image pixels, but that increased runtime by around 20%
    for y in 0..height {
        for x in 0..width {
            // JUSTIFICATION
            //  Benefit
            //      Using checked indexing here makes this gradients 1.1x slower,
            //      as measured by bench_sobel_gradients
            //  Correctness
            //      x and y are in bounds for image by construction,
            //      vertical and horizontal are the result of calling filter_clamped on image,
            //      and filter_clamped returns an image of the same size as its input
            let (h, v) = unsafe { (pass1.unsafe_get_pixel(x, y), pass2.unsafe_get_pixel(x, y)) };
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

/// Convolves an image with the [`HORIZONTAL_SOBEL`](static.HORIZONTAL_SOBEL.html)
/// kernel to detect horizontal gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::SOBEL_HORIZONTAL_3X3)` which is more flexible and explicit"
)]
pub fn horizontal_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::SOBEL_HORIZONTAL_3X3)
}

/// Convolves an image with the [`VERTICAL_SOBEL`](static.VERTICAL_SOBEL.html)
/// kernel to detect vertical gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::SOBEL_VERTICAL_3X3)` which is more flexible and explicit"
)]
pub fn vertical_sobel(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::SOBEL_VERTICAL_3X3)
}

/// Convolves an image with the [`HORIZONTAL_SCHARR`](static.HORIZONTAL_SCHARR.html)
/// kernel to detect horizontal gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::SCHARR_HORIZONTAL_3X3)` which is more flexible and explicit"
)]
pub fn horizontal_scharr(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::SCHARR_HORIZONTAL_3X3)
}

/// Convolves an image with the [`VERTICAL_SCHARR`](static.VERTICAL_SCHARR.html)
/// kernel to detect vertical gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::SCHARR_VERTICAL_3X3)` which is more flexible and explicit"
)]
pub fn vertical_scharr(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::SCHARR_VERTICAL_3X3)
}

/// Convolves an image with the [`HORIZONTAL_PREWITT`](static.HORIZONTAL_PREWITT.html)
/// kernel to detect horizontal gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::PREWITT_HORIZONTAL_3X3)` which is more flexible and explicit"
)]
pub fn horizontal_prewitt(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::PREWITT_HORIZONTAL_3X3)
}

/// Convolves an image with the [`VERTICAL_PREWITT`](static.VERTICAL_PREWITT.html)
/// kernel to detect vertical gradients.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `filter_clamped(image, kernel::PREWITT_VERTICAL_3X3)` which is more flexible and explicit"
)]
pub fn vertical_prewitt(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::PREWITT_VERTICAL_3X3)
}

/// Returns the magnitudes of gradients in an image using Sobel filters.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `gradients_greyscale(image, kernel::SOBEL_HORIZONTAL_3X3, Kernel::SOBEL_VERTICAL_3X3)` which is more flexible and explicit"
)]
pub fn sobel_gradients(image: &GrayImage) -> Image<Luma<u16>> {
    gradients(
        image,
        kernel::SOBEL_HORIZONTAL_3X3,
        kernel::SOBEL_VERTICAL_3X3,
        |p| p,
    )
}

/// Returns the magnitudes of gradients in an image using Prewitt filters.
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `gradients_greyscale(image, kernel::PREWITT_HORIZONTAL_3X3, Kernel::PREWITT_VERTICAL_3X3)` which is more flexible and explicit"
)]
pub fn prewitt_gradients(image: &GrayImage) -> Image<Luma<u16>> {
    gradients(
        image,
        kernel::PREWITT_HORIZONTAL_3X3,
        kernel::PREWITT_VERTICAL_3X3,
        |p| p,
    )
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
#[deprecated(
    since = "0.25.0",
    note = "users should instead use `gradients(image, kernel::SOBEL_HORIZONTAL_3X3, Kernel::SOBEL_VERTICAL_3X3, f)` which is more flexible and explicit"
)]
pub fn sobel_gradient_map<P, F, Q>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16>,
    Q: Pixel,
    ChannelMap<P, u16>: HasBlack,
    F: Fn(ChannelMap<P, u16>) -> Q,
{
    gradients(
        image,
        kernel::SOBEL_HORIZONTAL_3X3,
        kernel::SOBEL_VERTICAL_3X3,
        f,
    )
}

#[cfg(test)]
mod tests {
    use crate::filter::filter_clamped;
    #[cfg(feature = "rayon")]
    use crate::filter::filter_clamped_parallel;

    use super::*;
    use image::{ImageBuffer, Luma};

    #[test]
    fn test_gradients_constant_image_sobel() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(
            gradients(
                &image,
                kernel::SOBEL_HORIZONTAL_3X3,
                kernel::SOBEL_VERTICAL_3X3,
                |p| p
            ),
            expected
        );
    }
    #[test]
    fn test_gradients_constant_image_scharr() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(
            gradients(
                &image,
                kernel::SCHARR_HORIZONTAL_3X3,
                kernel::SCHARR_VERTICAL_3X3,
                |p| p
            ),
            expected
        );
    }
    #[test]
    fn test_gradients_constant_image_prewitt() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(
            gradients(
                &image,
                kernel::PREWITT_HORIZONTAL_3X3,
                kernel::PREWITT_VERTICAL_3X3,
                |p| p
            ),
            expected
        );
    }
    #[test]
    fn test_gradients_constant_image_roberts() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(
            gradients(
                &image,
                kernel::ROBERTS_HORIZONTAL_2X2,
                kernel::ROBERTS_VERTICAL_2X2,
                |p| p
            ),
            expected
        );
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

        let filtered = filter_clamped(&image, kernel::SOBEL_HORIZONTAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_horizontal_sobel_gradient_image_parallel() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -4, -8, -4;
            -4, -8, -4;
            -4, -8, -4);

        let filtered = filter_clamped_parallel(&image, kernel::SOBEL_HORIZONTAL_3X3);
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

        let filtered = filter_clamped(&image, kernel::SOBEL_VERTICAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_vertical_sobel_gradient_image_parallel() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -4, -4, -4;
            -8, -8, -8;
            -4, -4, -4);

        let filtered = filter_clamped_parallel(&image, kernel::SOBEL_VERTICAL_3X3);
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

        let filtered = filter_clamped(&image, kernel::SCHARR_HORIZONTAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_horizontal_scharr_gradient_image_parallel() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -16, -32, -16;
            -16, -32, -16;
            -16, -32, -16);

        let filtered = filter_clamped_parallel(&image, kernel::SCHARR_HORIZONTAL_3X3);
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

        let filtered = filter_clamped(&image, kernel::SCHARR_VERTICAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_vertical_scharr_gradient_image_parallel() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -16, -16, -16;
            -32, -32, -32;
            -16, -16, -16);

        let filtered = filter_clamped_parallel(&image, kernel::SCHARR_VERTICAL_3X3);
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

        let filtered = filter_clamped(&image, kernel::PREWITT_HORIZONTAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_horizontal_prewitt_gradient_image_parallel() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -3, -6, -3;
            -3, -6, -3;
            -3, -6, -3);

        let filtered = filter_clamped_parallel(&image, kernel::PREWITT_HORIZONTAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_vertical_prewitt_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -3, -3, -3;
            -6, -6, -6;
            -3, -3, -3);

        let filtered = filter_clamped(&image, kernel::PREWITT_VERTICAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_vertical_prewitt_gradient_image_parallel() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            -3, -3, -3;
            -6, -6, -6;
            -3, -3, -3);

        let filtered = filter_clamped_parallel(&image, kernel::PREWITT_VERTICAL_3X3);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_roberts_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            0, -3, -3;
            1, -2, -2;
            1, -2, -2);

        let filtered = filter_clamped(&image, kernel::ROBERTS_HORIZONTAL_2X2);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_horizontal_roberts_gradient_image_parallel() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            0, -3, -3;
            1, -2, -2;
            1, -2, -2);

        let filtered = filter_clamped_parallel(&image, kernel::ROBERTS_HORIZONTAL_2X2);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_roberts_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            0, 3, 3;
            1, 4, 4;
            1, 4, 4);

        let filtered = filter_clamped(&image, kernel::ROBERTS_VERTICAL_2X2);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_vertical_roberts_gradient_image_parallel() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            0, 3, 3;
            1, 4, 4;
            1, 4, 4);

        let filtered = filter_clamped_parallel(&image, kernel::ROBERTS_VERTICAL_2X2);
        assert_pixels_eq!(filtered, expected);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_sobel_gradients(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let gradients = gradients(
                &image,
                kernel::SOBEL_HORIZONTAL_3X3,
                kernel::SOBEL_VERTICAL_3X3,
                |p| p,
            );
            black_box(gradients);
        });
    }
}
