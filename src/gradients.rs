//! Functions for computing gradients of image intensities.

use crate::definitions::{Clamp, HasBlack, Image};
use crate::filter::{filter, filter3x3, Kernel, SeparableKernel};
use crate::map::{ChannelMap, WithChannel};
use image::{GenericImage, GenericImageView, GrayImage, Luma, Pixel};
use itertools::multizip;

/// Different kernels for finding detecting gradients in images.
#[derive(Debug, Copy, Clone)]
pub enum GradientKernel {
    /// The 3x3 Sobel kernel
    /// See <https://en.wikipedia.org/wiki/Sobel_operator>
    Sobel,
    /// The 3x3 Scharr kernel
    Scharr,
    /// The 3x3 Prewitt kernel
    Prewitt,
    /// The 2x2 Roberts kernel
    /// See <https://en.wikipedia.org/wiki/Roberts_cross>
    Roberts,
}
impl<K> From<GradientKernel> for SeparableKernel<K>
where
    K: From<i8> + Copy,
{
    fn from(value: GradientKernel) -> Self {
        let x = match value {
            GradientKernel::Sobel => SeparableKernel {
                horizontal_kernel: Kernel::new(vec![-1, 0, 1, -2, 0, 2, -1, 0, 1], 3, 3),
                vertical_kernel: Kernel::new(vec![-1, -2, -1, 0, 0, 0, 1, 2, 1], 3, 3),
            },
            GradientKernel::Scharr => SeparableKernel {
                horizontal_kernel: Kernel::new(vec![-3, 0, 3, -10, 0, 10, -3, 0, 3], 3, 3),
                vertical_kernel: Kernel::new(vec![-3, -10, -3, 0, 0, 0, 3, 10, 3], 3, 3),
            },
            GradientKernel::Prewitt => SeparableKernel {
                horizontal_kernel: Kernel::new(vec![-1, 0, 1, -1, 0, 1, -1, 0, 1], 3, 3),
                vertical_kernel: Kernel::new(vec![-1, -1, -1, 0, 0, 0, 1, 1, 1], 3, 3),
            },
            GradientKernel::Roberts => SeparableKernel {
                horizontal_kernel: Kernel::new(vec![0, 1, -1, -0], 2, 2),
                vertical_kernel: Kernel::new(vec![1, 0, 0, -1], 2, 2),
            },
        };

        x.map(&K::from)
    }
}
impl GradientKernel {
    /// A slice of all the [`GradientKernel`] variants.
    pub const ALL: [GradientKernel; 4] = [
        GradientKernel::Sobel,
        GradientKernel::Scharr,
        GradientKernel::Prewitt,
        GradientKernel::Roberts,
    ];

    /// Returns the horizontal separable matrix of the gradient kernel.
    pub fn as_horizontal_kernel<K>(&self) -> Kernel<K>
    where
        K: From<i8> + Copy,
    {
        SeparableKernel::from(*self).horizontal_kernel
    }
    /// Returns the vertical separable matrix of the gradient kernel.
    pub fn as_vertical_kernel<K>(&self) -> Kernel<K>
    where
        K: From<i8> + Copy,
    {
        SeparableKernel::from(*self).vertical_kernel
    }
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
/// use imageproc::gradients::{gradients, GradientKernel};
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
/// let gradient_kernel = GradientKernel::Sobel;
///
/// assert_pixels_eq!(
///     gradients(&input, gradient_kernel, |p| p),
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
///     gradients(&input, gradient_kernel, |p| {
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
///     gradients(&input, gradient_kernel, |p| {
///         let max = cmp::max(cmp::max(p[0], p[1]), p[2]);
///         Luma([max])
///     }),
///     max_gradient
/// );
/// # }
pub fn gradients<P, F, Q>(
    image: &Image<P>,
    separable_kernel: impl Into<SeparableKernel<i32>>,
    f: F,
) -> Image<Q>
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16>,
    Q: Pixel,
    ChannelMap<P, u16>: HasBlack,
    F: Fn(ChannelMap<P, u16>) -> Q,
{
    let separable_kernel: SeparableKernel<i32> = separable_kernel.into();

    let horizontal: Image<ChannelMap<P, i16>> = filter(
        image,
        &separable_kernel.horizontal_kernel,
        |channel, acc| *channel = <i16 as Clamp<i32>>::clamp(acc),
    );
    let vertical: Image<ChannelMap<P, i16>> =
        filter(image, &separable_kernel.vertical_kernel, |channel, acc| {
            *channel = <i16 as Clamp<i32>>::clamp(acc)
        });

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
    use crate::{
        filter::{horizontal_filter, vertical_filter},
        utils::gray_bench_image,
    };
    use image::{ImageBuffer, Luma};

    #[test]
    fn test_gradients_constant_image_sobel() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(gradients(&image, GradientKernel::Sobel, |p| p), expected);
    }
    #[test]
    fn test_gradients_constant_image_scharr() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(gradients(&image, GradientKernel::Scharr, |p| p), expected);
    }
    #[test]
    fn test_gradients_constant_image_prewitt() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(gradients(&image, GradientKernel::Prewitt, |p| p), expected);
    }
    #[test]
    fn test_gradients_constant_image_roberts() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0u16]));

        assert_pixels_eq!(gradients(&image, GradientKernel::Roberts, |p| p), expected);
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

        let filtered = filter3x3(&image, &GradientKernel::Sobel.as_horizontal_kernel::<i16>());
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

        let filtered = filter3x3(&image, &GradientKernel::Sobel.as_vertical_kernel::<i16>());
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

        let filtered = filter3x3(
            &image,
            &GradientKernel::Scharr.as_horizontal_kernel::<i16>(),
        );
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

        let filtered = filter3x3(&image, &GradientKernel::Scharr.as_vertical_kernel::<i16>());
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

        let filtered = filter3x3(
            &image,
            &GradientKernel::Prewitt.as_horizontal_kernel::<i16>(),
        );
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

        let filtered = filter3x3(&image, &GradientKernel::Prewitt.as_vertical_kernel::<i16>());
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
            let gradients = gradients(&image, GradientKernel::Sobel, |p| p);
            black_box(gradients);
        });
    }
}
