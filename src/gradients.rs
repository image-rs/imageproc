//! Functions for computing gradients of image intensities.

use crate::definitions::{Clamp, HasBlack, Image};
use crate::filter::{filter3x3, Kernel};
use crate::map::{ChannelMap, WithChannel};
use image::{GenericImage, GenericImageView, GrayImage, Luma, Pixel, Primitive};
use itertools::multizip;
use num::Num;

// region Gradient kernels

/// Sobel kernel for detecting vertical gradients.
///
/// Used by the [`vertical_sobel`](fn.vertical_sobel.html) function.
#[rustfmt::skip]
pub static VERTICAL_SOBEL: Kernel<i32> = Kernel {
    data:
        &[-1, -2, -1,
           0,  0,  0,
           1,  2,  1],
    width: 3,
    height: 3
};

/// Sobel kernel for detecting horizontal gradients.
///
/// Used by the [`horizontal_sobel`](fn.horizontal_sobel.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_SOBEL: Kernel<i32> = Kernel {
    data:
    &[-1, 0, 1,
      -2, 0, 2,
      -1, 0, 1],
    width: 3,
    height: 3
};

/// Scharr kernel for detecting vertical gradients.
///
/// Used by the [`vertical_scharr`](fn.vertical_scharr.html) function.
#[rustfmt::skip]
pub static VERTICAL_SCHARR: Kernel<i32> = Kernel {
    data:
    &[-3, -10, -3,
       0,   0,  0,
       3,  10,  3],
    width: 3,
    height: 3
};


/// Scharr kernel for detecting horizontal gradients.
///
/// Used by the [`horizontal_scharr`](fn.horizontal_scharr.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_SCHARR: Kernel<i32> = Kernel {
    data:
    &[ -3,  0,   3,
      -10,  0,  10,
       -3,  0,   3],
    width: 3,
    height: 3
};


/// Prewitt kernel for detecting vertical gradients.
///
/// Used by the [`vertical_prewitt`](fn.vertical_prewitt.html) function.
#[rustfmt::skip]
pub static VERTICAL_PREWITT: Kernel<i32> = Kernel {
    data:
    &[-1, -1, -1,
       0,  0,  0,
       1,  1,  1],
    width: 3,
    height: 3
};


/// Prewitt kernel for detecting horizontal gradients.
///
/// Used by the [`horizontal_prewitt`](fn.horizontal_prewitt.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_PREWITT: Kernel<i32> = Kernel {
    data:
    &[-1, 0, 1,
      -1, 0, 1,
      -1, 0, 1],
    width: 3,
    height: 3
};


/// Roberts kernel for detecting vertical gradients.
///
/// Used by the [`vertical_roberts`](fn.vertical_roberts.html) function.
#[rustfmt::skip]
pub static VERTICAL_ROBERTS: Kernel<i32> = Kernel {
    data:
    &[ 0, 1,
      -1, 0],
    width: 2,
    height: 2
};


/// Roberts kernel for detecting horizontal gradients.
///
/// Used by the [`horizontal_roberts`](fn.horizontal_roberts.html) function.
#[rustfmt::skip]
pub static HORIZONTAL_ROBERTS: Kernel<i32> = Kernel {
    data:
    &[ 1,  0,
       0, -1],
    width: 2,
    height: 2
};

// endregion

/// Used for specifying a gradient kernel for new API
pub enum GradientKernel {
    Sobel,
    Scharr,
    Prewitt,
    Roberts
}

impl GradientKernel {
    pub fn horizontal_kernel(&self) -> &Kernel<i32> {
        match self {
            GradientKernel::Sobel   => {&HORIZONTAL_SOBEL}
            GradientKernel::Scharr  => {&HORIZONTAL_SCHARR}
            GradientKernel::Prewitt => {&HORIZONTAL_PREWITT}
            GradientKernel::Roberts => {&HORIZONTAL_ROBERTS}
        }
    }

    pub fn vertical_kernel(&self) -> &Kernel<i32> {
        match self {
            GradientKernel::Sobel   => {&VERTICAL_SOBEL}
            GradientKernel::Scharr  => {&VERTICAL_SCHARR}
            GradientKernel::Prewitt => {&VERTICAL_PREWITT}
            GradientKernel::Roberts => {&VERTICAL_ROBERTS}
        }
    }

    pub fn horizontal_gradient(&self, image: &GrayImage) -> Image<Luma<i16>> {
        self.horizontal_kernel().filter::<_, _, Luma<i16>>(
            image,
            |channel, acc| *channel = u16::clamp(acc)
        )
    }

    pub fn vertical_gradient(&self, image: &GrayImage) -> Image<Luma<i16>> {
        self.vertical_kernel().filter::<_, _, Luma<i16>>(
            image,
            |channel, acc| *channel = u16::clamp(acc)
        )
    }

    /// Computes the vertical and horizontal gradients and calls `f` with the gradient vector to compute each output
    /// pixel.
    ///
    /// Classic use cases for `gradient_map` provide a function `f` that computes a norm or the angle of the
    /// gradient vector.
    // todo: Give examples.
    pub fn gradient_map<P, F, Q>(&self, image: &GrayImage, f: F) -> Image<Q>
        where
            P: Pixel<Subpixel  = u8> + WithChannel<u16> + WithChannel<i16>,
            F: Fn(ChannelMap<P, u16>) -> Q,
            Q: Pixel,
            ChannelMap<P, u16>: HasBlack,
    {
        let horizontal = self.horizontal_gradient(image);
        let vertical = self.vertical_gradient(image);

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
                    *p = f(*h, *v);
                }

                // JUSTIFICATION
                //  Benefit
                //      Using checked indexing here makes this sobel_gradients 1.1x slower,
                //      as measured by bench_sobel_gradients
                //  Correctness
                //      x and y are in bounds for image by construction,
                //      and out has the same dimensions
                unsafe {
                    out.unsafe_put_pixel(x, y, p);
                }
            }
        }

        out

    }


    /// Computes the magnitude of the gradient at every point. Uses the usual euclidean ($L_2$) norm.
    pub fn gradient_magnitude(&self, image: &GrayImage) -> Image<Luma<i16>> {
        self.gradient_map(image, |h, v| f32::sqrt(h*h + v*v)  )
    }

    // ToDo: Implement gradient magnitude AND angle, as both can be computed simultaneously.
    // ToDo: Get peer review regarding intermediate data types and `Clamp`. I think it is not correct as written.

    // TODO: Returns directions as well as magnitudes.
    // TODO: Support filtering without allocating a fresh image - filtering functions could
    //       take some kind of pixel-sink. This would allow us to compute gradient magnitudes
    //       and directions without allocating intermediates for vertical and horizontal gradients.
    // TODO: As an alternative to the above, reuse the allocated `horizontal` and `vertical` to return
    //       Both magnitude and angle.
}


/// A synonym for `method.horizontal_gradient(image)`
pub fn horizontal_gradient(image: &GrayImage, method: GradientKernel ) -> Image<Luma<i16>> {
    method.horizontal_gradient(image)
}


/// A synonym for `method.vertical_gradient(image)`
pub fn vertical_gradient(image: &GrayImage, method: GradientKernel ) -> Image<Luma<i16>> {
    method.vertical_gradient(image)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::{ImageBuffer, Luma};
    use test::{black_box, Bencher};
    use GradientKernel::*;

    #[rustfmt::skip::macros(gray_image)]
    #[test]
    fn test_gradients_constant_image() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0i16]));
        assert_pixels_eq!(Sobel.horizontal_gradient(&image), expected);
        assert_pixels_eq!(Sobel.vertical_gradient(&image), expected);
        assert_pixels_eq!(Scharr.horizontal_gradient(&image), expected);
        assert_pixels_eq!(Scharr.vertical_gradient(&image), expected);
        assert_pixels_eq!(Prewitt.horizontal_gradient(&image), expected);
        assert_pixels_eq!(Prewitt.vertical_gradient(&image), expected);
        assert_pixels_eq!(Roberts.horizontal_gradient(&image), expected);
        assert_pixels_eq!(Roberts.vertical_gradient(&image), expected);
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

        let filtered = Sobel.horizontal_gradient(&image);
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

        let filtered = Sobel.vertical_gradient(&image);
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

        let filtered = Scharr.horizontal_gradient(&image);
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

        let filtered = Scharr.vertical_gradient(&image);
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

        let filtered = Prewitt.horizontal_gradient(&image);
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

        let filtered = Prewitt.vertical_gradient(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_roberts_gradient_image() {
        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -2, -2, -3;
            -2, -2, -3;
             1,  1,  0);

        let filtered = Roberts.horizontal_gradient(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_roberts_gradient_image() {
        let image = gray_image!(
            3, 6, 9;
            2, 5, 8;
            1, 4, 7);

        let expected = gray_image!(type: i16,
            4, 4, 1;
            4, 4, 1;
            3, 3, 0);

        let filtered = Roberts.vertical_gradient(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_sobel_gradients(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let gradients = Sobel.gradient_magnitude(&image);
            black_box(gradients);
        });
    }
}
