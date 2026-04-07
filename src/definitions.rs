//! Trait definitions and type aliases.

use crate::geometric_transformations::Border;
use crate::kernel::Kernel;
use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Rgb, Rgba};
use num::Num;

/// An `ImageBuffer` containing Pixels of type P with storage `Vec<P::Subpixel>`.
/// Most operations in this library only support inputs of type `Image`, rather
/// than arbitrary `image::GenericImage`s. This is obviously less flexible, but
/// has the advantage of allowing many functions to be more performant. We may want
/// to add more flexibility later, but this should not be at the expense of performance.
/// When specialisation lands we should be able to do this by defining traits for images
/// with contiguous storage.
pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

/// Image containers that can sample outside of their boundaries.
pub trait BoundaryAccess<P: Pixel> {
    /// Returns the pixel or falls back to the implementation defined by [`Border`].
    fn get_pixel_or_extend(&self, x: i64, y: i64, extend: Border<P>) -> P;
    /// Computes convolution with the specified kernel, accumulating into `acc`.
    /// The coordinate points to the location where the center of the kernel is matched.
    /// It is assumed that `acc` is filled with `K::zero()`.
    fn convolve_at<K: Copy + Num>(
        &self,
        kernel: Kernel<K>,
        acc: &mut [K],
        x: i64,
        y: i64,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>;
    /// Computes convolution with the specified horizontal kernel, accumulating into `acc`.
    /// The coordinate points to the location where the center of the kernel is matched.
    /// It is assumed that `acc` is filled with `K::zero()`.
    /// It is assumed that `y` is within image bounds; no assumptions are made about `x`.
    fn convolve_horizontal_at<K: Copy + Num>(
        &self,
        kernel: &[K],
        acc: &mut [K],
        x: i64,
        y: u32,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>;
    /// Computes convolution with the specified vertical kernel, accumulating into `acc`.
    /// The coordinate points to the location where the center of the kernel is matched.
    /// It is assumed that `acc` is filled with `K::zero()`.
    /// It is assumed that `x` is within image bounds; no assumptions are made about `y`.
    fn convolve_vertical_at<K: Copy + Num>(
        &self,
        kernel: &[K],
        acc: &mut [K],
        x: u32,
        y: i64,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>;
}
impl<P: Pixel, I: GenericImageView<Pixel = P>> BoundaryAccess<P> for I {
    fn get_pixel_or_extend(&self, x: i64, y: i64, extend: Border<P>) -> P {
        let (w, h) = self.dimensions();
        match extend {
            Border::Constant(default) => {
                if x < 0 || x >= w as i64 || y < 0 || y >= h as i64 {
                    default
                } else {
                    unsafe { self.unsafe_get_pixel(x as u32, y as u32) }
                }
            }
            Border::Replicate => {
                let x = x.clamp(0, w as i64 - 1) as u32;
                let y = y.clamp(0, h as i64 - 1) as u32;
                unsafe { self.unsafe_get_pixel(x, y) }
            }
            Border::Wrap => {
                let x = x.rem_euclid(w as i64) as u32;
                let y = y.rem_euclid(h as i64) as u32;
                unsafe { self.unsafe_get_pixel(x, y) }
            }
        }
    }
    fn convolve_at<K: Copy + Num>(
        &self,
        kernel: Kernel<K>,
        acc: &mut [K],
        x: i64,
        y: i64,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>,
    {
        fn accumulate<P: Pixel, K: Copy + Num>(acc: &mut [K], pixel: &P, weight: K)
        where
            P::Subpixel: Into<K>,
        {
            acc.iter_mut().zip(pixel.channels()).for_each(|(a, &c)| {
                *a = *a + c.into() * weight;
            });
        }
        let (w, h) = self.dimensions();
        let (w, h) = (w as i64, h as i64);
        let (k_width, k_height) = (kernel.width as i64, kernel.height as i64);
        let (x, y) = (x - k_width / 2, y - k_height / 2);
        match extend {
            Border::Constant(default) => {
                let k_y_min = y.max(0) - y;
                let k_y_max = ((y + k_height).min(h) - y).max(0);
                let k_x_min = x.max(0) - x;
                let k_x_max = ((x + k_width).min(w) - x).max(0);
                for k_y in 0..k_y_min {
                    for k_x in 0..k_width {
                        accumulate(acc, &default, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                }
                for k_y in k_y_min..k_y_max {
                    for k_x in 0..k_x_min {
                        accumulate(acc, &default, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                    // pixels intersecting with the kernel
                    for k_x in k_x_min..k_x_max {
                        let pixel =
                            unsafe { self.unsafe_get_pixel((x + k_x) as u32, (y + k_y) as u32) };
                        accumulate(acc, &pixel, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                    for k_x in k_x_max..k_width {
                        accumulate(acc, &default, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                }
                for k_y in k_y_max..k_height {
                    for k_x in 0..k_width {
                        accumulate(acc, &default, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                }
            }
            Border::Replicate => {
                for k_y in 0..k_height {
                    let y_p = (y + k_y).clamp(0, h - 1) as u32;
                    for k_x in 0..k_width {
                        let x_p = (x + k_x).clamp(0, w - 1) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y_p) };
                        accumulate(acc, &pixel, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                }
            }
            Border::Wrap => {
                for k_y in 0..k_height {
                    let y_p = (y + k_y).rem_euclid(h) as u32;
                    for k_x in 0..k_width {
                        let x_p = (x + k_x).rem_euclid(w) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y_p) };
                        accumulate(acc, &pixel, unsafe {
                            kernel.get_unchecked(k_x as u32, k_y as u32)
                        });
                    }
                }
            }
        }
    }
    fn convolve_horizontal_at<K: Copy + Num>(
        &self,
        kernel: &[K],
        acc: &mut [K],
        x: i64,
        y: u32,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>,
    {
        fn accumulate<P: Pixel, K: Copy + Num>(acc: &mut [K], pixel: &P, weight: K)
        where
            P::Subpixel: Into<K>,
        {
            acc.iter_mut().zip(pixel.channels()).for_each(|(a, &c)| {
                *a = *a + c.into() * weight;
            });
        }
        let w = self.width() as i64;
        let k_width = kernel.len() as i64;
        let x = x - k_width / 2;
        let check_left = x < 0;
        let check_right = x + k_width >= w;
        match extend {
            Border::Constant(default) => {
                let k_x_min = (x.max(0) - x) as usize;
                let k_x_max = ((x + k_width).min(w) - x).max(0) as usize;
                for k_x in 0..k_x_min {
                    accumulate(acc, &default, kernel[k_x]);
                }
                // pixels intersecting with the kernel
                for k_x in k_x_min..k_x_max {
                    let pixel = unsafe { self.unsafe_get_pixel((x + k_x as i64) as u32, y) };
                    accumulate(acc, &pixel, kernel[k_x]);
                }
                for k_x in k_x_max..k_width as usize {
                    accumulate(acc, &default, kernel[k_x]);
                }
            }
            Border::Replicate => match (check_left, check_right) {
                (false, false) => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (false, true) => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64).min(w - 1) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (true, false) => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64).max(0) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (true, true) => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64).clamp(0, w - 1) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
            },
            Border::Wrap => match (check_left, check_right) {
                (false, false) => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
                _ => {
                    for (k_x, &k) in kernel.iter().enumerate() {
                        let x_p = (x + k_x as i64).rem_euclid(w) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x_p, y) };
                        accumulate(acc, &pixel, k);
                    }
                }
            },
        }
    }
    fn convolve_vertical_at<K: Copy + Num>(
        &self,
        kernel: &[K],
        acc: &mut [K],
        x: u32,
        y: i64,
        extend: Border<P>,
    ) where
        P::Subpixel: Into<K>,
    {
        fn accumulate<P: Pixel, K: Copy + Num>(acc: &mut [K], pixel: &P, weight: K)
        where
            P::Subpixel: Into<K>,
        {
            acc.iter_mut().zip(pixel.channels()).for_each(|(a, &c)| {
                *a = *a + c.into() * weight;
            });
        }
        let h = self.height() as i64;
        let k_height = kernel.len() as i64;
        let y = y - k_height / 2;
        let check_up = y < 0;
        let check_down = y + k_height >= h;
        match extend {
            Border::Constant(default) => {
                let k_y_min = (y.max(0) - y) as usize;
                let k_y_max = ((y + k_height).min(h) - y).max(0) as usize;
                for k_y in 0..k_y_min {
                    accumulate(acc, &default, kernel[k_y]);
                }
                // pixels intersecting with the kernel
                for k_y in k_y_min..k_y_max {
                    let pixel = unsafe { self.unsafe_get_pixel(x, (y + k_y as i64) as u32) };
                    accumulate(acc, &pixel, kernel[k_y]);
                }
                for k_y in k_y_max..k_height as usize {
                    accumulate(acc, &default, kernel[k_y]);
                }
            }
            Border::Replicate => match (check_up, check_down) {
                (false, false) => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (false, true) => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64).min(h - 1) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (true, false) => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64).max(0) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
                (true, true) => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64).clamp(0, h - 1) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
            },
            Border::Wrap => match (check_up, check_down) {
                (false, false) => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
                _ => {
                    for (k_y, &k) in kernel.iter().enumerate() {
                        let y_p = (y + k_y as i64).rem_euclid(h) as u32;
                        let pixel = unsafe { self.unsafe_get_pixel(x, y_p) };
                        accumulate(acc, &pixel, k);
                    }
                }
            },
        }
    }
}

/// Pixels which have a named Black value.
pub trait HasBlack {
    /// Returns a black pixel of this type.
    fn black() -> Self;
}

/// Pixels which have a named White value.
pub trait HasWhite {
    /// Returns a white pixel of this type.
    fn white() -> Self;
}

macro_rules! impl_black_white {
    ($for_:ty, $min:expr, $max:expr) => {
        impl HasBlack for $for_ {
            fn black() -> Self {
                $min
            }
        }

        impl HasWhite for $for_ {
            fn white() -> Self {
                $max
            }
        }
    };
}

impl_black_white!(Luma<u8>, Luma([u8::MIN]), Luma([u8::MAX]));
impl_black_white!(Luma<u16>, Luma([u16::MIN]), Luma([u16::MAX]));
impl_black_white!(
    LumaA<u8>,
    LumaA([u8::MIN, u8::MAX]),
    LumaA([u8::MAX, u8::MAX])
);
impl_black_white!(
    LumaA<u16>,
    LumaA([u16::MIN, u16::MAX]),
    LumaA([u16::MAX, u16::MAX])
);
impl_black_white!(Rgb<u8>, Rgb([u8::MIN; 3]), Rgb([u8::MAX; 3]));
impl_black_white!(Rgb<u16>, Rgb([u16::MIN; 3]), Rgb([u16::MAX; 3]));
impl_black_white!(
    Rgba<u8>,
    Rgba([u8::MIN, u8::MIN, u8::MIN, u8::MAX]),
    Rgba([u8::MAX, u8::MAX, u8::MAX, u8::MAX])
);
impl_black_white!(
    Rgba<u16>,
    Rgba([u16::MIN, u16::MIN, u16::MIN, u16::MAX]),
    Rgba([u16::MAX, u16::MAX, u16::MAX, u16::MAX])
);

/// Something with a 2d position.
pub trait Position {
    /// x-coordinate.
    fn x(&self) -> u32;
    /// y-coordinate.
    fn y(&self) -> u32;
}

/// Something with a score.
pub trait Score {
    /// Score of this item.
    fn score(&self) -> f32;
}

/// A type to which we can clamp a value of type T.
/// Implementations are not required to handle `NaN`s gracefully.
pub trait Clamp<T> {
    /// Clamp `x` to a valid value for this type.
    fn clamp(x: T) -> Self;
}

/// Creates an implementation of Clamp<From> for type To.
macro_rules! implement_clamp {
    ($from:ty, $to:ty, $min:expr, $max:expr, $min_from:expr, $max_from:expr) => {
        impl Clamp<$from> for $to {
            fn clamp(x: $from) -> $to {
                if x < $max_from as $from {
                    if x > $min_from as $from {
                        x as $to
                    } else {
                        $min
                    }
                } else {
                    $max
                }
            }
        }
    };
}

/// Implements Clamp<T> for T, for all input types T.
macro_rules! implement_identity_clamp {
    ( $($t:ty),* ) => {
        $(
            impl Clamp<$t> for $t {
                fn clamp(x: $t) -> $t { x }
            }
        )*
    };
}

implement_clamp!(i16, u8, u8::MIN, u8::MAX, u8::MIN as i16, u8::MAX as i16);
implement_clamp!(u16, u8, u8::MIN, u8::MAX, u8::MIN as u16, u8::MAX as u16);
implement_clamp!(i32, u8, u8::MIN, u8::MAX, u8::MIN as i32, u8::MAX as i32);
implement_clamp!(u32, u8, u8::MIN, u8::MAX, u8::MIN as u32, u8::MAX as u32);
implement_clamp!(f32, u8, u8::MIN, u8::MAX, u8::MIN as f32, u8::MAX as f32);
implement_clamp!(f64, u8, u8::MIN, u8::MAX, u8::MIN as f64, u8::MAX as f64);

implement_clamp!(
    i32,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as i32,
    u16::MAX as i32
);
implement_clamp!(
    f32,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as f32,
    u16::MAX as f32
);
implement_clamp!(
    f64,
    u16,
    u16::MIN,
    u16::MAX,
    u16::MIN as f64,
    u16::MAX as f64
);

implement_clamp!(
    i32,
    i16,
    i16::MIN,
    i16::MAX,
    i16::MIN as i32,
    i16::MAX as i32
);

implement_identity_clamp!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

#[cfg(test)]
mod tests {
    use super::Clamp;

    #[test]
    fn test_clamp_f32_u8() {
        let t: u8 = Clamp::clamp(255f32);
        assert_eq!(t, 255u8);
        let u: u8 = Clamp::clamp(300f32);
        assert_eq!(u, 255u8);
        let v: u8 = Clamp::clamp(0f32);
        assert_eq!(v, 0u8);
        let w: u8 = Clamp::clamp(-5f32);
        assert_eq!(w, 0u8);
    }

    #[test]
    fn test_clamp_f32_u16() {
        let t: u16 = Clamp::clamp(65535f32);
        assert_eq!(t, 65535u16);
        let u: u16 = Clamp::clamp(300000f32);
        assert_eq!(u, 65535u16);
        let v: u16 = Clamp::clamp(0f32);
        assert_eq!(v, 0u16);
        let w: u16 = Clamp::clamp(-5f32);
        assert_eq!(w, 0u16);
    }
}
