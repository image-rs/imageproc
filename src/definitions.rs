//! Trait definitions and type aliases.

use image::{Bgr, Bgra, ImageBuffer, Luma, LumaA, Pixel, Rgb, Rgba};
use std::{i16, u16, u8};

/// An `ImageBuffer` containing Pixels of type P with storage `Vec<P::Subpixel>`.
/// Most operations in this library only support inputs of type `Image`, rather
/// than arbitrary `image::GenericImage`s. This is obviously less flexible, but
/// has the advantage of allowing many functions to be more performant. We may want
/// to add more flexibility later, but this should not be at the expense of performance.
/// When specialisation lands we should be able to do this by defining traits for images
/// with contiguous storage.
pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

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
impl_black_white!(Bgr<u8>, Bgr([u8::MIN; 3]), Bgr([u8::MAX; 3]));
impl_black_white!(Bgr<u16>, Bgr([u16::MIN; 3]), Bgr([u16::MAX; 3]));
impl_black_white!(
    Bgra<u8>,
    Bgra([u8::MIN, u8::MIN, u8::MIN, u8::MAX]),
    Bgra([u8::MAX, u8::MAX, u8::MAX, u8::MAX])
);
impl_black_white!(
    Bgra<u16>,
    Bgra([u16::MIN, u16::MIN, u16::MIN, u16::MAX]),
    Bgra([u16::MAX, u16::MAX, u16::MAX, u16::MAX])
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
