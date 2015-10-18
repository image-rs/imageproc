//! Trait definitions and type aliases.

use image::{
    Rgb,
    Rgba,
    Luma,
    Pixel,
    Primitive,
    ImageBuffer
};
use num::{
    Bounded,
    NumCast
};
use std::{
    u8,
    u16
};

use std::cmp::Ordering::{
    Equal, Greater, Less
};

/// An ImageBuffer containing Pixels of type P with storage
/// Vec<P::Subpixel>.
// TODO: This produces a compiler warning about trait bounds
// TODO: not being enforced in type definitions. In this case
// TODO: they are. Can we get rid of the warning?
pub type VecBuffer<P: Pixel> = ImageBuffer<P, Vec<P::Subpixel>>;

/// Used to name the type we get by replacing
/// the channel type of a given Pixel type.
pub trait WithChannel<C: Primitive>: Pixel {
    type Pixel: Pixel<Subpixel=C> + 'static;
}

pub type ChannelMap<Pix, Sub> = <Pix as WithChannel<Sub>>::Pixel;

impl<T, U> WithChannel<U> for Rgb<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Rgb<U>;
}

impl<T, U> WithChannel<U> for Rgba<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Rgba<U>;
}

impl<T, U> WithChannel<U> for Luma<T>
    where T: Primitive + 'static,
          U: Primitive + 'static {
    type Pixel = Luma<U>;
}

/// Pixels which have a named Black value.
pub trait HasBlack {
    fn black() -> Self;
}

/// Pixels which have a named White value.
pub trait HasWhite {
    fn white() -> Self;
}

impl HasBlack for Luma<u8> {
    fn black() -> Self {
        Luma([0u8])
    }
}

impl HasWhite for Luma<u8> {
    fn white() -> Self {
        Luma([u8::MAX])
    }
}

impl HasBlack for Luma<u16> {
    fn black() -> Self {
        Luma([0u16])
    }
}

impl HasWhite for Luma<u16> {
    fn white() -> Self {
        Luma([u16::MAX])
    }
}

impl HasBlack for Rgb<u8> {
    fn black() -> Self {
        Rgb([0u8; 3])
    }
}

impl HasWhite for Rgb<u8> {
    fn white() -> Self {
        Rgb([u8::MAX; 3])
    }
}

/// A type to which we can clamp a value of type T.
/// Implementations are not required to handle NaNs gracefully.
pub trait Clamp<T> {
    fn clamp(x: T) -> Self;
}

/// Creates an implementation of Clamp<From> for type To.
// TODO: improve performance
macro_rules! implement_clamp {
    ($from:ty, $to:ty) => (
        impl Clamp<$from> for $to {
            fn clamp(x: $from) -> $to {
                clamp_impl(x)
            }
        }
    )
}

implement_clamp!(f32, u8);
implement_clamp!(f32, u16);
implement_clamp!(f64, u8);
implement_clamp!(f64, u16);
implement_clamp!(i32, u8);
implement_clamp!(i32, u16);
implement_clamp!(i32, i16);
implement_clamp!(u16, u8);

/// Clamp a value from a type with larger range to one with
/// a smaller range. Deliberately not exported - should be used
/// via the Clamp trait.
fn clamp_impl<From: NumCast + PartialOrd, To: NumCast + Bounded>(x: From) -> To {
    let to_max = <From as NumCast>::from(To::max_value()).unwrap();
    let to_min = <From as NumCast>::from(To::min_value()).unwrap();
    let clamped = partial_max(partial_min(x, to_max).unwrap(), to_min).unwrap();
    To::from(clamped).unwrap()
}

/// Copied from std::cmp as it's now deprecated.
fn partial_max<T: PartialOrd>(v1: T, v2: T) -> Option<T> {
    match v1.partial_cmp(&v2) {
        Some(Equal) | Some(Less) => Some(v2),
        Some(Greater) => Some(v1),
        None => None
    }
}

/// Copied from std::cmp as it's now deprecated.
fn partial_min<T: PartialOrd>(v1: T, v2: T) -> Option<T> {
    match v1.partial_cmp(&v2) {
        Some(Less) | Some(Equal) => Some(v1),
        Some(Greater) => Some(v2),
        None => None
    }
}

#[cfg(test)]
mod test {

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
