//! Traits that I can't find elsewhere.

use image::{
    Rgb,
    Rgba,
    Luma,
    Pixel,
    Primitive
};

use std::{
    u8,
    u16
};

/// Used to name the type we get by replacing
/// the channel type of a given Pixel type.
pub trait WithChannel<C>: Pixel {
    type Pixel;
}

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

/// A type to which we can clamp an f32. Implementations
/// are not required to handle NaNs gracefully.
// TODO: Switch to using the conv crate when that lets
// TODO: you handle this ergonomically.
pub trait Clamp {
    fn clamp(x: f32) -> Self;
}

impl Clamp for u8 {
    fn clamp(x: f32) -> u8 {
        if x < 0f32 { return 0;}
        if x > 255f32 { return 255; }
        x as u8
    }
}

impl Clamp for u16 {
    fn clamp(x: f32) -> u16 {
        if x < 0f32 { return 0;}
        if x > 65535f32 { return 65535; }
        x as u16
    }
}

#[cfg(test)]
mod test {

    use super::Clamp;

    #[test]
    fn test_clamp_u8() {
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
    fn test_clamp_u16() {
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
