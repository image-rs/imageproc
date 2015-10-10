//! Traits that I can't find elsewhere.

use image::{
    Rgb,
    Rgba,
    Luma,
    Pixel,
    Primitive
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

/// A type that can be converted to an f32
pub trait ToFloat: Copy {
    fn to_float(self) -> f32;
}

/// A type to which we can clamp an f32. Implementations
/// are not required to handle NaNs gracefully.
pub trait Clamp {
    fn clamp(x: f32) -> Self;
}

impl ToFloat for u8 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}

impl ToFloat for i8 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}

impl ToFloat for u16 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}

impl ToFloat for i16 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}


impl ToFloat for u32 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}

impl ToFloat for i32 {
    fn to_float(self) -> f32 {
        return self as f32;
    }
}

impl ToFloat for f32 {
    fn to_float(self) -> f32 {
        return self;
    }
}

impl Clamp for u8 {
    fn clamp(x: f32) -> u8 {
        if x < 0f32 { return 0;}
        if x > 255f32 { return 255; }
        x as u8
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
}
