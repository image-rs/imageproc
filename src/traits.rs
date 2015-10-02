//! Traits that I can't find elsewhere.

/// A type that can be converted to an f32
pub trait ToFloat: Copy {
    fn to_float(self) -> f32;
}

/// A type to which we can clamp an f32
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
