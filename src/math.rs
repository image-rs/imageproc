//! Assorted mathematical helper functions.

use conv::ValueInto;
use num::Zero;
pub use nalgebra::{Mat2, Vec2};
use nalgebra::{Eye, Inv};

// L2 norm of a vector.
pub fn l2_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x * x).sqrt()
}

/// Helper for a conversion that we know can't fail.
pub fn cast<T, U>(x: T) -> U where T: ValueInto<U> {
    match x.value_into() {
        Ok(y) => y,
        Err(_) => panic!("Failed to convert"),
    }
}

/// A 2d affine transformation.
// TODO: Should we switch to homogeneous coordinates?
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Affine2 {
    pub linear: Mat2<f32>,
    pub translation: Vec2<f32>
}

impl Affine2 {

    /// Constructs an Affine2 with given linear transformation
    /// and translation.
    pub fn new(linear: Mat2<f32>, translation: Vec2<f32>) -> Affine2 {
        Affine2 { linear: linear, translation: translation }
    }

    /// The identity transformation.
    pub fn identity() -> Affine2 {
        Affine2 { linear: Mat2::new_identity(2), translation: Zero::zero() }
    }

    /// Returns the inverse of an affine transformation, or
    /// None if its linear part is singular.
    pub fn inverse(&self) -> Option<Affine2> {
        if let Some(inv) = self.linear.inv() {
            Some(Affine2 {
                linear: inv,
                translation: Vec2::new(0f32, 0f32) - inv * self.translation })
        }
        else {
            None
        }
    }

    /// Applies the affine transformation to a given vector.
    pub fn apply(&self, x: Vec2<f32>) -> Vec2<f32> {
        self.linear * x + self.translation
    }
}

#[cfg(test)]
mod test {

    use super::{
        Affine2
    };

    use nalgebra::{
        Mat2,
        Vec2
    };

    #[test]
    fn test_affine2_apply() {
        let aff = Affine2::new(
            Mat2::new(1f32, 2f32, 3f32, 4f32),
            Vec2::new(0f32, 1f32));
        let x = Vec2::new(1f32, 2f32);
        assert_eq!(aff.apply(x), Vec2::new(5f32, 12f32));
    }

    #[test]
    fn test_affine2_inverse_singular() {
        let aff = Affine2::new(
            Mat2::new(0f32, 1f32, 0f32, 1f32),
            Vec2::new(0f32, 1f32));
        assert_eq!(aff.inverse(), None);
    }

    #[test]
    fn test_affine2_inverse_nonsingular() {
        let mat = Mat2::new(1f32, 2f32, 3f32, 4f32);
        let aff = Affine2::new(mat, Vec2::new(7f32, -4f32));

        if let Some(inv) = aff.inverse() {
            let test = Vec2::new(4.2f32, 8.7f32);
            let actual = inv.apply(aff.apply(test));
            assert!((test[0] - actual[0]).abs() <= 1e-4,
                format!("left: {:?}, right: {:?}", test[0], actual[0]));
            assert!((test[1] - actual[1]).abs() <= 1e-4,
                format!("left: {:?}, right: {:?}", test[1], actual[1]));
        }
        else {
            assert!(false, "No inverse returned");
        }
    }
}
