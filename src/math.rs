//! Assorted mathematical helper functions.

use conv::ValueInto;

/// L2 norm of a vector.
pub fn l2_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x * x).sqrt()
}

/// Helper for a conversion that we know can't fail.
pub fn cast<T, U>(x: T) -> U
where
    T: ValueInto<U>,
{
    match x.value_into() {
        Ok(y) => y,
        Err(_) => panic!("Failed to convert"),
    }
}
