//! Assorted mathematical helper functions.

/// L1 norm of a vector.
pub fn l1_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x.abs())
}

/// L2 norm of a vector.
pub fn l2_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x * x).sqrt()
}
