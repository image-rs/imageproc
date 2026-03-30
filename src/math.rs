//! Assorted mathematical helper functions.

/// L1 norm of a vector.
pub fn l1_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x.abs())
}

/// L2 norm of a vector.
pub fn l2_norm(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |acc, x| acc + x * x).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_norm() {
        assert_eq!(l1_norm(&[]), 0.0);
        assert_eq!(l1_norm(&[1.0, -2.0, 3.5]), 6.5);
    }

    #[test]
    fn test_l2_norm() {
        assert_eq!(l2_norm(&[]), 0.0);
        assert_approx_eq!(l2_norm(&[3.0, 4.0]), 5.0);
        assert_approx_eq!(l2_norm(&[1.0, -2.0, 2.0]), 3.0);
    }
}
