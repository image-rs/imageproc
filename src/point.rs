//! A 2d point type.

use num::{Num, NumCast};
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// A 2d point.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point<T> {
    /// x-coordinate.
    pub x: T,
    /// y-coordinate.
    pub y: T,
}

impl<T> Point<T> {
    /// Construct a point at (x, y).
    pub fn new(x: T, y: T) -> Point<T> {
        Point::<T> { x, y }
    }
}

impl<T: Num> Add for Point<T> {
    type Output = Self;

    fn add(self, other: Point<T>) -> Point<T> {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: Num + Copy> AddAssign for Point<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}

impl<T: Num> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Point<T>) -> Point<T> {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl<T: Num + Copy> SubAssign for Point<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
    }
}

impl<T: NumCast> Point<T> {
    /// Converts to a Point<f64>. Panics if the cast fails.
    pub(crate) fn to_f64(&self) -> Point<f64> {
        Point::new(self.x.to_f64().unwrap(), self.y.to_f64().unwrap())
    }

    /// Converts to a Point<i32>. Panics if the cast fails.
    pub(crate) fn to_i32(&self) -> Point<i32> {
        Point::new(self.x.to_i32().unwrap(), self.y.to_i32().unwrap())
    }
}

/// Returns the Euclidean distance between two points.
pub(crate) fn distance<T: NumCast>(p: Point<T>, q: Point<T>) -> f64 {
    distance_sq(p, q).sqrt()
}

/// Returns the square of the Euclidean distance between two points.
pub(crate) fn distance_sq<T: NumCast>(p: Point<T>, q: Point<T>) -> f64 {
    let p = p.to_f64();
    let q = q.to_f64();
    (p.x - q.x).powf(2.0) + (p.y - q.y).powf(2.0)
}

/// A fixed rotation. This struct exists solely to cache the values of `sin(theta)` and `cos(theta)` when
/// applying a fixed rotation to multiple points.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Rotation {
    sin_theta: f64,
    cos_theta: f64,
}

impl Rotation {
    /// A rotation of `theta` radians.
    pub(crate) fn new(theta: f64) -> Rotation {
        let (sin_theta, cos_theta) = theta.sin_cos();
        Rotation {
            sin_theta,
            cos_theta,
        }
    }
}

impl Point<f64> {
    /// Rotates a point.
    pub(crate) fn rotate(&self, rotation: Rotation) -> Point<f64> {
        let x = self.x * rotation.cos_theta + self.y * rotation.sin_theta;
        let y = self.y * rotation.cos_theta - self.x * rotation.sin_theta;
        Point::new(x, y)
    }

    /// Inverts a rotation.
    pub(crate) fn invert_rotation(&self, rotation: Rotation) -> Point<f64> {
        let x = self.x * rotation.cos_theta - self.y * rotation.sin_theta;
        let y = self.y * rotation.cos_theta + self.x * rotation.sin_theta;
        Point::new(x, y)
    }
}

/// A line of the form Ax + By + C = 0.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Line {
    a: f64,
    b: f64,
    c: f64,
}

impl Line {
    /// Returns the `Line` that passes through p and q.
    pub fn from_points(p: Point<f64>, q: Point<f64>) -> Line {
        let a = p.y - q.y;
        let b = q.x - p.x;
        let c = p.x * q.y - q.x * p.y;
        Line { a, b, c }
    }

    /// Computes the shortest distance from this line to the given point.
    pub fn distance_from_point(&self, point: Point<f64>) -> f64 {
        let Line { a, b, c } = self;
        (a * point.x + b * point.y + c).abs() / (a.powf(2.0) + b.powf(2.)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_from_points() {
        let p = Point::new(5.0, 7.0);
        let q = Point::new(10.0, 3.0);
        assert_eq!(
            Line::from_points(p, q),
            Line {
                a: 4.0,
                b: 5.0,
                c: -55.0
            }
        );
    }

    #[test]
    fn distance_between_line_and_point() {
        assert_approx_eq!(
            Line {
                a: 8.0,
                b: 7.0,
                c: 5.0
            }
            .distance_from_point(Point::new(2.0, 3.0)),
            3.9510276472,
            1e-10
        );
    }
}
