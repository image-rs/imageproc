//! Basic operations on 2d points.

use num::{Num, NumCast};
use std::ops::{Add, Sub};

/// A 2D point.
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

impl<T: NumCast> Point<T> {
    /// Converts to a Point<f64>. Panics if the cast fails.
    pub(crate) fn to_f64(&self) -> Point<f64> {
        Point::new(self.x.to_f64().unwrap(), self.y.to_f64().unwrap())
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
        Rotation { sin_theta, cos_theta }
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

impl<T: Num> Add for Point<T> {
    type Output = Self;

    fn add(self, other: Point<T>) -> Point<T> {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: Num> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Point<T>) -> Point<T> {
        Point::new(self.x - other.x, self.y - other.y)
    }
}
