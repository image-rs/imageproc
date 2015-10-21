//! Functions for creating and evaluating [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features).

use image::{
    GenericImage,
    Luma
};

/// A Haar filter whose value on an integral image is the weighted sum
/// of the values of the integral image at the given points.
// TODO: these structs are pretty big. Look into instead just storing
// TODO: the offsets between sample points.
pub struct HaarFilter {
    points: [u32; 18],
    weights: [i8; 9],
    count: usize
}

/// Whether the top left region in a Haar filter is counted
/// with positive or negative sign.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sign {
    Positive,
    Negative
}

impl HaarFilter {

    /// If Sign is Positive then returns the following Haar feature.
    ///
    ///     A   B   C
    ///       +   -
    ///     D   E   F
    ///
    /// If Sign is Negative then the + and - signs are reversed.
    /// The distance between A and B is dx1, between B and C is dx2,
    /// and between A and D is dy.
    ///
    /// Given an integral image I, the value of this feature is
    /// I(A) - 2I(B) + I(C) - I(D) + 2I(E) - I(F) multiplied by sign.
    pub fn two_region_horizontal(
        top_left: (u32, u32), dx1: u32, dx2: u32, dy: u32, sign: Sign)
            -> HaarFilter {

        let left   = top_left.0;
        let x_mid  = left + dx1;
        let right  = x_mid + dx2;
        let top    = top_left.1;
        let bottom = top + dy;
        let mul    = if sign == Sign::Positive {1i8} else {-1i8};

        HaarFilter {
            points: [
                left, top,     // A
                x_mid, top,    // B
                right, top,    // C
                left, bottom,  // D
                x_mid, bottom, // E
                right, bottom, // F
                0, 0, 0, 0, 0, 0],
            weights: [
                mul,           // A
                -2i8 * mul,    // B
                mul,           // C
                -mul,          // D
                2i8 * mul,     // E
                -mul,          // F
                0, 0, 0],
            count: 6 }
    }

    /// If Sign is Positive then returns the following Haar feature.
    ///
    ///     A   B
    ///       +
    ///     C   D
    ///       -
    ///     E   F
    ///
    /// If Sign is Negative then the + and - signs are reversed.
    /// The distance between A and B is dx, between A and C is dy1,
    /// and between C and E is dy2.
    ///
    /// Given an integral image I, the value of this feature is
    /// I(A) - I(B) - 2I(C) + 2I(D) + I(E) - I(F) multipled by sign.
    pub fn two_region_vertical(
        top_left: (u32, u32), dx: u32, dy1: u32, dy2: u32, sign: Sign)
            -> HaarFilter {

        let left   = top_left.0;
        let right  = left + dx;
        let top    = top_left.1;
        let y_mid  = top + dy1;
        let bottom = y_mid + dy2;
        let mul    = if sign == Sign::Positive {1i8} else {-1i8};

        HaarFilter {
            points: [
                left, top,      // A
                right, top,     // B
                left, y_mid,    // C
                right, y_mid,   // D
                left, bottom,   // E
                right, bottom,  // F
                0, 0, 0, 0, 0, 0],
            weights: [
                mul,             // A
                -mul,            // B
                -2i8 * mul,      // C
                2i8 * mul,       // D
                mul,             // E
                -mul,            // F
                0, 0, 0],
            count: 6 }
    }

    /// If Sign is Positive then returns the following Haar feature.
    ///
    ///     A   B   C   D
    ///       +   -   +
    ///     E   F   G   H
    ///
    /// If Sign is Negative then the + and - signs are reversed.
    /// The distance between A and B is dx1, between B and C is dx2, between
    /// C and D is dx3, and between A and E is dy.
    ///
    /// Given an integral image I, the value of this feature is
    /// I(A) - 2I(B) + 2I(C) - I(D) - I(E) + 2I(F) - 2I(G) + I(H) multipled by sign.
    pub fn three_region_horizontal(
        top_left: (u32, u32), dx1: u32, dx2: u32, dx3: u32, dy: u32, sign: Sign)
            -> HaarFilter {

        let left        = top_left.0;
        let x_left_mid  = left + dx1;
        let x_right_mid = x_left_mid + dx2;
        let right       = x_right_mid + dx3;
        let top         = top_left.1;
        let bottom      = top + dy;
        let mul         = if sign == Sign::Positive {1i8} else {-1i8};

        HaarFilter {
            points: [
                left, top,           // A
                x_left_mid, top,     // B
                x_right_mid, top,    // C
                right, top,          // D
                left, bottom,        // E
                x_left_mid, bottom,  // F
                x_right_mid, bottom, // G
                right, bottom,       // H
                0, 0],

            weights: [
                mul,         // A
                -2i8 * mul,  // B
                2i8 * mul,   // C
                -mul,        // D
                -mul,        // E
                2i8 * mul,   // F
                -2i8 * mul,  // G
                mul,         // H
                0],
                count: 6 }
    }

    /// If Sign is Positive then returns the following Haar feature.
    ///
    ///     A   B
    ///       +
    ///     C   D
    ///       -
    ///     E   F
    ///       +
    ///     G   H
    ///
    /// If Sign is Negative then the + and - signs are reversed.
    /// The distance between A and B is dx, between A and C is dy1, between
    /// C and E is dy2, and between E and G is dy3.
    ///
    /// Given an integral image I, the value of this feature is
    /// I(A) - I(B) - 2I(C) + 2I(D) + 2I(E) - 2I(F) - I(G) + I(H) multiplied by sign.
    pub fn three_region_vertical(
        top_left: (u32, u32), dx: u32, dy1: u32, dy2: u32, dy3: u32, sign: Sign)
            -> HaarFilter {
        HaarFilter { points: [0u32; 18], weights: [0i32; 9], count: 0 }
    }

    /// If Sign is Positive then returns the following Haar feature.
    ///
    ///     A   B   C
    ///       +   -
    ///     D   E   F
    ///       -   +
    ///     G   H   I
    ///
    /// If Sign is Negative then the + and - signs are reversed.
    /// The distance between A and B is dx1, between B and C is dx1, between
    /// A and D is dy1, and between D and G is dy2.
    ///
    /// Given an integral image I, the value of this feature is
    /// I(A) - 2I(B) - 2I(D) + 4I(E) + I(C) - 2I(F) + I(G) - 2I(H) + I(I) multiplied by sign.
    pub fn four_region(
        top_left: (u32, u32), dx1: u32, dx2: u32, dy1: u32, dy2: u32, sign: Sign)
            -> HaarFilter {
        HaarFilter { points: [0u32; 18], weights: [0i32; 9], count: 0 }
    }

    /// Evaluates the Haar filter on an integral image.
    pub fn evaluate<I>(&self, integral: &I ) -> i32
        where I: GenericImage<Pixel=Luma<u32>> {

        let mut sum = 0i32;
        for i in 0..self.count {
            let p = integral.get_pixel(self.points[2 * i], self.points[2 * i + 1])[0];
            sum += p as i32 * self.weights[i];
        }
        sum
    }
}

#[cfg(test)]
mod test {

    use super::{
        HaarFilter
    };

}
