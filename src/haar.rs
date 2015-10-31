//! Functions for creating and evaluating [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features).

use image::{
    GenericImage,
    Luma
};
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Mul;

/// Whether the top left region in a Haar filter is counted
/// with positive or negative sign.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sign { Positive, Negative }

/// A Haar filter whose value on an integral image is the weighted sum
/// of the values of the integral image at the given points.
// TODO: these structs are pretty big. Look into instead just storing
// TODO: the offsets between sample points. We should only need 10 bytes/filter,
// TODO: meaning we could fit a typical cascade in L1 cache.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HaarFilter {
    points: [u32; 18],
    weights: [i8; 9],
    count: usize
}

impl HaarFilter {

    /// Evaluates the Haar filter on an integral image.
    pub fn evaluate<I>(&self, integral: &I ) -> i32
        where I: GenericImage<Pixel=Luma<u32>> {

        let mut sum = 0i32;
        for i in 0..self.count {
            let p = integral.get_pixel(self.points[2 * i], self.points[2 * i + 1])[0];
            sum += p as i32 * self.weights[i] as i32;
        }
        sum
    }

    /// Returns the following feature (with signs reversed if Sign == Sign::Negative).
    /// <pre>
    ///     A   B   C
    ///       +   -
    ///     D   E   F
    /// </pre>
    /// A = (top, left), B.x = left + dx1, C.x = B.x + dx2, and D.y = A.y + dy.
    pub fn two_region_horizontal(top: u32, left: u32, dx1: u32, dx2: u32, dy: u32, sign: Sign)
        -> HaarFilter {

        combine_alternating(&[
            eval_points(top, left,       dx1, dy),
            eval_points(top, left + dx1, dx2, dy)]) * multiplier(sign)
    }

    /// Returns the following feature (with signs reversed if Sign == Sign::Negative).
    /// <pre>
    ///     A   B
    ///       +
    ///     C   D
    ///       -
    ///     E   F
    /// </pre>
    /// A = (top, left), B.x = left + dx, C.y = top + dy1, and E.y = C.y + dy2.
    pub fn two_region_vertical(top: u32, left: u32, dx: u32, dy1: u32, dy2: u32, sign: Sign)
        -> HaarFilter {

        combine_alternating(&[
            eval_points(top,       left, dx, dy1),
            eval_points(top + dy1, left, dx, dy2)]) * multiplier(sign)
    }

    /// Returns the following feature (with signs reversed if Sign == Sign::Negative).
    /// <pre>
    ///     A   B   C   D
    ///       +   -   +
    ///     E   F   G   H
    /// </pre>
    /// A = (top, left), B.x = left + dx1, C.x = B.x + dx2, D.x = C.x + dx3, and E.y = top + dy.
    pub fn three_region_horizontal(
        top: u32, left: u32, dx1: u32, dx2: u32, dx3: u32, dy: u32, sign: Sign)
            -> HaarFilter {

        combine_alternating(&[
            eval_points(top, left,             dx1, dy),
            eval_points(top, left + dx1,       dx2, dy),
            eval_points(top, left + dx1 + dx2, dx3, dy),
            ]) * multiplier(sign)
    }

    /// Returns the following feature (with signs reversed if Sign == Sign::Negative).
    /// <pre>
    ///     A   B
    ///       +
    ///     C   D
    ///       -
    ///     E   F
    ///       +
    ///     G   H
    /// </pre>
    /// A = (top, left), B.x = left + dx, C.y = top + dy1, E.y = C.y + dy2, and G.y = E.y + dy3.
    pub fn three_region_vertical(
        top: u32, left: u32, dx: u32, dy1: u32, dy2: u32, dy3: u32, sign: Sign)
            -> HaarFilter {

        combine_alternating(&[
            eval_points(top,             left, dx, dy1),
            eval_points(top + dy1,       left, dx, dy2),
            eval_points(top + dy1 + dy2, left, dx, dy3),
            ]) * multiplier(sign)
    }

    /// Returns the following feature (with signs reversed if Sign == Sign::Negative).
    /// <pre>
    ///     A   B   C
    ///       +   -
    ///     D   E   F
    ///       -   +
    ///     G   H   I
    /// </pre>
    /// A = (top, left), B.x = left + dx1, C.x = B.x + dx2, D.y = top + dy1, and G.y = D.y + dy2.
    pub fn four_region(
        top: u32, left: u32, dx1: u32, dx2: u32, dy1: u32, dy2: u32, sign: Sign)
            -> HaarFilter {

        combine_alternating(&[
            eval_points(top,       left,       dx1, dy1),
            eval_points(top,       left + dx1, dx2, dy1),
            eval_points(top + dy1, left + dx1, dx2, dy2),
            eval_points(top + dy1, left,       dx1, dy2),
            ]) * multiplier(sign)
    }
}

/// See comment on eval_points.
struct EvalPoints {
    points: [(u32, u32); 4],
    weights: [i8; 4]
}

impl EvalPoints {
    fn new(points: [(u32, u32); 4], weights: [i8; 4]) -> EvalPoints {
        EvalPoints { points: points, weights: weights }
    }
}

impl Mul<i8> for HaarFilter {
    type Output = HaarFilter;

    fn mul(self, rhs: i8) -> HaarFilter {
        let mut copy = self;
        for i in 0..copy.weights.len() {
            copy.weights[i] *= rhs;
        }
        copy
    }
}

/// Points at which to evaluate an integral image to produce the sum of the
/// pixel intensities of all points within a rectangle. Only valid when the
/// rectangle is wholly contained in the image boundaries. The only non-trivial
/// cases are when the rectangle touches the left or top image boundaries. In this
/// case we need to evaluate fewer than four points, and return weights of zero
/// for the other points.
fn eval_points(top: u32, left: u32, width: u32, height: u32) -> EvalPoints {

    let mut points = [
        (0u32, 0u32),
        (0u32, 0u32),
        (0u32, 0u32),
        (left + width - 1, top + height - 1)];

    let mut weights = [0i8, 0i8, 0i8, 1i8];

    if top > 0 && left > 0 {
        points[0] = (left - 1, top - 1);
        weights[0] = 1i8;
    }
    if top > 0 {
        points[1] = (left + width - 1, top - 1);
        weights[1] = -1i8;
    }
    if left > 0 {
        points[2] = (left - 1, top + height - 1);
        weights[2] = -1i8;
    }

    EvalPoints::new(points, weights)
}

/// Combine sets of evaluation points with alternating signs.
/// The first entry of rects is counted with positive sign.
// TODO: check that we don't have too many distinct points. This
// TODO: function isn't exported, so we just need to check the HaarFilter uses
// TODO: haven't messed anything up.
fn combine_alternating(rects: &[EvalPoints]) -> HaarFilter {

    // Aggregate weights of all points, remove any with zero weight, and
    // order lexicographically by location.
    let mut sign = 1i8;
    let sorted_points = rects
        .iter()
        .fold(HashMap::new(), |mut acc, rect| {
            for i in 0..4 {
                *acc.entry(rect.points[i]).or_insert(0) += sign * rect.weights[i];
            }
            sign *= -1i8;
            acc
            })
        .into_iter()
        .filter(|kv| kv.1 != 0)
        .sorted_by(|a, b| Ord::cmp(&((a.0).1, (a.0).0), &((b.0).1, (b.0).0)));

    let mut count = 0;
    let mut points = [0u32; 18];
    let mut weights = [0i8; 9];

    for pw in sorted_points {
        points[2 * count] = (pw.0).0;
        points[2 * count + 1] = (pw.0).1;
        weights[count] = pw.1;
        count += 1;
    }

    HaarFilter {
        points: points,
        weights: weights,
        count: count }
}

fn multiplier(sign: Sign) -> i8 {
    if sign == Sign::Positive {1} else {-1}
}

#[cfg(test)]
mod test {

    use super::{
        combine_alternating,
        EvalPoints,
        HaarFilter,
        Sign
    };
    use image::{
        ImageBuffer
    };
    use integralimage::{
        integral_image
    };

    #[test]
    fn test_combine_alternating() {
        let a = (0, 0);
        let b = (1, 0);
        let c = (2, 0);
        let d = (3, 0);
        let e = (0, 1);
        let f = (1, 1);
        let g = (2, 1);
        let h = (3, 1);

        let left  = EvalPoints::new([a, b, e, f], [1, -1, -1, 1]);
        let mid   = EvalPoints::new([b, c, f, g], [1, -1, -1, 1]);
        let right = EvalPoints::new([c, d, g, h], [1, -1, -1, 1]);

        let filter = combine_alternating(&[left, mid, right]);
        let expected = HaarFilter {
            points: [0, 0, 1, 0, 2, 0, 3, 0, 0, 1, 1, 1, 2, 1, 3, 1, 0, 0],
            weights: [1, -2, 2, -1, -1, 2, -2, 1, 0],
            count: 8
        };

        assert_eq!(filter, expected);
    }

    #[test]
    fn test_two_region_horizontal() {
        // Two region horizontally aligned filter:
        // A   B   C
        //   +   -
        // D   E   F
        let image = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8, 3u8,     4u8,     5u8,
                 /***+++++++++*****-----***/
            6u8, /**/7u8, 8u8,/**/ 9u8, /**/0u8,
            9u8, /**/8u8, 7u8,/**/ 6u8, /**/5u8,
            4u8, /**/3u8, 2u8,/**/ 1u8, /**/0u8,
                 /***+++++++++*****-----***/
            6u8, 5u8, 4u8, 2u8, 1u8]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter::two_region_horizontal(1, 1, 2, 1, 3, Sign::Positive);
        assert_eq!(filter.evaluate(&integral), 19i32);
    }

    #[test]
    fn test_three_region_vertical() {
        // Three region vertically aligned filter:
        // A   B
        //   +
        // C   D
        //   -
        // E   F
        //   +
        // G   H
        let image = ImageBuffer::from_raw(5, 5, vec![
        /*****************/
        /*-*/1u8, 2u8,/*-*/ 3u8, 4u8, 5u8,
        /*****************/
        /*+*/6u8, 7u8,/*+*/ 8u8, 9u8, 0u8,
        /*+*/9u8, 8u8,/*+*/ 7u8, 6u8, 5u8,
        /*****************/
        /*-*/4u8, 3u8,/*-*/ 2u8, 1u8, 0u8,
        /*****************/
             6u8, 5u8,      4u8, 2u8, 1u8]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter::three_region_vertical(0, 0, 2, 1, 2, 1, Sign::Negative);
        assert_eq!(filter.evaluate(&integral), 20i32);
    }

    #[test]
    fn test_four_region() {
        // Four region filter:
        // A   B   C
        //   +   -
        // D   E   F
        //   -   +
        // G   H   I
        let image = ImageBuffer::from_raw(5, 5, vec![
        1u8,    2u8, 3u8,     4u8,     5u8,
            /************************/
        6u8,/**/7u8, 8u8,/**/ 9u8,/**/ 0u8,
            /************************/
        9u8,/**/8u8, 7u8,/**/ 6u8,/**/ 5u8,
        4u8,/**/3u8, 2u8,/**/ 1u8,/**/ 0u8,
            /************************/
        6u8,    5u8, 4u8,     2u8,     1u8]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter::four_region(1, 1, 2, 1, 1, 2, Sign::Positive);

        assert_eq!(filter.evaluate(&integral), -7i32);
    }
}
