//! Functions for creating and evaluating [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features).

use definitions::{HasBlack,HasWhite,VecBuffer};
use image::{GenericImage,ImageBuffer,Luma};
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Mul;

/// Whether the top left region in a Haar filter is counted
/// with positive or negative sign.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Sign {
    /// Top left region is counted with a positive sign.
    Positive,
    /// Top left region is counted with a negative sign.
    Negative
}

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

/// Returns a vector of all valid Haar filters for an image with given width and height.
pub fn enumerate_haar_filters(width: u32, height: u32) -> Vec<HaarFilter> {
    let signs = [Sign::Positive, Sign::Negative];
    let mut features = Vec::with_capacity(number_of_haar_filters(width, height) as usize);

    for y0 in 0..height {
        for x0 in 0..width {
            for h0 in 1..(height - y0) + 1 {
                for w0 in 1..(width - x0) + 1 {
                    for w1 in 1..(width - x0 - w0) + 1 {
                        for sign in &signs {
                            features.push(
                                HaarFilter::two_region_horizontal(y0, x0, w0, w1, h0, *sign));
                        }

                        for w2 in 1..(width - x0 - w0 - w1) + 1 {
                            for sign in &signs {
                                features.push(
                                    HaarFilter::three_region_horizontal(y0, x0, w0, w1, w2, h0, *sign));
                            }
                        }

                        for h1 in 1..(height - y0 - h0) + 1 {
                            for sign in &signs {
                                features.push(
                                    HaarFilter::four_region(y0, x0, w0, w1, h0, h1, *sign));
                            }
                        }
                    }

                    for h1 in 1..(height - y0 - h0) + 1 {
                        for sign in &signs {
                            features.push(
                                HaarFilter::two_region_vertical(y0, x0, w0, h0, h1, *sign));
                        }

                        for h2 in 1..(height - y0 - h0 - h1) + 1 {
                            for sign in &signs {
                                features.push(
                                    HaarFilter::three_region_vertical(y0, x0, w0, h0, h1, h2, *sign));
                            }
                        }
                    }
                }
            }
        }
    }

    features
}

/// Returns the number of distinct Haar filters for an image of the given dimensions.
/// Includes positive and negative, two and three region, vertical and horizontal filters,
/// as well as positive and negative four region filters.
///
/// Consider two-region positive horizontal Haar filters in an image of height 1.
/// There is exactly one such filter for each choice of L, M and R below.
///
/// <pre>
///     L   M     R
/// | | | | | | | | |
///  . . + + - - - .
/// </pre>
///
/// There are (width + 1) dividing lines between pixels, so ((width + 1) choose 3) such filters.
/// For an image of arbitrary height there are ((height + 1) choose 2) choices for the top and
/// bottom of each two region positive Haar filter, and every positive filter has a corresponding
/// negative filter. Thus there are ((width + 1) choose 3) * ((height + 1) choose 2) * 2 filters
/// of this type in an image with dimensions (width, height).
///
/// A very similar argument applies to the other filter types.
pub fn number_of_haar_filters(width: u32, height: u32) -> u32 {
    // Two-region horizontal
    n_choose_k(width + 1, 3) * n_choose_k(height + 1, 2) * 2 +
    // Two-region vertical
    n_choose_k(height + 1, 3) * n_choose_k(width + 1, 2) * 2 +
    // Three-region horizontal
    n_choose_k(width + 1, 4) * n_choose_k(height + 1, 2) * 2 +
    // Three-region vertical
    n_choose_k(height + 1, 4) * n_choose_k(width + 1, 2) * 2 +
    // Four-region
    n_choose_k(width + 1, 3) * n_choose_k(height + 1, 3) * 2
}

fn n_choose_k(n: u32, k: u32) -> u32 {
    if k > n {
        return 0;
    }
    let k = if k * 2 > n { n - k } else { k };
    if k == 0 {
        return 1;
    }
    let mut r = n;
    for i in 2..(k + 1) {
        r *= n - i + 1;
        r /= i;
    }
    r
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

/// See comment on `eval_points`.
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
/// rectangle is wholly contained in the image boundaries.
fn eval_points(top: u32, left: u32, width: u32, height: u32) -> EvalPoints {
    let right = left + width - 1;
    let bottom = top + height - 1;

    EvalPoints::new(
        [(left, top), (left, bottom + 1), (right + 1, top), (right + 1, bottom + 1)],
        [1i8, -1i8, -1i8, 1i8]
    )
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

/// Draws the given Haar filter on an image, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_filter<I>(image: &I, filter: HaarFilter) -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite + 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_haar_filter_mut(&mut out, filter);
    out
}

/// Draws the given Haar filter on an image in place, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_filter_mut<I>(image: &mut I, filter: HaarFilter)
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite
{
    let (width, height) = image.dimensions();
    for y in 0..height {
        for x in 0..width {
            let mut weight = 0;
            for i in 0..filter.count {
                if y < filter.points[2 * i + 1] && x < filter.points[2 * i] {
                    weight += filter.weights[i];
                }
            }
            assert!(weight == 0 || weight == 1 || weight == -1);
            unsafe {
                if weight > 0 {
                    image.unsafe_put_pixel(x, y, I::Pixel::white());
                }
                if weight < 0 {
                    image.unsafe_put_pixel(x, y, I::Pixel::black());
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use super::{
        combine_alternating,
        draw_haar_filter,
        enumerate_haar_filters,
        EvalPoints,
        HaarFilter,
        Sign,
        number_of_haar_filters
    };
    use image::{
        GrayImage,
        ImageBuffer
    };
    use integralimage::{
        integral_image
    };
    use utils::gray_bench_image;
    use test;

    #[test]
    fn test_number_of_haar_filters() {
        for h in 0..6 {
            for w in 0..6 {
                let filters = enumerate_haar_filters(w, h);
                let actual = filters.len() as u32;
                let expected = number_of_haar_filters(w, h);
                assert_eq!(actual, expected);
            }
        }
    }

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
            6u8,     5u8, 4u8,     2u8,     1u8]).unwrap();

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

    #[test]
    fn test_enumerate() {
        assert_eq!(enumerate_haar_filters(1, 1).len(), 0);
        assert_eq!(enumerate_haar_filters(1, 2).len(), 2);
        assert_eq!(enumerate_haar_filters(2, 1).len(), 2);
        assert_eq!(enumerate_haar_filters(3, 1).len(), 10);
        assert_eq!(enumerate_haar_filters(1, 3).len(), 10);
        assert_eq!(enumerate_haar_filters(2, 2).len(), 14);
    }

    #[test]
    fn test_draw_haar_filter_two_region_horizontal() {
        // Two region horizontally aligned filter:
        // A   B   C
        //   +   -
        // D   E   F
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8, 3u8,     4u8,     5u8,
                 /***+++++++++*****-----***/
            6u8, /**/7u8, 8u8,/**/ 9u8, /**/0u8,
            9u8, /**/8u8, 7u8,/**/ 6u8, /**/5u8,
            4u8, /**/3u8, 2u8,/**/ 1u8, /**/0u8,
                 /***+++++++++*****-----***/
            6u8,     5u8, 4u8,     2u8,     1u8]).unwrap();

        let filter = HaarFilter::two_region_horizontal(1, 1, 2, 1, 3, Sign::Positive);
        let actual = draw_haar_filter(&image, filter);

        let expected = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8,  3u8,        4u8,     5u8,
                 /***+++++++++++++*****-----***/
            6u8, /**/255u8, 255u8,/**/ 0u8, /**/0u8,
            9u8, /**/255u8, 255u8,/**/ 0u8, /**/5u8,
            4u8, /**/255u8, 255u8,/**/ 0u8, /**/0u8,
                 /***+++++++++++++*****-----***/
            6u8,     5u8,   4u8,       2u8,     1u8]).unwrap();

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_draw_haar_filter_four_region() {
        // Four region filter:
        // A   B   C
        //   +   -
        // D   E   F
        //   -   +
        // G   H   I
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
        1u8,    2u8, 3u8,     4u8,     5u8,
            /************************/
        6u8,/**/7u8, 8u8,/**/ 9u8,/**/ 0u8,
            /************************/
        9u8,/**/8u8, 7u8,/**/ 6u8,/**/ 5u8,
        4u8,/**/3u8, 2u8,/**/ 1u8,/**/ 0u8,
            /************************/
        6u8,    5u8, 4u8,     2u8,     1u8]).unwrap();

        let filter = HaarFilter::four_region(1, 1, 2, 1, 1, 2, Sign::Positive);
        let actual = draw_haar_filter(&image, filter);

        let expected = ImageBuffer::from_raw(5, 5, vec![
        1u8,    2u8,   3u8,       4u8,       5u8,
            /******************************/
        6u8,/**/255u8, 255u8,/**/ 0u8,  /**/ 0u8,
            /******************************/
        9u8,/**/0u8,   0u8,  /**/ 255u8,/**/ 5u8,
        4u8,/**/0u8,   0u8,  /**/ 255u8,/**/ 0u8,
            /******************************/
        6u8,    5u8,   4u8,       2u8,     1u8]).unwrap();

        assert_pixels_eq!(actual, expected);
    }

    #[bench]
    fn bench_evaluate_all_filters_10x10(b: &mut test::Bencher) {
        // 163350 filters in total
        let filters = enumerate_haar_filters(10, 10);
        let image = gray_bench_image(10, 10);
        let integral = integral_image(&image);

        b.iter(|| {
            for filter in &filters {
                let x = filter.evaluate(&integral);
                test::black_box(x);
            }
        });
    }
}
