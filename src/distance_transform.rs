//! Functions for computing distance transforms - the distance of each pixel in an
//! image from the nearest pixel of interest.

use crate::definitions::Image;
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use std::cmp::min;
use std::{f64, u8};

/// How to measure distance between coordinates.
/// See the [`distance_transform`](fn.distance_transform.html) documentation for examples.
///
/// Note that this enum doesn't currently include the `L2` norm. As `Norm`
/// is used by the [`morphology`](../morphology/index.html) functions, this means that we
/// don't support using the `L2` norm for any of those functions.
///
/// This module does support calculating the `L2` distance function, via the
/// [`euclidean_squared_distance_transform`](fn.euclidean_squared_distance_transform.html)
/// function, but the signature of this function is not currently compatible with those for
/// computing `L1` and `LInf` distance transforms. It would be nice to unify these functions
/// in future.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Defines d((x1, y1), (x2, y2)) to be abs(x1 - x2) + abs(y1 - y2).
    /// Also known as the Manhattan or city block norm.
    L1,
    /// Defines d((x1, y1), (x2, y2)) to be max(abs(x1 - x2), abs(y1 - y2)).
    /// Also known as the chessboard norm.
    LInf,
}

/// Returns an image showing the distance of each pixel from a foreground pixel in the original image.
///
/// A pixel belongs to the foreground if it has non-zero intensity. As the image
/// has a bit-depth of 8, distances saturate at 255.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::distance_transform::{distance_transform, Norm};
///
/// let image = gray_image!(
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0,   1,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0
/// );
///
/// // L1 norm
/// let l1_distances = gray_image!(
///     4,   3,   2,   3,   4;
///     3,   2,   1,   2,   3;
///     2,   1,   0,   1,   2;
///     3,   2,   1,   2,   3;
///     4,   3,   2,   3,   4
/// );
///
/// assert_pixels_eq!(distance_transform(&image, Norm::L1), l1_distances);
///
/// // LInf norm
/// let linf_distances = gray_image!(
///     2,   2,   2,   2,   2;
///     2,   1,   1,   1,   2;
///     2,   1,   0,   1,   2;
///     2,   1,   1,   1,   2;
///     2,   2,   2,   2,   2
/// );
///
/// assert_pixels_eq!(distance_transform(&image, Norm::LInf), linf_distances);
/// # }
/// ```
pub fn distance_transform(image: &GrayImage, norm: Norm) -> GrayImage {
    let mut out = image.clone();
    distance_transform_mut(&mut out, norm);
    out
}

/// Updates an image in place so that each pixel contains its distance from a foreground pixel in the original image.
///
/// A pixel belongs to the foreground if it has non-zero intensity. As the image has a bit-depth of 8,
/// distances saturate at 255.
///
/// See the [`distance_transform`](fn.distance_transform.html) documentation for examples.
pub fn distance_transform_mut(image: &mut GrayImage, norm: Norm) {
    distance_transform_impl(image, norm, DistanceFrom::Foreground);
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub(crate) enum DistanceFrom {
    Foreground,
    Background,
}

pub(crate) fn distance_transform_impl(image: &mut GrayImage, norm: Norm, from: DistanceFrom) {
    let max_distance = Luma([min(image.width() + image.height(), 255u32) as u8]);

    unsafe {
        // Top-left to bottom-right
        for y in 0..image.height() {
            for x in 0..image.width() {
                if from == DistanceFrom::Foreground {
                    if image.unsafe_get_pixel(x, y)[0] > 0u8 {
                        image.unsafe_put_pixel(x, y, Luma([0u8]));
                        continue;
                    }
                } else if image.unsafe_get_pixel(x, y)[0] == 0u8 {
                    image.unsafe_put_pixel(x, y, Luma([0u8]));
                    continue;
                }

                image.unsafe_put_pixel(x, y, max_distance);

                if x > 0 {
                    check(image, x, y, x - 1, y);
                }

                if y > 0 {
                    check(image, x, y, x, y - 1);

                    if norm == Norm::LInf {
                        if x > 0 {
                            check(image, x, y, x - 1, y - 1);
                        }
                        if x < image.width() - 1 {
                            check(image, x, y, x + 1, y - 1);
                        }
                    }
                }
            }
        }

        // Bottom-right to top-left
        for y in (0..image.height()).rev() {
            for x in (0..image.width()).rev() {
                if x < image.width() - 1 {
                    check(image, x, y, x + 1, y);
                }

                if y < image.height() - 1 {
                    check(image, x, y, x, y + 1);

                    if norm == Norm::LInf {
                        if x < image.width() - 1 {
                            check(image, x, y, x + 1, y + 1);
                        }
                        if x > 0 {
                            check(image, x, y, x - 1, y + 1);
                        }
                    }
                }
            }
        }
    }
}

// Sets image[current_x, current_y] to min(image[current_x, current_y], image[candidate_x, candidate_y] + 1).
// We avoid overflow by performing the arithmetic at type u16. We could use u8::saturating_add instead, but
// (based on the benchmarks tests) this appears to be considerably slower.
unsafe fn check(
    image: &mut GrayImage,
    current_x: u32,
    current_y: u32,
    candidate_x: u32,
    candidate_y: u32,
) {
    let current = image.unsafe_get_pixel(current_x, current_y)[0] as u16;
    let candidate_incr = image.unsafe_get_pixel(candidate_x, candidate_y)[0] as u16 + 1;
    if candidate_incr < current {
        image.unsafe_put_pixel(current_x, current_y, Luma([candidate_incr as u8]));
    }
}

/// Computes the square of the `L2` (Euclidean) distance transform of `image`. Distances are to the
/// nearest foreground pixel, where a pixel is counted as foreground if it has non-zero value.
///
/// Uses the algorithm from [Distance Transforms of Sampled Functions] to achieve time linear
/// in the size of the image.
///
/// [Distance Transforms of Sampled Functions]: https://www.cs.cornell.edu/~dph/papers/dt.pdf
pub fn euclidean_squared_distance_transform(image: &Image<Luma<u8>>) -> Image<Luma<f64>> {
    let (width, height) = image.dimensions();
    let mut result = ImageBuffer::new(width, height);
    let mut column_envelope = LowerEnvelope::new(height as usize);

    // Compute 1d transforms of each column
    for x in 0..width {
        let source = Column { image, column: x };
        let mut sink = ColumnMut {
            image: &mut result,
            column: x,
        };
        distance_transform_1d_mut(&source, &mut sink, &mut column_envelope);
    }

    let mut row_buffer = vec![0f64; width as usize];
    let mut row_envelope = LowerEnvelope::new(width as usize);

    // Compute 1d transforms of each row
    for y in 0..height {
        for x in 0..width {
            row_buffer[x as usize] = result.get_pixel(x, y)[0];
        }
        let mut sink = Row {
            image: &mut result,
            row: y,
        };
        distance_transform_1d_mut(&row_buffer, &mut sink, &mut row_envelope);
    }

    result
}

struct LowerEnvelope {
    // Indices of the parabolas in the lower envelope.
    locations: Vec<usize>,
    // Points at which the parabola in the lower envelope
    // changes. The parabola centred at locations[i] has the least
    // values of all parabolas in the lower envelope for all
    // coordinates in [ boundaries[i], boundaries[i + 1] ).
    boundaries: Vec<f64>,
}

impl LowerEnvelope {
    fn new(image_side: usize) -> LowerEnvelope {
        LowerEnvelope {
            locations: vec![0; image_side],
            boundaries: vec![f64::NAN; image_side + 1],
        }
    }
}

trait Sink {
    fn put(&mut self, idx: usize, value: f64);
    fn len(&self) -> usize;
}

trait Source {
    fn get(&self, idx: usize) -> f64;
    fn len(&self) -> usize;
}

struct Row<'a> {
    image: &'a mut Image<Luma<f64>>,
    row: u32,
}

impl<'a> Sink for Row<'a> {
    fn put(&mut self, idx: usize, value: f64) {
        unsafe {
            self.image
                .unsafe_put_pixel(idx as u32, self.row, Luma([value]));
        }
    }
    fn len(&self) -> usize {
        self.image.width() as usize
    }
}

struct ColumnMut<'a> {
    image: &'a mut Image<Luma<f64>>,
    column: u32,
}

impl<'a> Sink for ColumnMut<'a> {
    fn put(&mut self, idx: usize, value: f64) {
        unsafe {
            self.image
                .unsafe_put_pixel(self.column, idx as u32, Luma([value]));
        }
    }
    fn len(&self) -> usize {
        self.image.height() as usize
    }
}

impl Source for Vec<f64> {
    fn get(&self, idx: usize) -> f64 {
        self[idx]
    }
    fn len(&self) -> usize {
        self.len()
    }
}

impl Source for [f64] {
    fn get(&self, idx: usize) -> f64 {
        self[idx]
    }
    fn len(&self) -> usize {
        self.len()
    }
}

struct Column<'a> {
    image: &'a Image<Luma<u8>>,
    column: u32,
}

impl<'a> Source for Column<'a> {
    fn get(&self, idx: usize) -> f64 {
        let pixel = unsafe { self.image.unsafe_get_pixel(self.column, idx as u32)[0] as f64 };
        if pixel > 0f64 {
            0f64
        } else {
            f64::INFINITY
        }
    }
    fn len(&self) -> usize {
        self.image.height() as usize
    }
}

fn distance_transform_1d_mut<S, T>(f: &S, result: &mut T, envelope: &mut LowerEnvelope)
where
    S: Source,
    T: Sink,
{
    assert!(result.len() == f.len());
    assert!(envelope.boundaries.len() == f.len() + 1);
    assert!(envelope.locations.len() == f.len());

    if f.len() == 0 {
        return;
    }

    // Index of rightmost parabola in the lower envelope
    let mut k = 0;

    // First parabola is the best current value as we've not looked
    // at any other yet
    envelope.locations[0] = 0;

    // First parabola has the lowest value for all x coordinates
    envelope.boundaries[0] = f64::NEG_INFINITY;
    envelope.boundaries[1] = f64::INFINITY;

    for q in 1..f.len() {
        if f.get(q) == f64::INFINITY {
            continue;
        }

        if k == 0 && f.get(envelope.locations[k]) == f64::INFINITY {
            envelope.locations[k] = q;
            envelope.boundaries[k] = f64::NEG_INFINITY;
            envelope.boundaries[k + 1] = f64::INFINITY;
            continue;
        }

        // Let p = locations[k], i.e. the centre of the rightmost
        // parabola in the current approximation to the lower envelope.
        //
        // We find the intersection of this parabola with
        // the parabola centred at q to determine if the latter
        // is part of the lower envelope (and if the former should
        // be removed from our current approximation to it).
        let mut s = intersection(f, envelope.locations[k], q);

        while s <= envelope.boundaries[k] {
            // The parabola centred at q is the best we've seen for an
            // intervals that extends past the lower bound of the region
            // where we believed that the parabola centred at p gave the
            // least value
            k -= 1;
            s = intersection(f, envelope.locations[k], q);
        }

        k += 1;
        envelope.locations[k] = q;
        envelope.boundaries[k] = s;
        envelope.boundaries[k + 1] = f64::INFINITY;
    }

    let mut k = 0;
    for q in 0..f.len() {
        while envelope.boundaries[k + 1] < q as f64 {
            k += 1;
        }
        let dist = q as f64 - envelope.locations[k] as f64;
        result.put(q, dist * dist + f.get(envelope.locations[k]));
    }
}

/// Returns the intersection of the parabolas f(p) + (x - p) ^ 2 and f(q) + (x - q) ^ 2.
fn intersection<S: Source + ?Sized>(f: &S, p: usize, q: usize) -> f64 {
    // The intersection s of the two parabolas satisfies:
    //
    // f[q] + (q - s) ^ 2 = f[p] + (s - q) ^ 2
    //
    // Rearranging gives:
    //
    // s = [( f[q] + q ^ 2 ) - ( f[p] + p ^ 2 )] / (2q - 2p)
    let fq = f.get(q);
    let fp = f.get(p);
    let p = p as f64;
    let q = q as f64;

    ((fq + q * q) - (fp + p * p)) / (2.0 * q - 2.0 * p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::Image;
    use crate::property_testing::GrayTestImage;
    use crate::utils::{gray_bench_image, pixel_diff_summary};
    use image::{GrayImage, Luma};
    use quickcheck::{quickcheck, TestResult};
    use std::cmp::max;
    use std::f64;
    use test::{black_box, Bencher};

    #[test]
    fn test_distance_transform_saturation() {
        // A single foreground pixel in the top-left
        let image = GrayImage::from_fn(300, 300, |x, y| match (x, y) {
            (0, 0) => Luma([255u8]),
            _ => Luma([0u8]),
        });

        // Distances should not overflow
        let expected = GrayImage::from_fn(300, 300, |x, y| Luma([min(255, max(x, y)) as u8]));

        let distances = distance_transform(&image, Norm::LInf);
        assert_pixels_eq!(distances, expected);
    }

    impl<'a> Sink for Vec<f64> {
        fn put(&mut self, idx: usize, value: f64) {
            self[idx] = value;
        }
        fn len(&self) -> usize {
            self.len()
        }
    }

    fn distance_transform_1d(f: &Vec<f64>) -> Vec<f64> {
        let mut r = vec![0.0; f.len()];
        let mut e = LowerEnvelope::new(f.len());
        distance_transform_1d_mut(f, &mut r, &mut e);
        r
    }

    #[test]
    fn test_distance_transform_1d_constant() {
        let f = vec![0.0, 0.0, 0.0];
        let dists = distance_transform_1d(&f);
        assert_eq!(dists, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_distance_transform_1d_descending_gradient() {
        let f = vec![7.0, 5.0, 3.0, 1.0];
        let dists = distance_transform_1d(&f);
        assert_eq!(dists, &[6.0, 4.0, 2.0, 1.0]);
    }

    #[test]
    fn test_distance_transform_1d_ascending_gradient() {
        let f = vec![1.0, 3.0, 5.0, 7.0];
        let dists = distance_transform_1d(&f);
        assert_eq!(dists, &[1.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_distance_transform_1d_with_infinities() {
        let f = vec![f64::INFINITY, f64::INFINITY, 5.0, f64::INFINITY];
        let dists = distance_transform_1d(&f);
        assert_eq!(dists, &[9.0, 6.0, 5.0, 6.0]);
    }

    // Simple implementation of 1d distance transform which performs an
    // exhaustive search. Used to valid the more complicated lower-envelope
    // implementation against.
    fn distance_transform_1d_reference(f: &[f64]) -> Vec<f64> {
        let mut ret = vec![0.0; f.len()];
        for q in 0..f.len() {
            ret[q] = (0..f.len())
                .map(|p| {
                    let dist = p as f64 - q as f64;
                    dist * dist + f[p]
                })
                .fold(0.0 / 0.0, f64::min);
        }
        ret
    }

    #[test]
    fn test_distance_transform_1d_matches_reference_implementation() {
        fn prop(f: Vec<f64>) -> bool {
            let expected = distance_transform_1d_reference(&f);
            let actual = distance_transform_1d(&f);
            expected == actual
        }
        quickcheck(prop as fn(Vec<f64>) -> bool);
    }

    fn euclidean_squared_distance_transform_reference(image: &Image<Luma<u8>>) -> Image<Luma<f64>> {
        let (width, height) = image.dimensions();

        let mut dists = Image::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut min = f64::INFINITY;
                for yc in 0..height {
                    for xc in 0..width {
                        let pc = image.get_pixel(xc, yc)[0];
                        if pc > 0 {
                            let dx = xc as f64 - x as f64;
                            let dy = yc as f64 - y as f64;

                            min = f64::min(min, dx * dx + dy * dy);
                        }
                    }
                }

                dists.put_pixel(x, y, Luma([min]));
            }
        }

        dists
    }

    #[test]
    fn test_euclidean_squared_distance_transform_matches_reference_implementation() {
        fn prop(image: GrayTestImage) -> TestResult {
            let expected = euclidean_squared_distance_transform_reference(&image.0);
            let actual = euclidean_squared_distance_transform(&image.0);
            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err),
            }
        }
        quickcheck(prop as fn(GrayTestImage) -> TestResult);
    }

    #[test]
    fn test_euclidean_squared_distance_transform_example() {
        let image = gray_image!(
            1, 0, 0, 0, 0;
            0, 1, 0, 0, 0;
            1, 1, 1, 0, 0;
            0, 0, 0, 0, 0;
            0, 0, 1, 0, 0
        );

        let expected = gray_image!(type: f64,
            0.0, 1.0, 2.0, 5.0, 8.0;
            1.0, 0.0, 1.0, 2.0, 5.0;
            0.0, 0.0, 0.0, 1.0, 4.0;
            1.0, 1.0, 1.0, 2.0, 5.0;
            4.0, 1.0, 0.0, 1.0, 4.0
        );

        let dist = euclidean_squared_distance_transform(&image);
        assert_pixels_eq_within!(dist, expected, 1e-6);
    }

    macro_rules! bench_euclidean_squared_distance_transform {
        ($name:ident, side: $s:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let distance = euclidean_squared_distance_transform(&image);
                    black_box(distance);
                })
            }
        };
    }

    bench_euclidean_squared_distance_transform!(bench_euclidean_squared_distance_transform_10, side: 10);
    bench_euclidean_squared_distance_transform!(bench_euclidean_squared_distance_transform_100, side: 100);
    bench_euclidean_squared_distance_transform!(bench_euclidean_squared_distance_transform_200, side: 200);

    macro_rules! bench_distance_transform {
        ($name:ident, $norm:expr, side: $s:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let distance = distance_transform(&image, $norm);
                    black_box(distance);
                })
            }
        };
    }

    bench_distance_transform!(bench_distance_transform_l1_10, Norm::L1, side: 10);
    bench_distance_transform!(bench_distance_transform_l1_100, Norm::L1, side: 100);
    bench_distance_transform!(bench_distance_transform_l1_200, Norm::L1, side: 200);
    bench_distance_transform!(bench_distance_transform_linf_10, Norm::LInf, side: 10);
    bench_distance_transform!(bench_distance_transform_linf_100, Norm::LInf, side: 100);
    bench_distance_transform!(bench_distance_transform_linf_200, Norm::LInf, side: 200);
}
