//! Functions for suppressing non-maximal values.

use crate::definitions::{Position, Score};
use image::{GenericImage, ImageBuffer, Luma, Primitive};
use std::cmp;

/// Returned image has zeroes for all inputs pixels which do not have the greatest
/// intensity in the (2 * radius + 1) square block centred on them.
/// Ties are resolved lexicographically.
pub fn suppress_non_maximum<I, C>(image: &I, radius: u32) -> ImageBuffer<Luma<C>, Vec<C>>
where
    I: GenericImage<Pixel = Luma<C>>,
    C: Primitive + Ord + 'static,
{
    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<Luma<C>, Vec<C>> = ImageBuffer::new(width, height);
    if width == 0 || height == 0 {
        return out;
    }

    // We divide the image into a grid of blocks of size r * r. We find the maximum
    // value in each block, and then test whether this is in fact the maximum value
    // in the (2r + 1) * (2r + 1) block centered on it. Any pixel that's not maximal
    // within its r * r grid cell can't be a local maximum so we need only perform
    // the (2r + 1) * (2r + 1) search once per r * r grid cell (as opposed to once
    // per pixel in the naive implementation of this algorithm).

    for y in (0..height).step_by(radius as usize + 1) {
        for x in (0..width).step_by(radius as usize + 1) {
            let mut best_x = x;
            let mut best_y = y;
            let mut mi = image.get_pixel(x, y)[0];

            // These mins are necessary for when radius > min(width, height)
            for cy in y..cmp::min(height, y + radius + 1) {
                for cx in x..cmp::min(width, x + radius + 1) {
                    let ci = unsafe { image.unsafe_get_pixel(cx, cy)[0] };
                    if ci < mi {
                        continue;
                    }
                    if ci > mi || (cx, cy) < (best_x, best_y) {
                        best_x = cx;
                        best_y = cy;
                        mi = ci;
                    }
                }
            }

            let x0 = if radius >= best_x { 0 } else { best_x - radius };
            let x1 = x;
            let x2 = cmp::min(width, x + radius + 1);
            let x3 = cmp::min(width, best_x + radius + 1);

            let y0 = if radius >= best_y { 0 } else { best_y - radius };
            let y1 = y;
            let y2 = cmp::min(height, y + radius + 1);
            let y3 = cmp::min(height, best_y + radius + 1);

            // Above initial r * r block
            let mut failed = contains_greater_value(image, best_x, best_y, mi, y0, y1, x0, x3);
            // Left of initial r * r block
            failed |= contains_greater_value(image, best_x, best_y, mi, y1, y2, x0, x1);
            // Right of initial r * r block
            failed |= contains_greater_value(image, best_x, best_y, mi, y1, y2, x2, x3);
            // Below initial r * r block
            failed |= contains_greater_value(image, best_x, best_y, mi, y2, y3, x0, x3);

            if !failed {
                unsafe { out.unsafe_put_pixel(best_x, best_y, Luma([mi])) };
            }
        }
    }

    out
}

/// Returns true if the given block contains a larger value than
/// the input, or contains an equal value with lexicographically
/// lesser coordinates.
fn contains_greater_value<I, C>(
    image: &I,
    x: u32,
    y: u32,
    v: C,
    y_lower: u32,
    y_upper: u32,
    x_lower: u32,
    x_upper: u32,
) -> bool
where
    I: GenericImage<Pixel = Luma<C>>,
    C: Primitive + Ord + 'static,
{
    for cy in y_lower..y_upper {
        for cx in x_lower..x_upper {
            let ci = unsafe { image.unsafe_get_pixel(cx, cy)[0] };
            if ci < v {
                continue;
            }
            if ci > v || (cx, cy) < (x, y) {
                return true;
            }
        }
    }
    false
}

/// Returns all items which have the highest score in the
/// (2 * radius + 1) square block centred on them. Ties are resolved lexicographically.
pub fn local_maxima<T>(ts: &[T], radius: u32) -> Vec<T>
where
    T: Position + Score + Copy,
{
    let mut ordered_ts = ts.to_vec();
    ordered_ts.sort_by_key(|&c| (c.y(), c.x()));
    let height = match ordered_ts.last() {
        Some(t) => t.y(),
        None => 0,
    };

    let mut ts_by_row = vec![vec![]; (height + 1) as usize];
    for t in &ordered_ts {
        ts_by_row[t.y() as usize].push(t);
    }

    let mut max_ts = vec![];
    for t in &ordered_ts {
        let cx = t.x();
        let cy = t.y();
        let cs = t.score();

        let mut is_max = true;
        let row_lower = if radius > cy { 0 } else { cy - radius };
        let row_upper = if cy + radius + 1 > height {
            height
        } else {
            cy + radius + 1
        };
        for y in row_lower..row_upper {
            for c in &ts_by_row[y as usize] {
                if c.x() + radius < cx {
                    continue;
                }
                if c.x() > cx + radius {
                    break;
                }
                if c.score() > cs {
                    is_max = false;
                    break;
                }
                if c.score() < cs {
                    continue;
                }
                // Break tiebreaks lexicographically
                if (c.y(), c.x()) < (cy, cx) {
                    is_max = false;
                    break;
                }
            }
            if !is_max {
                break;
            }
        }

        if is_max {
            max_ts.push(*t);
        }
    }

    max_ts
}

#[cfg(test)]
mod tests {
    use super::{local_maxima, suppress_non_maximum};
    use crate::definitions::{Position, Score};
    use crate::noise::gaussian_noise_mut;
    use crate::property_testing::GrayTestImage;
    use crate::utils::pixel_diff_summary;
    use image::{GenericImage, GrayImage, ImageBuffer, Luma, Primitive};
    use quickcheck::{quickcheck, TestResult};
    use std::cmp;
    use test::Bencher;

    #[derive(PartialEq, Debug, Copy, Clone)]
    struct T {
        x: u32,
        y: u32,
        score: f32,
    }

    impl T {
        fn new(x: u32, y: u32, score: f32) -> T {
            T { x, y, score }
        }
    }

    impl Position for T {
        fn x(&self) -> u32 {
            self.x
        }
        fn y(&self) -> u32 {
            self.y
        }
    }

    impl Score for T {
        fn score(&self) -> f32 {
            self.score
        }
    }

    #[test]
    fn test_local_maxima() {
        let ts = vec![
            // Suppress vertically
            T::new(0, 0, 8f32),
            T::new(0, 3, 10f32),
            T::new(0, 6, 9f32),
            // Suppress horizontally
            T::new(5, 5, 10f32),
            T::new(7, 5, 15f32),
            // Tiebreak
            T::new(12, 20, 10f32),
            T::new(13, 20, 10f32),
            T::new(13, 21, 10f32),
        ];

        let expected = vec![
            T::new(0, 3, 10f32),
            T::new(7, 5, 15f32),
            T::new(12, 20, 10f32),
        ];

        let max = local_maxima(&ts, 3);
        assert_eq!(max, expected);
    }

    #[bench]
    fn bench_local_maxima_dense(b: &mut Bencher) {
        let mut ts = vec![];
        for x in 0..20 {
            for y in 0..20 {
                let score = (x * y) % 15;
                ts.push(T::new(x, y, score as f32));
            }
        }
        b.iter(|| local_maxima(&ts, 15));
    }

    #[bench]
    fn bench_local_maxima_sparse(b: &mut Bencher) {
        let mut ts = vec![];
        for x in 0..20 {
            for y in 0..20 {
                ts.push(T::new(50 * x, 50 * y, 50f32));
            }
        }
        b.iter(|| local_maxima(&ts, 15));
    }

    #[test]
    fn test_suppress_non_maximum() {
        let mut image = GrayImage::new(25, 25);
        // Suppress vertically
        image.put_pixel(0, 0, Luma([8u8]));
        image.put_pixel(0, 3, Luma([10u8]));
        image.put_pixel(0, 6, Luma([9u8]));
        // Suppress horizontally
        image.put_pixel(5, 5, Luma([10u8]));
        image.put_pixel(7, 5, Luma([15u8]));
        // Tiebreak
        image.put_pixel(12, 20, Luma([10u8]));
        image.put_pixel(13, 20, Luma([10u8]));
        image.put_pixel(13, 21, Luma([10u8]));

        let mut expected = GrayImage::new(25, 25);
        expected.put_pixel(0, 3, Luma([10u8]));
        expected.put_pixel(7, 5, Luma([15u8]));
        expected.put_pixel(12, 20, Luma([10u8]));

        let actual = suppress_non_maximum(&image, 3);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_suppress_non_maximum_handles_radius_greater_than_image_side() {
        // Don't care about output pixels, just want to make sure that
        // we don't go out of bounds when radius exceeds width or height.
        let image = GrayImage::new(7, 3);
        let r = suppress_non_maximum(&image, 5);
        let image = GrayImage::new(3, 7);
        let s = suppress_non_maximum(&image, 5);
        // Use r and s to silence warnings about unused variables.
        assert!(r.width() == 7);
        assert!(s.width() == 3);
    }

    #[bench]
    fn bench_suppress_non_maximum_increasing_gradient(b: &mut Bencher) {
        // Increasing gradient in both directions. This can be a worst-case for
        // early-abort strategies.
        let img = ImageBuffer::from_fn(40, 20, |x, y| Luma([(x + y) as u8]));
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    #[bench]
    fn bench_suppress_non_maximum_decreasing_gradient(b: &mut Bencher) {
        let width = 40u32;
        let height = 20u32;
        let img = ImageBuffer::from_fn(width, height, |x, y| {
            Luma([((width - x) + (height - y)) as u8])
        });
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    #[bench]
    fn bench_suppress_non_maximum_noise_7(b: &mut Bencher) {
        let mut img: GrayImage = ImageBuffer::new(40, 20);
        gaussian_noise_mut(&mut img, 128f64, 30f64, 1);
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    #[bench]
    fn bench_suppress_non_maximum_noise_3(b: &mut Bencher) {
        let mut img: GrayImage = ImageBuffer::new(40, 20);
        gaussian_noise_mut(&mut img, 128f64, 30f64, 1);
        b.iter(|| suppress_non_maximum(&img, 3));
    }

    #[bench]
    fn bench_suppress_non_maximum_noise_1(b: &mut Bencher) {
        let mut img: GrayImage = ImageBuffer::new(40, 20);
        gaussian_noise_mut(&mut img, 128f64, 30f64, 1);
        b.iter(|| suppress_non_maximum(&img, 1));
    }

    /// Reference implementation of suppress_non_maximum. Used to validate
    /// the (presumably faster) actual implementation.
    fn suppress_non_maximum_reference<I, C>(image: &I, radius: u32) -> ImageBuffer<Luma<C>, Vec<C>>
    where
        I: GenericImage<Pixel = Luma<C>>,
        C: Primitive + Ord + 'static,
    {
        let (width, height) = image.dimensions();
        let mut out = ImageBuffer::new(width, height);
        out.copy_from(image, 0, 0).unwrap();

        let iradius = radius as i32;
        let iheight = height as i32;
        let iwidth = width as i32;

        // We update zero values from out as we go, so to check intensities
        // we need to read values from the input image.
        for y in 0..height {
            for x in 0..width {
                let intensity = image.get_pixel(x, y)[0];
                let mut is_max = true;

                let y_lower = cmp::max(0, y as i32 - iradius);
                let y_upper = cmp::min(y as i32 + iradius + 1, iheight);
                let x_lower = cmp::max(0, x as i32 - iradius);
                let x_upper = cmp::min(x as i32 + iradius + 1, iwidth);

                for py in y_lower..y_upper {
                    for px in x_lower..x_upper {
                        let v = image.get_pixel(px as u32, py as u32)[0];
                        // Handle intensity tiebreaks lexicographically
                        let candidate_is_lexically_earlier = (px as u32, py as u32) < (x, y);
                        if v > intensity || (v == intensity && candidate_is_lexically_earlier) {
                            is_max = false;
                            break;
                        }
                    }
                }

                if !is_max {
                    out.put_pixel(x, y, Luma([C::zero()]));
                }
            }
        }

        out
    }

    #[test]
    fn test_suppress_non_maximum_matches_reference_implementation() {
        fn prop(image: GrayTestImage) -> TestResult {
            let expected = suppress_non_maximum_reference(&image.0, 3);
            let actual = suppress_non_maximum(&image.0, 3);
            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err),
            }
        }
        quickcheck(prop as fn(GrayTestImage) -> TestResult);
    }

    #[test]
    fn test_step() {
        assert_eq!((0u32..5).step_by(4).collect::<Vec<u32>>(), vec![0, 4]);
        assert_eq!((0u32..4).step_by(4).collect::<Vec<u32>>(), vec![0]);
        assert_eq!((4u32..4).step_by(4).collect::<Vec<u32>>(), vec![]);
    }
}
