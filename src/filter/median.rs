use crate::definitions::{BoundaryAccess, Image};
use crate::geometric_transformations::Border;
use image::Pixel;

/// Applies a median filter of given dimensions to an image. Each output pixel is the median
/// of the pixels in a `(2 * x_radius + 1) * (2 * y_radius + 1)` kernel of pixels in the input image.
///
/// Sampling outside of image boundaries is controlled by `extend`.
/// Performs O(max(x_radius, y_radius)) operations per pixel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
/// use imageproc::geometric_transformations::Border;
///
/// let image = gray_image!(
///     1,   2,   3;
///   200,   6,   7;
///     9, 100,  11
/// );
///
/// // Padding by continuity means that the values we use
/// // for computing medians of boundary pixels are:
/// //
/// //   1     1     2     3     3
/// //      -----------------
/// //   1 |   1     2     3 |   3
/// //
/// // 200 | 200     6     7 |   7
/// //
/// //   9 |   9   100    11 |  11
/// //      -----------------
/// //   9     9   100    11    11
/// //
/// // Here we choose `Border<P>` corresponding to it:
/// let extend = Border::Replicate;
///
/// let filtered = gray_image!(
///     2,  3,  3;
///     9,  7,  7;
///     9, 11, 11
/// );
///
/// assert_pixels_eq!(median_filter(&image, 1, 1, extend), filtered);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
/// use imageproc::geometric_transformations::Border;
///
/// // Image channels are handled independently.
/// // This example sets the red channel to have the same
/// // contents as the image from the grayscale example,
/// // the green channel to a vertically inverted copy of that
/// // image and the blue channel to be constant.
/// //
/// // See the grayscale image example for an explanation of how
/// // boundary conditions are handled.
///
/// let image = rgb_image!(
///     [  1,   9, 10], [  2, 100,  10], [  3,  11,  10];
///     [200, 200, 10], [  6,   6,  10], [  7,   7,  10];
///     [  9,   1, 10], [100,   2,  10], [ 11,   3,  10]
/// );
///
/// let filtered = rgb_image!(
///     [ 2,  9, 10], [ 3, 11, 10], [ 3, 11, 10];
///     [ 9,  9, 10], [ 7,  7, 10], [ 7,  7, 10];
///     [ 9,  2, 10], [11,  3, 10], [11,  3, 10]
/// );
///
/// assert_pixels_eq!(median_filter(&image, 1, 1, Border::Replicate), filtered);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
/// use imageproc::geometric_transformations::Border;
///
/// // This example uses a kernel with x_radius sets to 2
/// // and y_radius sets to 1, which leads to 5 * 3 kernel size.
///
/// let image = gray_image!(
///     1, 2, 3, 4, 5;
///     255, 200, 4, 11, 7;
///     42, 17, 3, 2, 1;
///     9, 100, 11, 13, 14;
///     15, 87, 99, 21, 45
/// );
///
/// let filtered = gray_image!(
///     2, 3, 4, 5, 5;
///     17, 4, 4, 4, 4;
///     42, 13, 11, 11, 7;
///     15, 15, 15, 14, 14;
///     15, 15, 21, 45, 45
/// );
///
/// assert_pixels_eq!(median_filter(&image, 2, 1, Border::Replicate), filtered);
/// # }
/// ```
#[must_use = "the function does not modify the original image"]
pub fn median_filter<P>(
    image: &Image<P>,
    x_radius: u32,
    y_radius: u32,
    extend: Border<P>,
) -> Image<P>
where
    P: Pixel<Subpixel = u8>,
{
    let (width, height) = image.dimensions();

    // Safety note: we rely on image dimensions being non-zero for uncheched indexing to be in bounds
    if width == 0 || height == 0 {
        return image.clone();
    }

    // Safety note: we perform unchecked indexing in several places after checking at type i32 that a coordinate is in bounds
    if (width + x_radius) > i32::MAX as u32 || (height + y_radius) > i32::MAX as u32 {
        panic!("(width + x_radius) and (height + y_radius) must both be <= i32::MAX");
    }

    let mut out = Image::<P>::new(width, height);
    let mut hist = initialise_histogram_for_top_left_pixel(image, x_radius, y_radius, extend);
    slide_down_column(&mut hist, image, &mut out, 0, x_radius, y_radius, extend);

    for x in 1..width {
        if x % 2 == 0 {
            slide_right(&mut hist, image, x, 0, x_radius, y_radius, extend);
            slide_down_column(&mut hist, image, &mut out, x, x_radius, y_radius, extend);
        } else {
            slide_right(&mut hist, image, x, height - 1, x_radius, y_radius, extend);
            slide_up_column(&mut hist, image, &mut out, x, x_radius, y_radius, extend);
        }
    }
    out
}

fn initialise_histogram_for_top_left_pixel<P>(
    image: &Image<P>,
    x_radius: u32,
    y_radius: u32,
    extend: Border<P>,
) -> HistSet
where
    P: Pixel<Subpixel = u8>,
{
    let kernel_size = (2 * x_radius + 1) * (2 * y_radius + 1);
    let num_channels = P::CHANNEL_COUNT;

    let mut hist = HistSet::new(num_channels, kernel_size);
    let rx = x_radius as i64;
    let ry = y_radius as i64;

    for dy in -ry..(ry + 1) {
        for dx in -rx..(rx + 1) {
            let pixel = image.get_pixel_or_extend(dx, dy, extend);
            unsafe {
                hist.incr(pixel);
            }
        }
    }

    hist
}

fn slide_right<P>(
    hist: &mut HistSet,
    image: &Image<P>,
    x: u32,
    y: u32,
    rx: u32,
    ry: u32,
    extend: Border<P>,
) where
    P: Pixel<Subpixel = u8>,
{
    let rx = rx as i64;
    let ry = ry as i64;

    let prev_x = x as i64 - rx - 1;
    let next_x = x as i64 + rx;

    for dy in -ry..(ry + 1) {
        let py = y as i64 + dy;

        let prev_pixel = image.get_pixel_or_extend(prev_x, py, extend);
        // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
        unsafe {
            hist.decr(prev_pixel);
        }
        let next_pixel = image.get_pixel_or_extend(next_x, py, extend);
        // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
        unsafe {
            hist.incr(next_pixel);
        }
    }
}

fn slide_down_column<P>(
    hist: &mut HistSet,
    image: &Image<P>,
    out: &mut Image<P>,
    x: u32,
    rx: u32,
    ry: u32,
    extend: Border<P>,
) where
    P: Pixel<Subpixel = u8>,
{
    let rx = rx as i64;
    let ry = ry as i64;

    // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
    unsafe {
        let pixel = out.get_pixel_mut(x, 0);
        hist.set_to_median(pixel);
    }

    for y in 1..image.height() {
        let prev_y = y as i64 - ry - 1;
        let next_y = y as i64 + ry;

        for dx in -rx..(rx + 1) {
            let px = x as i64 + dx;

            let prev_pixel = image.get_pixel_or_extend(px, prev_y, extend);
            // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
            unsafe {
                hist.decr(prev_pixel);
            }

            let next_pixel = image.get_pixel_or_extend(px, next_y, extend);
            // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
            unsafe {
                hist.incr(next_pixel);
            }
        }

        // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
        unsafe {
            let pixel = out.get_pixel_mut(x, y);
            hist.set_to_median(pixel);
        }
    }
}

fn slide_up_column<P>(
    hist: &mut HistSet,
    image: &Image<P>,
    out: &mut Image<P>,
    x: u32,
    rx: u32,
    ry: u32,
    extend: Border<P>,
) where
    P: Pixel<Subpixel = u8>,
{
    let height = image.height();

    let rx = rx as i64;
    let ry = ry as i64;

    // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
    unsafe {
        let pixel = out.get_pixel_mut(x, height - 1);
        hist.set_to_median(pixel);
    }

    for y in (0..(height - 1)).rev() {
        let prev_y = y as i64 + ry + 1;
        let next_y = y as i64 - ry;

        for dx in -rx..(rx + 1) {
            let px = x as i64 + dx;

            let prev_pixel = image.get_pixel_or_extend(px, prev_y, extend);
            // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
            unsafe {
                hist.decr(prev_pixel);
            }
            let next_pixel = image.get_pixel_or_extend(px, next_y, extend);
            // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
            unsafe {
                hist.incr(next_pixel);
            }
        }

        // Safety: hist.data.len() == P::CHANNEL_COUNT by construction
        unsafe {
            let pixel = out.get_pixel_mut(x, y);
            hist.set_to_median(pixel);
        }
    }
}

// A collection of 256-slot histograms, one per image channel.
// Used to implement median_filter.
struct HistSet {
    // One histogram per image channel.
    data: Vec<[u32; 256]>,
    // Calls to `median` will only return the correct answer
    // if there are `expected_count` entries in the relevant
    // histogram in `data`.
    expected_count: u32,
}

impl HistSet {
    fn new(num_channels: u8, expected_count: u32) -> HistSet {
        HistSet {
            data: vec![[0u32; 256]; num_channels as usize],
            expected_count,
        }
    }

    /// Safety: requires P::CHANNEL_COUNT <= self.data.len()
    unsafe fn incr<P: Pixel<Subpixel = u8>>(&mut self, pixel: P) {
        let channels = pixel.channels();
        unsafe {
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) += 1;
            }
        }
    }

    /// Safety: requires P::CHANNEL_COUNT <= self.data.len()
    unsafe fn decr<P: Pixel<Subpixel = u8>>(&mut self, pixel: P) {
        let channels = pixel.channels();
        unsafe {
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) -= 1;
            }
        }
    }

    /// Safety: requires P::CHANNEL_COUNT <= self.data.len()
    unsafe fn set_to_median<P: Pixel<Subpixel = u8>>(&self, pixel: &mut P) {
        let channels = pixel.channels_mut();
        unsafe {
            for c in 0..channels.len() {
                *channels.get_unchecked_mut(c) = self.channel_median(c as u8);
            }
        }
    }

    /// Safety: requires c < self.data.len()
    unsafe fn channel_median(&self, c: u8) -> u8 {
        unsafe {
            let hist = self.data.get_unchecked(c as usize);
            let mut count = 0;
            for i in 0..256 {
                count += *hist.get_unchecked(i);
                if 2 * count >= self.expected_count {
                    return i as u8;
                }
            }
            255
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use test::{Bencher, black_box};

    macro_rules! bench_median_filter {
        ($name:ident, side: $s:expr, x_radius: $rx:expr, y_radius: $ry:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let filtered = median_filter(&image, $rx, $ry, Border::Replicate);
                    black_box(filtered);
                })
            }
        };
    }

    bench_median_filter!(bench_median_filter_s100_r1, side: 100, x_radius: 1,y_radius: 1);
    bench_median_filter!(bench_median_filter_s100_r4, side: 100, x_radius: 4,y_radius: 4);
    bench_median_filter!(bench_median_filter_s100_r8, side: 100, x_radius: 8,y_radius: 8);

    // benchmark on non-square kernels
    bench_median_filter!(bench_median_filter_s100_rx1_ry4, side: 100, x_radius: 1,y_radius: 4);
    bench_median_filter!(bench_median_filter_s100_rx1_ry8, side: 100, x_radius: 1,y_radius: 8);
    bench_median_filter!(bench_median_filter_s100_rx4_ry8, side: 100, x_radius: 4,y_radius: 1);
    bench_median_filter!(bench_median_filter_s100_rx8_ry1, side: 100, x_radius: 8,y_radius: 1);
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::proptest_utils::arbitrary_image;
    use image::{GrayImage, Luma};
    use proptest::prelude::*;
    use std::cmp::{max, min};

    // Reference implementation of median filter - written to be as simple as possible,
    // to validate faster versions against.
    fn reference_median_filter(image: &GrayImage, x_radius: u32, y_radius: u32) -> GrayImage {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return image.clone();
        }

        let mut out = GrayImage::new(width, height);
        let x_filter_side = (2 * x_radius + 1) as usize;
        let y_filter_side = (2 * y_radius + 1) as usize;
        let mut neighbors = vec![0u8; x_filter_side * y_filter_side];

        let rx = x_radius as i32;
        let ry = y_radius as i32;

        for y in 0..height {
            for x in 0..width {
                let mut idx = 0;

                for dy in -ry..(ry + 1) {
                    for dx in -rx..(rx + 1) {
                        let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;
                        let py = min(max(0, y as i32 + dy), (height - 1) as i32) as u32;

                        neighbors[idx] = image.get_pixel(px, py)[0];

                        idx += 1;
                    }
                }

                neighbors.sort();

                let m = median(&neighbors);
                out.put_pixel(x, y, Luma([m]));
            }
        }

        out
    }

    fn median(sorted: &[u8]) -> u8 {
        let mid = sorted.len() / 2;
        sorted[mid]
    }

    proptest! {
        #[test]
        fn test_median_filter_matches_reference_implementation(image in arbitrary_image::<Luma<u8>>(0..10, 0..10), x_radius in 0_u32..5, y_radius in 0_u32..5) {
            let expected = reference_median_filter(&image, x_radius, y_radius);
            let actual = median_filter(&image, x_radius, y_radius, Border::Replicate);

            prop_assert_eq!(actual, expected);
        }
    }
}
