use crate::definitions::Image;
use image::{GenericImageView, Pixel};
use std::cmp::{max, min};

/// Applies a median filter of given dimensions to an image. Each output pixel is the median
/// of the pixels in a `(2 * x_radius + 1) * (2 * y_radius + 1)` kernel of pixels in the input image.
///
/// Pads by continuity. Performs O(max(x_radius, y_radius)) operations per pixel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
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
///
/// let filtered = gray_image!(
///     2,  3,  3;
///     9,  7,  7;
///     9, 11, 11
/// );
///
/// assert_pixels_eq!(median_filter(&image, 1, 1), filtered);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
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
/// assert_pixels_eq!(median_filter(&image, 1, 1), filtered);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
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
/// assert_pixels_eq!(median_filter(&image, 2, 1), filtered);
/// # }
/// ```
pub fn median_filter<P>(image: &Image<P>, x_radius: u32, y_radius: u32) -> Image<P>
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        return image.clone();
    }

    let mut out = Image::<P>::new(width, height);
    let rx = x_radius as i32;
    let ry = y_radius as i32;

    let mut hist = initialise_histogram_for_top_left_pixel(&image, x_radius, y_radius);
    slide_down_column(&mut hist, &image, &mut out, 0, rx, ry);

    for x in 1..width {
        if x % 2 == 0 {
            slide_right(&mut hist, &image, x, 0, rx, ry);
            slide_down_column(&mut hist, &image, &mut out, x, rx, ry);
        } else {
            slide_right(&mut hist, &image, x, height - 1, rx, ry);
            slide_up_column(&mut hist, &image, &mut out, x, rx, ry);
        }
    }
    out
}

fn initialise_histogram_for_top_left_pixel<P>(
    image: &Image<P>,
    x_radius: u32,
    y_radius: u32,
) -> HistSet
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let (width, height) = image.dimensions();
    let kernel_size = (2 * x_radius + 1) * (2 * y_radius + 1);
    let num_channels = P::CHANNEL_COUNT;

    let mut hist = HistSet::new(num_channels, kernel_size);
    let rx = x_radius as i32;
    let ry = y_radius as i32;

    for dy in -ry..(ry + 1) {
        let py = min(max(0, dy), height as i32 - 1) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, dx), width as i32 - 1) as u32;

            hist.incr(image, px, py);
        }
    }

    hist
}

fn slide_right<P>(hist: &mut HistSet, image: &Image<P>, x: u32, y: u32, rx: i32, ry: i32)
where
    P: Pixel<Subpixel = u8> + 'static,
{
    let (width, height) = image.dimensions();

    let prev_x = max(0, x as i32 - rx - 1) as u32;
    let next_x = min(x as i32 + rx, width as i32 - 1) as u32;

    for dy in -ry..(ry + 1) {
        let py = min(max(0, y as i32 + dy), (height - 1) as i32) as u32;

        hist.decr(image, prev_x, py);
        hist.incr(image, next_x, py);
    }
}

fn slide_down_column<P>(
    hist: &mut HistSet,
    image: &Image<P>,
    out: &mut Image<P>,
    x: u32,
    rx: i32,
    ry: i32,
) where
    P: Pixel<Subpixel = u8> + 'static,
{
    let (width, height) = image.dimensions();
    hist.set_to_median(out, x, 0);

    for y in 1..height {
        let prev_y = max(0, y as i32 - ry - 1) as u32;
        let next_y = min(y as i32 + ry, height as i32 - 1) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            hist.decr(image, px, prev_y);
            hist.incr(image, px, next_y);
        }

        hist.set_to_median(out, x, y);
    }
}

fn slide_up_column<P>(
    hist: &mut HistSet,
    image: &Image<P>,
    out: &mut Image<P>,
    x: u32,
    rx: i32,
    ry: i32,
) where
    P: Pixel<Subpixel = u8> + 'static,
{
    let (width, height) = image.dimensions();
    hist.set_to_median(out, x, height - 1);

    for y in (0..(height - 1)).rev() {
        let prev_y = min(y as i32 + ry + 1, height as i32 - 1) as u32;
        let next_y = max(0, y as i32 - ry) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            hist.decr(image, px, prev_y);
            hist.incr(image, px, next_y);
        }

        hist.set_to_median(out, x, y);
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
        // Can't use vec![[0u32; 256], num_channels as usize]
        // because arrays of length > 32 aren't cloneable.
        let mut data = vec![];
        for _ in 0..num_channels {
            data.push([0u32; 256]);
        }

        HistSet {
            data,
            expected_count,
        }
    }

    fn incr<P>(&mut self, image: &Image<P>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8> + 'static,
    {
        unsafe {
            let pixel = image.unsafe_get_pixel(x, y);
            let channels = pixel.channels();
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) += 1;
            }
        }
    }

    fn decr<P>(&mut self, image: &Image<P>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8> + 'static,
    {
        unsafe {
            let pixel = image.unsafe_get_pixel(x, y);
            let channels = pixel.channels();
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) -= 1;
            }
        }
    }

    fn set_to_median<P>(&self, image: &mut Image<P>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8> + 'static,
    {
        unsafe {
            let target = image.get_pixel_mut(x, y);
            let channels = target.channels_mut();
            for c in 0..channels.len() {
                *channels.get_unchecked_mut(c) = self.channel_median(c as u8);
            }
        }
    }

    fn channel_median(&self, c: u8) -> u8 {
        let hist = unsafe { self.data.get_unchecked(c as usize) };

        let mut count = 0;

        for i in 0..256 {
            unsafe {
                count += *hist.get_unchecked(i);
            }

            if 2 * count >= self.expected_count {
                return i as u8;
            }
        }

        255
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property_testing::GrayTestImage;
    use crate::utils::gray_bench_image;
    use crate::utils::pixel_diff_summary;
    use image::{GrayImage, Luma};
    use quickcheck::{quickcheck, TestResult};
    use std::cmp::{max, min};
    use test::{black_box, Bencher};

    macro_rules! bench_median_filter {
        ($name:ident, side: $s:expr, x_radius: $rx:expr, y_radius: $ry:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let filtered = median_filter(&image, $rx, $ry);
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

                        neighbors[idx] = image.get_pixel(px, py)[0] as u8;

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

    #[test]
    fn test_median_filter_matches_reference_implementation() {
        fn prop(image: GrayTestImage, x_radius: u32, y_radius: u32) -> TestResult {
            let x_radius = x_radius % 5;
            let y_radius = y_radius % 5;
            let expected = reference_median_filter(&image.0, x_radius, y_radius);
            let actual = median_filter(&image.0, x_radius, y_radius);

            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err),
            }
        }
        quickcheck(prop as fn(GrayTestImage, u32, u32) -> TestResult);
    }
}
