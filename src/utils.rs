//! Utils for testing and debugging.

use image::{DynamicImage, GenericImage, GenericImageView, GrayImage, Luma, open, Pixel, Rgb, RgbImage};

use std::u32;
use std::fmt;
use std::fmt::Write;
use std::path::Path;
use itertools::Itertools;
use std::collections::HashSet;
use std::cmp::{max, min};

/// Helper for defining greyscale images.
///
/// Columns are separated by commas and rows by semi-colons.
/// By default a subpixel type of `u8` is used but this can be
/// overridden, as shown in the examples.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::{GrayImage, ImageBuffer, Luma};
///
/// // An empty grayscale image with pixel type Luma<u8>
/// let empty = gray_image!();
///
/// assert_pixels_eq!(
///     empty,
///     GrayImage::from_raw(0, 0, vec![]).unwrap()
/// );
///
/// // A single pixel grayscale image with pixel type Luma<u8>
/// let single_pixel = gray_image!(1);
///
/// assert_pixels_eq!(
///     single_pixel,
///     GrayImage::from_raw(1, 1, vec![1]).unwrap()
/// );
///
/// // A single row grayscale image with pixel type Luma<u8>
/// let single_row = gray_image!(1, 2, 3);
///
/// assert_pixels_eq!(
///     single_row,
///     GrayImage::from_raw(3, 1, vec![1, 2, 3]).unwrap()
/// );
///
/// // A grayscale image with 2 rows and 3 columns
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let equivalent = GrayImage::from_raw(3, 2, vec![
///     1, 2, 3,
///     4, 5, 6
/// ]).unwrap();
///
/// // An empty grayscale image with pixel type Luma<i16>.
/// let empty_i16 = gray_image!(type: i16);
///
/// assert_pixels_eq!(
///     empty_i16,
///     ImageBuffer::<Luma<i16>, Vec<i16>>::from_raw(0, 0, vec![]).unwrap()
/// );
///
/// // A grayscale image with 2 rows, 3 columns and pixel type Luma<i16>
/// let image_i16 = gray_image!(type: i16,
///     1, 2, 3;
///     4, 5, 6);
///
/// let expected_i16 = ImageBuffer::<Luma<i16>, Vec<i16>>::from_raw(3, 2, vec![
///     1, 2, 3,
///     4, 5, 6]).unwrap();
///
/// assert_pixels_eq!(image_i16, expected_i16);
/// # }
/// ```
#[macro_export]
macro_rules! gray_image {
    // Empty image with default channel type u8
    () => {
        gray_image!(type: u8)
    };
        // Empty image with the given channel type
    (type: $channel_type:ty) => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<$channel_type>, Vec<$channel_type>>::new(0, 0)
        }
    };
    // Non-empty image of default channel type u8
    ($( $( $x: expr ),*);*) => {
        gray_image!(type: u8, $( $( $x ),*);*)
    };
    // Non-empty image of given channel type
    (type: $channel_type:ty, $( $( $x: expr ),*);*) => {
        {
            use image::{ImageBuffer, Luma};

            let nested_array = [ $( [ $($x),* ] ),* ];
            let height = nested_array.len() as u32;
            let width = nested_array[0].len() as u32;

            let flat_array: Vec<_> = nested_array.into_iter()
                .flat_map(|row| row.into_iter())
                .cloned()
                .collect();

            ImageBuffer::<Luma<$channel_type>, Vec<$channel_type>>::from_raw(width, height, flat_array)
                .unwrap()
        }
    }
}

/// Helper for defining RGB images.
///
/// Pixels are delineated by square brackets, columns are
/// separated by commas and rows are separated by semi-colons.
/// By default a subpixel type of `u8` is used but this can be
/// overridden, as shown in the examples.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::{ImageBuffer, Rgb, RgbImage};
///
/// // An empty image with pixel type Rgb<u8>
/// let empty = rgb_image!();
///
/// assert_pixels_eq!(
///     empty,
///     RgbImage::from_raw(0, 0, vec![]).unwrap()
/// );
///
/// // A single pixel image with pixel type Rgb<u8>
/// let single_pixel = rgb_image!([1, 2, 3]);
///
/// assert_pixels_eq!(
///     single_pixel,
///     RgbImage::from_raw(1, 1, vec![1, 2, 3]).unwrap()
/// );
///
/// // A single row image with pixel type Rgb<u8>
/// let single_row = rgb_image!([1, 2, 3], [4, 5, 6]);
///
/// assert_pixels_eq!(
///     single_row,
///     RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap()
/// );
///
/// // An image with 2 rows and 2 columns
/// let image = rgb_image!(
///     [1,  2,  3], [ 4,  5,  6];
///     [7,  8,  9], [10, 11, 12]);
///
/// let equivalent = RgbImage::from_raw(2, 2, vec![
///     1,  2,  3,  4,  5,  6,
///     7,  8,  9, 10, 11, 12
/// ]).unwrap();
///
/// assert_pixels_eq!(image, equivalent);
///
/// // An empty image with pixel type Rgb<i16>.
/// let empty_i16 = rgb_image!(type: i16);
///
/// // An image with 2 rows, 3 columns and pixel type Rgb<i16>
/// let image_i16 = rgb_image!(type: i16,
///     [1, 2, 3], [4, 5, 6];
///     [7, 8, 9], [10, 11, 12]);
///
/// let expected_i16 = ImageBuffer::<Rgb<i16>, Vec<i16>>::from_raw(2, 2, vec![
///     1, 2, 3, 4, 5, 6,
///     7, 8, 9, 10, 11, 12],
///     ).unwrap();
/// # }
/// ```
#[macro_export]
macro_rules! rgb_image {
    // Empty image with default channel type u8
    () => {
        rgb_image!(type: u8)
    };
    // Empty image with the given channel type
    (type: $channel_type:ty) => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<$channel_type>, Vec<$channel_type>>::new(0, 0)
        }
    };
    // Non-empty image of default channel type u8
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        rgb_image!(type: u8, $( $( [$r, $g, $b]),*);*)
    };
    // Non-empty image of given channel type
    (type: $channel_type:ty, $( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            use image::{ImageBuffer, Rgb};
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            let height = nested_array.len() as u32;
            let width = nested_array[0].len() as u32;

            let flat_array: Vec<_> = nested_array.into_iter()
                .flat_map(|row| row.into_iter().flat_map(|p| p.into_iter()))
                .cloned()
                .collect();

            ImageBuffer::<Rgb<$channel_type>, Vec<$channel_type>>::from_raw(width, height, flat_array)
                .unwrap()
        }
    }
}


/// Helper for defining RGBA images.
///
/// Pixels are delineated by square brackets, columns are
/// separated by commas and rows are separated by semi-colons.
/// By default a subpixel type of `u8` is used but this can be
/// overridden, as shown in the examples.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::{ImageBuffer, Rgba, RgbaImage};
///
/// // An empty image with pixel type Rgba<u8>
/// let empty = rgba_image!();
///
/// assert_pixels_eq!(
///     empty,
///     RgbaImage::from_raw(0, 0, vec![]).unwrap()
/// );
///
/// // A single pixel image with pixel type Rgba<u8>
/// let single_pixel = rgba_image!([1, 2, 3, 4]);
///
/// assert_pixels_eq!(
///     single_pixel,
///     RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4]).unwrap()
/// );
///
/// // A single row image with pixel type Rgba<u8>
/// let single_row = rgba_image!([1, 2, 3, 10], [4, 5, 6, 20]);
///
/// assert_pixels_eq!(
///     single_row,
///     RgbaImage::from_raw(2, 1, vec![1, 2, 3, 10, 4, 5, 6, 20]).unwrap()
/// );
///
/// // An image with 2 rows and 2 columns
/// let image = rgba_image!(
///     [1,  2,  3, 10], [ 4,  5,  6, 20];
///     [7,  8,  9, 30], [10, 11, 12, 40]);
///
/// let equivalent = RgbaImage::from_raw(2, 2, vec![
///     1,  2,  3, 10,  4,  5,  6, 20,
///     7,  8,  9, 30, 10, 11, 12, 40
/// ]).unwrap();
///
/// assert_pixels_eq!(image, equivalent);
///
/// // An empty image with pixel type Rgba<i16>.
/// let empty_i16 = rgba_image!(type: i16);
///
/// // An image with 2 rows, 3 columns and pixel type Rgba<i16>
/// let image_i16 = rgba_image!(type: i16,
///     [1, 2, 3, 10], [ 4,  5,  6, 20];
///     [7, 8, 9, 30], [10, 11, 12, 40]);
///
/// let expected_i16 = ImageBuffer::<Rgba<i16>, Vec<i16>>::from_raw(2, 2, vec![
///     1, 2, 3, 10,  4,  5,  6, 20,
///     7, 8, 9, 30, 10, 11, 12, 40],
///     ).unwrap();
/// # }
/// ```
#[macro_export]
macro_rules! rgba_image {
    // Empty image with default channel type u8
    () => {
        rgba_image!(type: u8)
    };
    // Empty image with the given channel type
    (type: $channel_type:ty) => {
        {
            use image::{ImageBuffer, Rgba};
            ImageBuffer::<Rgba<$channel_type>, Vec<$channel_type>>::new(0, 0)
        }
    };
    // Non-empty image of default channel type u8
    ($( $( [$r: expr, $g: expr, $b: expr, $a:expr]),*);*) => {
        rgba_image!(type: u8, $( $( [$r, $g, $b, $a]),*);*)
    };
    // Non-empty image of given channel type
    (type: $channel_type:ty, $( $( [$r: expr, $g: expr, $b: expr, $a: expr]),*);*) => {
        {
            use image::{ImageBuffer, Rgba};
            let nested_array = [$( [ $([$r, $g, $b, $a]),*]),*];
            let height = nested_array.len() as u32;
            let width = nested_array[0].len() as u32;

            let flat_array: Vec<_> = nested_array.into_iter()
                .flat_map(|row| row.into_iter().flat_map(|p| p.into_iter()))
                .cloned()
                .collect();

            ImageBuffer::<Rgba<$channel_type>, Vec<$channel_type>>::from_raw(width, height, flat_array)
                .unwrap()
        }
    }
}

/// Human readable description of some of the pixels that differ
/// between left and right, or None if all pixels match.
pub fn pixel_diff_summary<I, J, P>(actual: &I, expected: &J) -> Option<String>
where
    P: Pixel + PartialEq,
    P::Subpixel: fmt::Debug,
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = P>,
{
    significant_pixel_diff_summary(actual, expected, |p, q| p != q)
}

/// Human readable description of some of the pixels that differ
/// signifcantly (according to provided function) between left
/// and right, or None if all pixels match.
pub fn significant_pixel_diff_summary<I, J, F, P>(
    actual: &I,
    expected: &J,
    is_significant_diff: F,
) -> Option<String>
where
    P: Pixel,
    P::Subpixel: fmt::Debug,
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = P>,
    F: Fn((u32, u32, I::Pixel), (u32, u32, J::Pixel)) -> bool,
{
    if actual.dimensions() != expected.dimensions() {
        return Some(format!(
            "dimensions do not match. \
            actual: {:?}, expected: {:?}",
            actual.dimensions(),
            expected.dimensions()
        ));
    }
    let diffs = pixel_diffs(actual, expected, is_significant_diff);
    if diffs.is_empty() {
        return None;
    }
    Some(describe_pixel_diffs(actual, expected, &diffs))
}

/// Panics if any pixels differ between the two input images.
#[macro_export]
macro_rules! assert_pixels_eq {
    ($actual:expr, $expected:expr) => ({
        assert_dimensions_match!($actual, $expected);
        match $crate::utils::pixel_diff_summary(&$actual, &$expected) {
            None => {},
            Some(err) => panic!(err)
        };
     })
}

/// Panics if any pixels differ between the two images by more than the
/// given tolerance in a single channel.
#[macro_export]
macro_rules! assert_pixels_eq_within {
    ($actual:expr, $expected:expr, $channel_tolerance:expr) => ({

        assert_dimensions_match!($actual, $expected);
        let diffs = $crate::utils::pixel_diffs(&$actual, &$expected, |p, q| {

            use image::Pixel;
            let cp = p.2.channels();
            let cq = q.2.channels();
            if cp.len() != cq.len() {
                panic!("pixels have different channel counts. \
                    actual: {:?}, expected: {:?}", cp.len(), cq.len())
            }

            let mut large_diff = false;
            for i in 0..cp.len() {
                let sp = cp[i];
                let sq = cq[i];
                // Handle unsigned subpixels
                let diff = if sp > sq {sp - sq} else {sq - sp};
                if diff > $channel_tolerance {
                    large_diff = true;
                    break;
                }
            }

            large_diff
        });
        if !diffs.is_empty() {
            panic!($crate::utils::describe_pixel_diffs(&$actual, &$expected, &diffs))
        }
    })
}

/// Panics if image dimensions do not match.
#[macro_export]
macro_rules! assert_dimensions_match {
    ($actual:expr, $expected:expr) => ({

        let actual_dim = $actual.dimensions();
        let expected_dim = $expected.dimensions();

        if actual_dim != expected_dim {
            panic!("dimensions do not match. \
                actual: {:?}, expected: {:?}", actual_dim, expected_dim)
        }
     })
}

/// Lists pixels that differ between left and right images.
pub fn pixel_diffs<I, J, F, P>(
    actual: &I,
    expected: &J,
    is_diff: F,
) -> Vec<(Diff<I::Pixel>)>
where
    P: Pixel,
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = P>,
    F: Fn((u32, u32, I::Pixel), (u32, u32, J::Pixel)) -> bool,
{
    if is_empty(actual) || is_empty(expected) {
        return vec![];
    }

    // Can't just call $image.pixels(), as that needn't hit the
    // trait pixels method - ImageBuffer defines its own pixels
    // method with a different signature
    GenericImageView::pixels(actual)
        .zip(GenericImageView::pixels(expected))
        .filter(|&(p, q)| is_diff(p, q))
        .map(|(p, q)| {
            assert!(p.0 == q.0 && p.1 == q.1, "Pixel locations do not match");
            Diff {
                x: p.0,
                y: p.1,
                actual: p.2,
                expected: q.2
        }})
        .collect::<Vec<_>>()
}

fn is_empty<I: GenericImage>(image: &I) -> bool {
    image.width() == 0 || image.height() == 0
}

/// A difference between two images
pub struct Diff<P> {
    /// x-coordinate of diff.
    pub x: u32,
    /// y-coordinate of diff.
    pub y: u32,
    /// Pixel value in expected image.
    pub expected: P,
    /// Pixel value in actual image.
    pub actual: P,
}

/// Gives a summary description of a list of pixel diffs for use in error messages.
pub fn describe_pixel_diffs<I, J, P>(actual: &I, expected: &J, diffs: &[Diff<P>]) -> String
where
    P: Pixel,
    P::Subpixel: fmt::Debug,
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = P>,
{
    let mut err = "pixels do not match.\n".to_owned();

    // Find the boundaries of the region containing diffs
    let top_left = diffs.iter().fold((u32::MAX, u32::MAX), |acc, ref d| {
        (acc.0.min(d.x), acc.1.min(d.y))
    });
    let bottom_right = diffs.iter().fold((0, 0), |acc, ref d| {
        (acc.0.max(d.x), acc.1.max(d.y))
    });

    // If all the diffs are contained in a small region of the image then render all of this
    // region, with a small margin.
    if max(bottom_right.0 - top_left.0, bottom_right.1 - top_left.1) < 6 {
        let left = max(0, top_left.0 as i32 - 2) as u32;
        let top = max(0, top_left.1 as i32 - 2) as u32;
        let right = min(actual.width() as i32 - 1, bottom_right.0 as i32 + 2) as u32;
        let bottom = min(actual.height() as i32 - 1, bottom_right.1 as i32 + 2) as u32;

        let diff_locations = diffs.iter().map(|d| (d.x, d.y)).collect::<HashSet<_>>();

        err.push_str(&colored(&"Actual:", Color::Red));
        let actual_rendered = render_image_region(
            actual,
            left,
            top,
            right,
            bottom,
            |x, y| if diff_locations.contains(&(x, y)) { Color::Red } else { Color::Cyan }
        );
        err.push_str(&actual_rendered);

        err.push_str(&colored(&"Expected:", Color::Green));
        let expected_rendered = render_image_region(
            expected,
            left,
            top,
            right,
            bottom,
            |x, y| if diff_locations.contains(&(x, y)) { Color::Green } else { Color::Cyan }
        );
        err.push_str(&expected_rendered);

        return err;
    }

    // Otherwise just list the first 5 diffs
    err.push_str(
        &(diffs
            .iter()
            .take(5)
            .map(|d| format!(
                "\nlocation: {}, actual: {}, expected: {} ",
                colored(&format!("{:?}", (d.x, d.y)), Color::Yellow),
                colored(&render_pixel(d.actual), Color::Red),
                colored(&render_pixel(d.expected), Color::Green))
            )
            .collect::<Vec<_>>()
            .join(""))
    );
    err
}

enum Color { Red, Green, Cyan, Yellow }

fn render_image_region<I, P, C>(image: &I, left: u32, top: u32, right: u32, bottom: u32, color: C) -> String
where
    P: Pixel,
    P::Subpixel: fmt::Debug,
    I: GenericImage<Pixel = P>,
    C: Fn(u32, u32) -> Color
{
    let mut rendered = String::new();

    // Render all the pixels first, so that we can determine the column width
    let mut rendered_pixels = vec![];
    for y in top..bottom + 1 {
        for x in left..right + 1 {
            let p = image.get_pixel(x, y);
            rendered_pixels.push(render_pixel(p));
        }
    }

    // Width of a column containing rendered pixels
    let pixel_column_width = rendered_pixels.iter().map(|p| p.len()).max().unwrap() + 1;
    // Maximum number of digits required to display a row or column number
    let max_digits = (max(1, max(right, bottom)) as f64).log10().ceil() as usize;
    // Each pixel column is labelled with its column number
    let pixel_column_width = pixel_column_width.max(max_digits + 1);
    let num_columns = (right - left + 1) as usize;

    // First row contains the column numbers
    write!(rendered, "\n{}", " ".repeat(max_digits + 4)).unwrap();
    for x in left..right + 1 {
        write!(rendered, "{x:>w$} ", x = x, w = pixel_column_width).unwrap();
    }

    // +--------------
    write!(rendered, "\n  {}+{}", " ".repeat(max_digits), "-".repeat((pixel_column_width + 1) * num_columns + 1)).unwrap();
    // row_number |
    write!(rendered, "\n  {y:>w$}| ", y = " ", w = max_digits).unwrap();

    let mut count = 0;
    for y in top..bottom + 1 {
        // Empty row, except for leading | separating row numbers from pixels
        write!(rendered, "\n  {y:>w$}| ", y = y, w = max_digits).unwrap();

        for x in left..right + 1 {
            // Pad pixel string to column width and right align
            let padded = format!("{c:>w$}", c = rendered_pixels[count], w = pixel_column_width);
            write!(rendered, "{} ", &colored(&padded, color(x, y))).unwrap();
            count += 1;
        }
        // Empty row, except for leading | separating row numbers from pixels
        write!(rendered, "\n  {y:>w$}| ", y = " ", w = max_digits).unwrap();
    }
    rendered.push_str("\n");
    rendered
}

fn render_pixel<P>(p: P) -> String
where
    P: Pixel,
    P::Subpixel: fmt::Debug,
{
    let cs = p.channels();
    match cs.len() {
        1 => format!("{:?}", cs[0]),
        _ => format!("[{}]", cs.iter().map(|c| format!("{:?}", c)).join(", "))
    }
}

fn colored(s: &str, c: Color) -> String {
    let escape_sequence = match c {
        Color::Red => "\x1b[31m",
        Color::Green => "\x1b[32m",
        Color::Cyan => "\x1b[36m",
        Color::Yellow => "\x1b[33m"
    };
    format!("{}{}\x1b[0m", escape_sequence, s)
}

/// Loads image at given path, panicking on failure.
pub fn load_image_or_panic<P: AsRef<Path> + fmt::Debug>(path: P) -> DynamicImage {
    open(path.as_ref()).expect(&format!("Could not load image at {:?}", path.as_ref()))
}

/// Gray image to use in benchmarks. This is neither noise nor
/// similar to natural images - it's just a convenience method
/// to produce an image that's not constant.
pub fn gray_bench_image(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let intensity = (x % 7 + y % 6) as u8;
            image.put_pixel(x, y, Luma([intensity]));
        }
    }
    image
}

/// RGB image to use in benchmarks. See comment on `gray_bench_image`.
pub fn rgb_bench_image(width: u32, height: u32) -> RgbImage {
    use std::cmp;
    let mut image = RgbImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = (x % 7 + y % 6) as u8;
            let g = 255u8 - r;
            let b = cmp::min(r, g);
            image.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_pixels_eq_passes() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        assert_pixels_eq!(image, image);
    }

    #[test]
    #[should_panic]
    fn test_assert_pixels_eq_fails() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let diff = gray_image!(
            00, 11, 02;
            10, 11, 12);

        assert_pixels_eq!(diff, image);
    }

    #[test]
    fn test_assert_pixels_eq_within_passes() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let diff = gray_image!(
            00, 02, 02;
            10, 11, 12);

        assert_pixels_eq_within!(diff, image, 1);
    }

    #[test]
    #[should_panic]
    fn test_assert_pixels_eq_within_fails() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let diff = gray_image!(
            00, 03, 02;
            10, 11, 12);

        assert_pixels_eq_within!(diff, image, 1);
    }

    #[test]
    fn test_pixel_diff_summary_handles_1x1_image() {
        let summary = pixel_diff_summary(&gray_image!(1), &gray_image!(0));
        assert_eq!(
            &summary.unwrap()[0..19],
            "pixels do not match");
    }
}
