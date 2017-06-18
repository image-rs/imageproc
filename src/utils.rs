//! Utils for testing and debugging.

use image::{DynamicImage, GenericImage, GrayImage, Luma, open, Pixel, Rgb, RgbImage};

use std::fmt;
use std::path::Path;

/// Implementation detail of the gray_image macros.
#[macro_export]
macro_rules! gray_image_from_nested_array {
    // This implementation is copied from the `matrix` macro
    // from https://github.com/AtheMathmo/rulinalg
    ($nested_array:tt, $channel_type:ty) => {
        {
            use image::{ImageBuffer, Luma};
            let height = $nested_array.len() as u32;
            let width = $nested_array[0].len() as u32;

            let flat_array: Vec<_> = $nested_array.into_iter()
                .flat_map(|row| row.into_iter())
                .cloned()
                .collect();

            ImageBuffer::<Luma<$channel_type>, Vec<$channel_type>>::from_raw(width, height, flat_array)
                .unwrap()
        }
    }
}

/// Helper for defining greyscale images with u8 subpixels. Columns are separated
/// by commas and rows by semi-colons.
///
/// Calls `ImageBuffer::from_raw`.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let equivalent = GrayImage::from_raw(3, 2, vec![
///     1, 2, 3,
///     4, 5, 6
/// ]).unwrap();
///
/// assert_pixels_eq!(image, equivalent);
/// # }
/// ```
#[macro_export]
macro_rules! gray_image {
    () => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<u8>, Vec<u8>>::new(0, 0)
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            gray_image_from_nested_array!(data_as_nested_array, u8)
        }
    }
}

/// Helper for defining greyscale images with i16 subpixels. Columns are separated
/// by commas and rows by semi-colons.
///
/// See the [`gray_image`](macro.gray_image.html) documentation for examples.
#[macro_export]
macro_rules! gray_image_i16 {
    () => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<i16>, Vec<i16>>::new(0, 0)
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            gray_image_from_nested_array!(data_as_nested_array, i16)
        }
    }
}

/// Helper for defining greyscale images with u16 subpixels. Columns are separated
/// by commas and rows by semi-colons.
///
/// See the [`gray_image`](macro.gray_image.html) documentation for examples.
#[macro_export]
macro_rules! gray_image_u16 {
    () => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<u16>, Vec<u16>>::new(0, 0)
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            gray_image_from_nested_array!(data_as_nested_array, u16)
        }
    }
}

/// Helper for defining greyscale images with i32 subpixels. Columns are separated
/// by commas and rows by semi-colons.
///
/// See the [`gray_image`](macro.gray_image.html) documentation for examples.
#[macro_export]
macro_rules! gray_image_i32 {
    () => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<i32>, Vec<i32>>::new(0, 0)
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            gray_image_from_nested_array!(data_as_nested_array, i32)
        }
    }
}

/// Helper for defining greyscale images with u32 subpixels. Columns are separated
/// by commas and rows by semi-colons.
///
/// See the [`gray_image`](macro.gray_image.html) documentation for examples.
#[macro_export]
macro_rules! gray_image_u32 {
    () => {
        {
            use image::{ImageBuffer, Luma};
            ImageBuffer::<Luma<u32>, Vec<u32>>::new(0, 0)
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            gray_image_from_nested_array!(data_as_nested_array, u32)
        }
    }
}

/// Implementation detail of the rgb_image macro.
#[macro_export]
macro_rules! rgb_image_from_nested_array {
    ($nested_array:tt, $channel_type:ty) => {
        {
            use image::{ImageBuffer, Rgb};
            let height = $nested_array.len() as u32;
            let width = $nested_array[0].len() as u32;

            let flat_array: Vec<_> = $nested_array.into_iter()
                .flat_map(|row| row.into_iter().flat_map(|p| p.into_iter()))
                .cloned()
                .collect();

            ImageBuffer::<Rgb<$channel_type>, Vec<$channel_type>>::from_raw(width, height, flat_array)
                .unwrap()
        }
    }
}

/// Helper for defining RGB images with u8 subpixels. Pixels are delineated by square
/// brackets, columns are separated by commas and rows are separated by semi-colons.
///
/// Calls `ImageBuffer::from_raw`.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::RgbImage;
///
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
/// # }
/// ```
#[macro_export]
macro_rules! rgb_image {
    () => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<u8>, Vec<u8>>::new(0, 0)
        }
    };
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            rgb_image_from_nested_array!(nested_array, u8)
        }
    }
}


/// Helper for defining RGB images with i16 subpixels. Pixels are delineated by square
/// brackets, columns are separated by commas and rows are separated by semi-colons.
///
/// See the [`rgb_image`](macro.rgb_image.html) documentation for examples.
#[macro_export]
macro_rules! rgb_image_i16 {
    () => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<i16>, Vec<i16>>::new(0, 0)
        }
    };
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            rgb_image_from_nested_array!(nested_array, i16)
        }
    }
}

/// Helper for defining RGB images with u16 subpixels. Pixels are delineated by square
/// brackets, columns are separated by commas and rows are separated by semi-colons.
///
/// See the [`rgb_image`](macro.rgb_image.html) documentation for examples.
#[macro_export]
macro_rules! rgb_image_u16 {
    () => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<u16>, Vec<u16>>::new(0, 0)
        }
    };
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            rgb_image_from_nested_array!(nested_array, u16)
        }
    }
}

/// Helper for defining RGB images with i32 subpixels. Pixels are delineated by square
/// brackets, columns are separated by commas and rows are separated by semi-colons.
///
/// See the [`rgb_image`](macro.rgb_image.html) documentation for examples.
#[macro_export]
macro_rules! rgb_image_i32 {
    () => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<i32>, Vec<i32>>::new(0, 0)
        }
    };
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            rgb_image_from_nested_array!(nested_array, i32)
        }
    }
}

/// Helper for defining RGB images with u32 subpixels. Pixels are delineated by square
/// brackets, columns are separated by commas and rows are separated by semi-colons.
///
/// See the [`rgb_image`](macro.rgb_image.html) documentation for examples.
#[macro_export]
macro_rules! rgb_image_u32 {
    () => {
        {
            use image::{ImageBuffer, Rgb};
            ImageBuffer::<Rgb<u32>, Vec<u32>>::new(0, 0)
        }
    };
    ($( $( [$r: expr, $g: expr, $b: expr]),*);*) => {
        {
            let nested_array = [$( [ $([$r, $g, $b]),*]),*];
            rgb_image_from_nested_array!(nested_array, u32)
        }
    }
}

/// Human readable description of some of the pixels that differ
/// between left and right, or None if all pixels match.
pub fn pixel_diff_summary<I, J, P>(actual: &I, expected: &J) -> Option<String>
where
    P: Pixel + PartialEq + fmt::Debug,
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
    P: Pixel + fmt::Debug,
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
    Some(describe_pixel_diffs(diffs.into_iter()))
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
            panic!($crate::utils::describe_pixel_diffs(diffs.into_iter()))
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
    left: &I,
    right: &J,
    is_diff: F,
) -> Vec<((u32, u32, I::Pixel), (u32, u32, I::Pixel))>
where
    P: Pixel,
    I: GenericImage<Pixel = P>,
    J: GenericImage<Pixel = P>,
    F: Fn((u32, u32, I::Pixel), (u32, u32, J::Pixel)) -> bool,
{
    if is_empty(left) || is_empty(right) {
        return vec![];
    }

    // Can't just call $image.pixels(), as that needn't hit the
    // trait pixels method - ImageBuffer defines its own pixels
    // method with a different signature
    GenericImage::pixels(left)
        .zip(GenericImage::pixels(right))
        .filter(|&(p, q)| is_diff(p, q))
        .collect::<Vec<_>>()
}

fn is_empty<I: GenericImage>(image: &I) -> bool {
    image.width() == 0 || image.height() == 0
}

/// Gives a summary description of a list of pixel diffs for use in error messages.
pub fn describe_pixel_diffs<I, P>(diffs: I) -> String
where
    I: Iterator<Item = (P, P)>,
    P: fmt::Debug,
{
    let mut err = "pixels do not match. ".to_owned();
    err.push_str(
        &(diffs
              .take(5)
              .map(|d| format!("\nactual: {:?}, expected {:?} ", d.0, d.1))
              .collect::<Vec<_>>()
              .join("")),
    );
    err
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
mod test {
    use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

    #[test]
    fn test_gray_image_empty() {
        let image = gray_image!();
        assert_eq!(image.dimensions(), (0, 0));
    }

    #[test]
    fn test_gray_image_single_element() {
        let image = gray_image!(1);
        let expected = GrayImage::from_raw(1, 1, vec![1]).unwrap();
        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_single_row() {
        let image = gray_image!(1, 2, 3);
        let expected = GrayImage::from_raw(3, 1, vec![1, 2, 3]).unwrap();
        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_multiple_rows_and_columns() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6);

        let expected = GrayImage::from_raw(3, 2, vec![
            1, 2, 3,
            4, 5, 6]).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_i16() {
        let image = gray_image_i16!(
            1, 2, 3;
            4, 5, 6);

        let expected = ImageBuffer::<Luma<i16>, Vec<i16>>::from_raw(3, 2, vec![
            1i16, 2, 3,
            4, 5, 6]).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_u16() {
        let image = gray_image_u16!(
            1, 2, 3;
            4, 5, 6);

        let expected = ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(3, 2, vec![
            1u16, 2, 3,
            4, 5, 6]).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_i32() {
        let image = gray_image_i32!(
            1, 2, 3;
            4, 5, 6);

        let expected = ImageBuffer::<Luma<i32>, Vec<i32>>::from_raw(3, 2, vec![
            1i32, 2, 3,
            4, 5, 6]).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_gray_image_u32() {
        let image = gray_image_u32!(
            1, 2, 3;
            4, 5, 6);

        let expected = ImageBuffer::<Luma<u32>, Vec<u32>>::from_raw(3, 2, vec![
            1u32, 2, 3,
            4, 5, 6]).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_empty() {
        let image = rgb_image!();
        assert_eq!(image.dimensions(), (0, 0));
    }

    #[test]
    fn test_rgb_image_single_element() {
        let image = rgb_image!([1, 2, 3]);
        let expected = RgbImage::from_raw(1, 1, vec![1, 2, 3]).unwrap();
        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_single_row() {
        let image = rgb_image!([1, 2, 3], [4, 5, 6]);
        let expected = RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_i16() {
        let image = rgb_image_i16!(
            [1, 2, 3], [4, 5, 6];
            [7, 8, 9], [10, 11, 12]);

        let expected = ImageBuffer::<Rgb<i16>, Vec<i16>>::from_raw(2, 2, vec![
            1i16, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12],
        ).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_u16() {
        let image = rgb_image_u16!(
            [1, 2, 3], [4, 5, 6];
            [7, 8, 9], [10, 11, 12]);

        let expected = ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(2, 2, vec![
            1u16, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12],
        ).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_i32() {
        let image = rgb_image_i32!(
            [1, 2, 3], [4, 5, 6];
            [7, 8, 9], [10, 11, 12]);

        let expected = ImageBuffer::<Rgb<i32>, Vec<i32>>::from_raw(2, 2, vec![
            1i32, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12],
        ).unwrap();

        assert_pixels_eq!(image, expected);
    }

    #[test]
    fn test_rgb_image_u32() {
        let image = rgb_image_u32!(
            [1, 2, 3], [4, 5, 6];
            [7, 8, 9], [10, 11, 12]);

        let expected = ImageBuffer::<Rgb<u32>, Vec<u32>>::from_raw(2, 2, vec![
            1u32, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12],
        ).unwrap();

        assert_pixels_eq!(image, expected);
    }

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
}
