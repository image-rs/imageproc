//! Functions for computing [integral images](https://en.wikipedia.org/wiki/Summed_area_table)
//! and running sums of rows and columns.

extern crate image;

use image::{
    Luma,
    GrayImage,
    GenericImage,
    ImageBuffer
};

use definitions::Image;

/// Compute the 2d running sum of a grayscale image.
///
/// An integral image I has width and height one greater than its source image F,
/// and is defined by I(x, y) = sum of F(x', y') for x' < x, y' < y, i.e. each pixel
/// in the integral image contains the sum of the pixel intensities of all input pixels
/// that are strictly above it and strictly to its left. In particular, the left column
/// and top row of an integral image are all 0, and the value of the bottom right pixel of
/// an integral image is equal to the sum of all pixels in the source image.
///
/// Integral images have the helpful property of allowing us to
/// compute the sum of pixel intensities in a rectangular region of an image
/// in constant time. Specifically, given a rectangle [l, r] * [t, b] in F,
/// the sum of the pixels in this rectangle is
/// I(r + 1, b + 1) - I(r + 1, t) - I(l, b + 1) + I(l, t).
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::integralimage::{integral_image, sum_image_pixels};
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let integral = gray_image_u32!(
///     0,  0,  0,  0;
///     0,  1,  3,  6;
///     0,  5, 12, 21);
///
/// assert_pixels_eq!(integral_image(&image), integral);
///
/// // Compute the sum of all pixels in the right two columns
/// assert_eq!(sum_image_pixels(&integral, 1, 0, 2, 1), 2 + 3 + 5 + 6);
///
/// // Compute the sum of all pixels in the top row
/// assert_eq!(sum_image_pixels(&integral, 0, 0, 2, 0), 1 + 2 + 3);
/// # }
/// ```
pub fn integral_image(image: &GrayImage) -> Image<Luma<u32>> {
    // TODO: Support more formats, make faster, add a new IntegralImage type
    // TODO: to make it harder to make off-by-one errors when computing sums of regions.
    let (in_width, in_height) = image.dimensions();
    let out_width = in_width + 1;
    let out_height = in_height + 1;

    let mut out = ImageBuffer::from_pixel(out_width, out_height, Luma([0u32]));

    if in_width == 0 || in_height == 0 {
        return out;
    }

    for y in 1..out_height {
        let mut sum = 0;
        for x in 1..out_width {
            unsafe {
                sum += image.unsafe_get_pixel(x - 1, y - 1)[0] as u32;
                let above = out.unsafe_get_pixel(x, y - 1)[0];
                out.unsafe_put_pixel(x, y, Luma([above + sum]))
            }
        }
    }

    out
}

/// Sums the pixels in positions [left, right] * [top, bottom] in F, where `integral_image` is the
/// integral image of F.
///
/// See the [`integral_image`](fn.integral_image.html) documentation for examples.
pub fn sum_image_pixels(integral_image: &Image<Luma<u32>>, left: u32, top: u32, right: u32, bottom: u32) -> u32 {
    // TODO: better type-safety. It's too easy to pass the original image in here by mistake.
    let sum = integral_image.get_pixel(right + 1, bottom + 1)[0] as i32
            - integral_image.get_pixel(right + 1, top)[0] as i32
            - integral_image.get_pixel(left, bottom + 1)[0] as i32
            + integral_image.get_pixel(left, top)[0] as i32;
    sum as u32
}

/// Computes the running sum of one row of image, padded
/// at the beginning and end. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all rows in an image.
pub fn row_running_sum(image: &GrayImage, row: u32, buffer: &mut [u32], padding: u32) {
    // TODO: faster, more formats
    let (width, height) = image.dimensions();
    assert!(buffer.len() >= (width + 2 * padding) as usize,
        format!("Buffer length {} is less than {} + 2 * {}", buffer.len(), width, padding));
    assert!(row < height, format!("row out of bounds: {} >= {}", row, height));

    unsafe {
        let mut sum = 0;
        for x in 0..padding {
            sum += image.unsafe_get_pixel(0, row)[0] as u32;
            *buffer.get_unchecked_mut(x as usize) = sum;
        }

        for x in 0..width {
            sum += image.unsafe_get_pixel(x, row)[0] as u32;
            *buffer.get_unchecked_mut((x + padding) as usize) = sum;
        }

        for x in 0..padding {
            sum += image.unsafe_get_pixel(width - 1, row)[0] as u32;
            *buffer.get_unchecked_mut((x + width + padding) as usize) = sum;
        }
    }
}

/// Computes the running sum of one column of image, padded
/// at the top and bottom. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all columns in an image.
// TODO: faster, more formats
pub fn column_running_sum(image: &GrayImage, column: u32, buffer: &mut [u32], padding: u32) {

    let (width, height) = image.dimensions();
    assert!(buffer.len() >= (height + 2 * padding) as usize,
        format!("Buffer length {} is less than {} + 2 * {}", buffer.len(), height, padding));
    assert!(column < width, format!("column out of bounds: {} >= {}", column, width));

    unsafe {
        let mut sum = 0;
        for y in 0..padding {
            sum += image.unsafe_get_pixel(column, 0)[0] as u32;
            *buffer.get_unchecked_mut(y as usize) = sum;
        }

        for y in 0..height {
            sum += image.unsafe_get_pixel(column, y)[0] as u32;
            *buffer.get_unchecked_mut((y + padding) as usize) = sum;
        }

        for y in 0..padding {
            sum += image.unsafe_get_pixel(column, height - 1)[0] as u32;
            *buffer.get_unchecked_mut((y + height + padding) as usize) = sum;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::{
        gray_bench_image,
        GrayTestImage,
        pixel_diff_summary
    };
    use image::{
        GenericImage,
        ImageBuffer,
        Luma
    };
    use quickcheck::{
        quickcheck,
        TestResult
    };
    use definitions::{
        Image
    };
    use test;

    #[test]
    fn test_sum_image_pixels() {
        let image = gray_image!(
            1, 2;
            3, 4);

        let integral = ::integralimage::integral_image(&image);

        assert_eq!(sum_image_pixels(&integral, 0, 0, 0, 0), 1);
        assert_eq!(sum_image_pixels(&integral, 0, 0, 1, 0), 3);
        assert_eq!(sum_image_pixels(&integral, 0, 0, 0, 1), 4);
        assert_eq!(sum_image_pixels(&integral, 0, 0, 1, 1), 10);
        assert_eq!(sum_image_pixels(&integral, 1, 0, 1, 0), 2);
        assert_eq!(sum_image_pixels(&integral, 1, 0, 1, 1), 6);
        assert_eq!(sum_image_pixels(&integral, 0, 1, 0, 1), 3);
        assert_eq!(sum_image_pixels(&integral, 0, 1, 1, 1), 7);
        assert_eq!(sum_image_pixels(&integral, 1, 1, 1, 1), 4);
    }

    #[test]
    fn test_integral_image() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6);

        let expected = gray_image_u32!(
            0,  0,  0,  0;
            0,  1,  3,  6;
            0,  5, 12, 21);

        assert_pixels_eq!(integral_image(&image), expected);
    }

    #[bench]
    fn bench_integral_image(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let integral = integral_image(&image);
            test::black_box(integral);
            });
    }

    /// Simple implementation of integral_image to validate faster versions against.
    fn integral_image_ref<I>(image: &I) -> Image<Luma<u32>>
        where I: GenericImage<Pixel=Luma<u8>>
    {
        let (in_width, in_height) = image.dimensions();
        let (out_width, out_height) = (in_width + 1, in_height + 1);
        let mut out = ImageBuffer::from_pixel(out_width, out_height, Luma([0u32]));

        for y in 1..out_height {
            for x in 0..out_width {
                let mut sum = 0u32;

                for iy in 0..y {
                    for ix in 0..x {
                        sum += image.get_pixel(ix, iy)[0] as u32;
                    }
                }

                out.put_pixel(x, y, Luma([sum]));
            }
        }

        out
    }

    #[test]
    fn test_integral_image_matches_reference_implementation() {
        fn prop(image: GrayTestImage) -> TestResult {
            let expected = integral_image_ref(&image.0);
            let actual = integral_image(&image.0);
            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err)
            }
        }
        quickcheck(prop as fn(GrayTestImage) -> TestResult);
    }

    #[test]
    fn test_row_running_sum() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6);

        let expected = [1, 2, 4, 7, 10];

        let mut buffer = [0; 5];
        row_running_sum(&image, 0, &mut buffer, 1);

        assert_eq!(buffer, expected);
    }

    #[bench]
    fn bench_row_running_sum(b: &mut test::Bencher) {
        let image = gray_bench_image(1000, 1);
        let mut buffer = [0; 1010];
        b.iter(|| {
            row_running_sum(&image, 0, &mut buffer, 5);
            });
    }

    #[test]
    fn test_column_running_sum() {
        let image = gray_image!(
            1, 4;
            2, 5;
            3, 6);

        let expected = [1, 2, 4, 7, 10];

        let mut buffer = [0; 5];
        column_running_sum(&image, 0, &mut buffer, 1);

        assert_eq!(buffer, expected);
    }

    #[bench]
    fn bench_column_running_sum(b: &mut test::Bencher) {
        let image = gray_bench_image(100, 1000);
        let mut buffer = [0; 1010];
        b.iter(|| {
            column_running_sum(&image, 0, &mut buffer, 5);
            });
    }
}
