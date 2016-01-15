//! Functions for computing [integral images](https://en.wikipedia.org/wiki/Summed_area_table)
//! and running sums of rows and columns.

extern crate image;

use image::{
    Luma,
    GenericImage,
    ImageBuffer
};

use definitions::{
    VecBuffer
};

use std::cmp;

/// Computes the integral image of an 8bpp grayscale image.
/// I is the integral image of an image F if I(x, y) is the
/// sum of F(x', y') for x' <= x, y' <= y. i.e. each pixel
/// in the integral image contains the sum of the pixel intensities
/// of all input pixels that are above it and to its left.
/// The integral image has the helpful property that it lets us
/// compute the sum of pixel intensities from any rectangular region
/// in the input image in constant time.
/// Specifically, given a rectangle in F with clockwise corners
/// A, B, C, D, with A at the upper left, the total pixel intensity
/// of this rectangle is I(C) - I(B) - I(D) + I(A).
// TODO: Support more formats.
// TODO: This is extremely slow. Fix that!
pub fn integral_image<I>(image: &I) -> VecBuffer<Luma<u32>>
    where I: GenericImage<Pixel=Luma<u8>> {
    padded_integral_image(image, 0, 0)
}

/// Computes the integral image of the result of padding image with
/// its boundary pixels for x_padding columns on either side and y_padding
/// rows at its top and bottom. Returned image has width image.width() + 2 * x_padding
/// and height image.height() + 2 * y_padding.
pub fn padded_integral_image<I>(image: &I, x_padding: u32, y_padding: u32) -> VecBuffer<Luma<u32>>
    where I: GenericImage<Pixel=Luma<u8>>
{
    let (in_width, in_height) = image.dimensions();
    let out_width = in_width + 2 * x_padding;
    let out_height = in_height + 2 * y_padding;

    let mut out: ImageBuffer<Luma<u32>, Vec<u32>> = ImageBuffer::new(out_width, out_height);

    if in_width == 0 || in_height == 0 {
        return out;
    }

    let mut top_sum = 0u32;
    for x in 0..out_width {
        let x_in = adjust_for_padding(x, x_padding, in_width);
        unsafe {
            top_sum += image.unsafe_get_pixel(x_in, 0)[0] as u32;
            out.unsafe_put_pixel(x, 0, Luma([top_sum]));
        }
    }

    for y in 1..out_height {
        let y_in = adjust_for_padding(y, y_padding, in_height);
        let mut sum = 0;

        for x in 0..out_width {
            let x_in = adjust_for_padding(x, x_padding, in_width);

            unsafe {
                sum += image.unsafe_get_pixel(x_in, y_in)[0] as u32;
                let above = out.unsafe_get_pixel(x, y - 1)[0];
                out.unsafe_put_pixel(x, y, Luma([above + sum]))
            }
        }
    }

    out
}

#[inline(always)]
fn adjust_for_padding(out_index: u32, padding: u32, upper: u32) -> u32 {
    if out_index < padding {
        return 0;
    } else if out_index >= upper + padding {
        return upper - 1;
    }
    return out_index - padding;
}

/// Computes the running sum of one row of image, padded
/// at the beginning and end. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all rows in an image.
// TODO: faster, more formats
pub fn row_running_sum<I>(image: &I, row: u32, buffer: &mut [u32], padding: u32)
    where I: GenericImage<Pixel=Luma<u8>> {

    let width = image.width();
    assert!(buffer.len() >= (width + 2 * padding) as usize,
        format!("Buffer length {} is less than {} + 2 * {}", buffer.len(), width, padding));
    assert!(row < image.height(),
        format!("row out of bound: {} >= {}", row, image.height()));

    for x in 0..padding {
        buffer[x as usize] = unsafe { image.unsafe_get_pixel(0, row)[0] as u32 };
    }

    for x in 0..width {
        let idx = (x + padding) as usize;
        buffer[idx] = unsafe { image.unsafe_get_pixel(x, row)[0] as u32 };
    }

    for x in 0..padding {
        let idx = (x + width + padding) as usize;
        buffer[idx] = unsafe { image.unsafe_get_pixel(width - 1, row)[0] as u32 };
    }

    for x in 1..width + 2 * padding {
        buffer[x as usize] += buffer[(x - 1) as usize] as u32;
    }
}

/// Computes the running sum of one column of image, padded
/// at the top and bottom. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all columns in an image.
// TODO: faster, more formats
pub fn column_running_sum<I>(image: &I, column: u32, buffer: &mut [u32], padding: u32)
    where I: GenericImage<Pixel=Luma<u8>> {

    let height = image.height();
    assert!(buffer.len() >= (height + 2 * padding) as usize,
        format!("Buffer length {} is less than {} + 2 * {}", buffer.len(), height, padding));
    assert!(column < image.width(),
        format!("column out of bound: {} >= {}", column, image.width()));

    for y in 0..padding {
        buffer[y as usize] = unsafe { image.unsafe_get_pixel(column, 0)[0] as u32 };
    }

    for y in 0..height {
        let idx = (y + padding) as usize;
        buffer[idx] = unsafe { image.unsafe_get_pixel(column, y)[0] as u32 };
    }

    for y in 0..padding {
        let idx = (y + height + padding) as usize;
        buffer[idx] = unsafe { image.unsafe_get_pixel(column, height - 1)[0] as u32 };
    }

    for y in 1..height + 2 * padding {
        buffer[y as usize] += buffer[(y - 1) as usize] as u32;
    }
}

#[cfg(test)]
mod test {

    use super::{
        column_running_sum,
        integral_image,
        padded_integral_image,
        row_running_sum
    };
    use utils::{
        gray_bench_image,
        GrayTestImage
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma
    };
    use quickcheck::{
        quickcheck
    };
    use definitions::{
        VecBuffer
    };
    use test;

    #[test]
    fn test_integral_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            1, 2, 3,
            4, 5, 6]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(3, 2, vec![
            1,  3,  6,
            5, 12, 21]).unwrap();

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

    #[test]
    fn test_padded_integral_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            1, 2, 3,
            4, 5, 6]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(5, 4, vec![
              1,  2,   4,   7,  10,
              2,  4,   8,  14,  20,
              6, 12,  21,  33,  45,
             10, 20,  34,  52,  70]).unwrap();

        assert_pixels_eq!(padded_integral_image(&image, 1, 1), expected);
    }

    #[test]
    fn test_row_running_sum() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            1, 2, 3,
            4, 5, 6]).unwrap();

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
        let image: GrayImage = ImageBuffer::from_raw(2, 3, vec![
            1, 4,
            2, 5,
            3, 6]).unwrap();

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

    /// Simple implementation of integral_image to validate faster versions against.
    fn integral_image_ref<I>(image: &I) -> VecBuffer<Luma<u32>>
        where I: GenericImage<Pixel=Luma<u8>>
    {
        let (width, height) = image.dimensions();
        let mut out = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;

                for iy in 0..(y + 1) {
                    for ix in 0..(x + 1) {
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
        fn prop(image: GrayTestImage) -> bool {
            let expected = integral_image_ref(&image.0);
            let actual = integral_image(&image.0);
            actual.pixels().eq(expected.pixels())
        }
        quickcheck(prop as fn(GrayTestImage) -> bool);
    }
}
