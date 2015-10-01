//! I is the integral image of an image F if I(x, y) is the
//! sum of F(x', y') for x' <= x, y' <= y. i.e. each pixel
//! in the integral image contains the sum of the pixel intensities
//! of all input pixels that are above it and to its left.
//! The integral image has the helpful property that it lets us
//! compute the sum of pixel intensities from any rectangular region
//! in the input image in constant time.
//! Specifically, given a rectangle in F with clockwise corners
//! A, B, C, D, with A at the upper left, the total pixel intensity
//! of this rectangle is I(C) - I(B) - I(D) + I(A).

extern crate image;

use image::{
    Luma,
    GenericImage,
    ImageBuffer
};

/// Computes the integral image of an 8bpp grayscale image.
// TODO: Support more formats.
// TODO: This is extremely slow. Fix that!
pub fn integral_image<I: GenericImage<Pixel=Luma<u8>> + 'static>(image: &I)
    -> ImageBuffer<Luma<u32>, Vec<u32>> {
    padded_integral_image(image, 0, 0)
}

/// Computes the integral image of the result of padding image
/// with its boundary pixels for x_padding columns on either
/// side and y_padding rows at its top and bottom.
/// Returned image has width image.width() + 2 * x_padding
/// and height image.height() + 2 * y_padding.
pub fn padded_integral_image<I: GenericImage<Pixel=Luma<u8>> + 'static>(
    image: &I,
    x_padding: u32,
    y_padding: u32)
    -> ImageBuffer<Luma<u32>, Vec<u32>> {

    let (in_width, in_height) = image.dimensions();
    let out_width = in_width + 2 * x_padding;
    let out_height = in_height + 2 * y_padding;

    let mut out: ImageBuffer<Luma<u32>, Vec<u32>>
        = ImageBuffer::new(out_width, out_height);

    for y in 0..out_height {
        for x in 0..out_width {

            let y_in: u32;
            if y < y_padding {
                y_in = 0;
            }
            else if y >= in_height + y_padding {
                y_in = in_height - 1;
            }
            else {
                y_in = y - y_padding;
            }
            let x_in: u32;
            if x < x_padding {
                x_in = 0;
            }
            else if x >= in_width + x_padding {
                x_in = in_width - 1;
            }
            else {
                x_in = x - x_padding;
            }

            let p = image.get_pixel(x_in, y_in);
            out.put_pixel(x, y, Luma([p[0] as u32]));
        }
    }

    for x in 1..out_width {
        (*out.get_pixel_mut(x, 0))[0] += out.get_pixel(x - 1, 0)[0];
    }

    for y in 1..out_height {
        (*out.get_pixel_mut(0, y))[0] += out.get_pixel(0, y - 1)[0];

        for x in 1..out_width {
            (*out.get_pixel_mut(x, y))[0] += out.get_pixel(x, y - 1)[0];
            (*out.get_pixel_mut(x, y))[0] += out.get_pixel(x - 1, y)[0];
            (*out.get_pixel_mut(x, y))[0] -= out.get_pixel(x - 1, y - 1)[0];
        }
    }

    out
}

/// Computes the running sum of one row of image, padded
/// at the beginning and end. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all rows in an image.
// TODO: faster, more formats
pub fn row_running_sum<I: GenericImage<Pixel=Luma<u8>> + 'static>(
    image: &I, row: u32, buffer: &mut [u32], padding: u32) {

    let width = image.width();
    assert!(buffer.len() >= (width + 2 * padding) as usize,
        format!("Buffer length {} is less than 2 * {} + {}",
            buffer.len(), width, padding));

    for x in 0..padding {
        buffer[x as usize] = image.get_pixel(0, row)[0] as u32;
    }

    for x in 0..width {
        let idx = (x + padding) as usize;
        buffer[idx] = image.get_pixel(x, row)[0] as u32;
    }

    for x in 0..padding {
        let idx = (x + width + padding) as usize;
        buffer[idx] = image.get_pixel(width - 1, row)[0] as u32;
    }

    for x in 1..width + 2 * padding {
        buffer[x as usize] += buffer[(x - 1) as usize] as u32;
    }
}

#[cfg(test)]
mod test {

    use super::{
        integral_image,
        padded_integral_image,
        row_running_sum
    };
    use utils::{
        gray_bench_image
    };
    use image::{
        GrayImage,
        ImageBuffer,
        Luma
    };
    use test;

    #[test]
    fn test_integral_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            1u8, 2u8, 3u8,
            4u8, 5u8, 6u8]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(3, 2, vec![
            1u32,  3u32,  6u32,
            5u32, 12u32, 21u32]).unwrap();

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
            1u8, 2u8, 3u8,
            4u8, 5u8, 6u8]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(5, 4, vec![
              1u32,  2u32,   4u32,   7u32,  10u32,
              2u32,  4u32,   8u32,  14u32,  20u32,
              6u32, 12u32,  21u32,  33u32,  45u32,
             10u32, 20u32,  34u32,  52u32,  70u32]).unwrap();

        assert_pixels_eq!(padded_integral_image(&image, 1, 1), expected);
    }

    #[test]
    fn test_row_running_sum() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            1u8, 2u8, 3u8,
            4u8, 5u8, 6u8]).unwrap();

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
}
