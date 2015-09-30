
extern crate image;

use image::{
    Luma,
    GenericImage,
    ImageBuffer
};

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
// TODO: Support more than just 8bpp grayscale.
// TODO: This is extremely slow. Fix that!
pub fn integral_image<I: GenericImage<Pixel=Luma<u8>> + 'static>(image: &I)
    -> ImageBuffer<Luma<u32>, Vec<u32>> {

    let (width, height) = image.dimensions();
    let mut out: ImageBuffer<Luma<u32>, Vec<u32>>
        = ImageBuffer::new(width, height);

    // TODO: use or write a better copying operation
    for y in 0..height {
        for x in 0..width {
            let p = image.get_pixel(x, y);
            out.put_pixel(x, y, Luma([p[0] as u32]));
        }
    }

    for x in 1..width {
        (*out.get_pixel_mut(x, 0))[0] += out.get_pixel(x - 1, 0)[0];
    }

    for y in 1..height {
        (*out.get_pixel_mut(0, y))[0] += out.get_pixel(0, y - 1)[0];

        for x in 1..width {
            (*out.get_pixel_mut(x, y))[0] += out.get_pixel(x, y - 1)[0];
            (*out.get_pixel_mut(x, y))[0] += out.get_pixel(x - 1, y)[0];
            (*out.get_pixel_mut(x, y))[0] -= out.get_pixel(x - 1, y - 1)[0];
        }
    }

    out
}

#[cfg(test)]
mod test {

    use super::{
        integral_image
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
}
