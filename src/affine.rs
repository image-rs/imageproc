//! Functions for affine transformations of images.

use image::{
    Pixel,
    GenericImage,
    ImageBuffer
};

/// Rotate an image clockwise about center by theta radians by choosing
/// the nearest source pixel to the pre-image of a given output pixel.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to default.
pub fn rotate_nearest<I: GenericImage + 'static>(
    image: &I,
    center: (f32, f32),
    theta: f32,
    default: I::Pixel)
    -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let center_x  = center.0;
    let center_y  = center.1;

    for y in (0..height) {
        let dy = y as f32 - center_y;
        let mut px = center_x + sin_theta * dy - cos_theta * center_x;
        let mut py = center_y + cos_theta * dy + sin_theta * center_x;

        for x in (0..width) {

            let rx = px.round();
            let ry = py.round();

            let x_out_of_bounds = rx < 0f32 || rx >= width as f32;
            let y_out_of_bounds = ry < 0f32 || ry >= height as f32;

            if x_out_of_bounds || y_out_of_bounds {
                out.put_pixel(x, y, default);
            }
            else {
                let source = image.get_pixel(rx as u32, ry as u32);
                out.put_pixel(x, y, source);
            }

            px += cos_theta;
            py -= sin_theta;
        }
    }

    out
}

/// Translates the input image by t. Note that image coordinates increase from
/// top left to bottom right. Output pixels whose pre-image are not in the input
/// image are set to the boundary pixel in the input image nearest to their pre-image.
pub fn translate<I: GenericImage + 'static>(image: &I, t: (i32, i32))
    -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

    use std::cmp;

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let w = width as i32;
    let h = height as i32;

    for y in 0..out.height() {
        for x in 0..out.width() {
            let x_in = cmp::max(0, cmp::min(x as i32 - t.0, w - 1));
            let y_in = cmp::max(0, cmp::min(y as i32 - t.1, h - 1));
            let p = image.get_pixel(x_in as u32, y_in as u32);
            out.put_pixel(x, y, p);
        }
    }

    out
}

#[cfg(test)]
mod test {

    use super::{
        rotate_nearest,
        translate
    };
    use utils::{
        gray_bench_image
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma
    };
    use test;

    #[test]
    fn test_rotate_nearest_zero_radians() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8]).unwrap();

        let rotated = rotate_nearest(&image, (1f32, 0f32), 0f32, Luma([99u8]));

        assert_pixels_eq!(rotated, image);
    }

    #[test]
    fn text_rotate_nearest_quarter_turn_clockwise() {
        use std::f32;

        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            11u8, 01u8, 99u8,
            12u8, 02u8, 99u8]).unwrap();

        let rotated
            = rotate_nearest(&image, (1f32, 0f32), f32::consts::PI / 2f32, Luma([99u8]));

        assert_pixels_eq!(rotated, expected);
    }

    #[test]
    fn text_rotate_nearest_half_turn_anticlockwise() {
        use std::f32;

        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            12u8, 11u8, 10u8,
            02u8, 01u8, 00u8]).unwrap();

        let rotated
            = rotate_nearest(&image, (1f32, 0.5f32), -f32::consts::PI, Luma([99u8]));

        assert_pixels_eq!(rotated, expected);
    }

    #[bench]
    fn bench_rotate_nearest(b: &mut test::Bencher) {
        let mut image: GrayImage = ImageBuffer::new(200, 200);
        for pix in image.pixels_mut() {
            *pix = Luma([15u8]);
        }

        b.iter(|| {
            let rotated = rotate_nearest(&image, (3f32, 3f32), 1f32, Luma([0u8]));
            test::black_box(rotated);
            });
    }

    #[test]
    fn test_translate_positive_x_positive_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8,
            20u8, 21u8, 22u8,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00u8, 00u8, 01u8,
            00u8, 00u8, 01u8,
            10u8, 10u8, 11u8,]).unwrap();

        let translated = translate(&image, (1, 1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_positive_x_negative_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8,
            20u8, 21u8, 22u8,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            10u8, 10u8, 11u8,
            20u8, 20u8, 21u8,
            20u8, 20u8, 21u8,]).unwrap();

        let translated = translate(&image, (1, -1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_large_x_large_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00u8, 01u8, 02u8,
            10u8, 11u8, 12u8,
            20u8, 21u8, 22u8,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00u8, 00u8, 00u8,
            00u8, 00u8, 00u8,
            00u8, 00u8, 00u8,]).unwrap();

        // Translating by more than the image width and height
        let translated = translate(&image, (5, 5));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_translate(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let translated = translate(&image, (30, 30));
            test::black_box(translated);
            });
    }
}
