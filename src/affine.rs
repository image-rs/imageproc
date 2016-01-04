//! Functions for affine transformations of images.

use image::{Pixel, GenericImage, ImageBuffer};

use definitions::{Clamp, HasBlack, VecBuffer};

use math::{Affine2, cast, Vec2};

use conv::ValueInto;

/// How to handle pixels whose pre-image lies between input pixels.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Interpolation {
    /// Choose the nearest pixel to the pre-image of the
    /// output pixel.
    Nearest,
    /// Bilinearly interpolate between the four pixels
    /// closest to the pre-image of the output pixel.
    Bilinear,
}

/// Applies an affine transformation to an image, or None if the provided
/// transformation is not invertible.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to black.
pub fn affine<I>(image: &I,
                 affine: Affine2,
                 interpolation: Interpolation)
                 -> Option<VecBuffer<I::Pixel>>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    affine_with_default(image, affine, I::Pixel::black(), interpolation)
}

/// Applies an affine transformation to an image, or None if the provided
/// transformation is not invertible.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to default.
pub fn affine_with_default<I>(image: &I,
                              affine: Affine2,
                              default: I::Pixel,
                              interpolation: Interpolation)
                              -> Option<VecBuffer<I::Pixel>>
    where I: GenericImage,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let inverse: Affine2;
    match affine.inverse() {
        None => return None,
        Some(inv) => inverse = inv,
    }

    let (width, height) = image.dimensions();
    let no_channel = I::Pixel::channel_count() as u32;
    let mut out = Vec::with_capacity((width * height * no_channel) as usize);

    for y in 0..height {
        for x in 0..width {

            let preimage = inverse.apply(Vec2::new(x as f32, y as f32));
            let px = preimage[0];
            let py = preimage[1];

            let pix = match interpolation {
                Interpolation::Nearest => {
                    nearest(image, px, py, width, height, default)
                }
                Interpolation::Bilinear => {
                    interpolate(image, px, py, width, height, default)
                }
            };

            for c in pix.channels() {
                out.push(*c);
            }
        }
    }

    Some(ImageBuffer::from_raw(width, height, out).unwrap())
}

/// Rotate an image clockwise about provided center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are black.
pub fn rotate<I>(image: &I,
                 center: (f32, f32),
                 theta: f32,
                 interpolation: Interpolation)
                 -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    rotate_with_default(image,
                        center,
                        theta,
                        <I::Pixel as HasBlack>::black(),
                        interpolation)
}

/// Rotate an image clockwise about its center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are black.
pub fn rotate_about_center<I>(image: &I,
                              theta: f32,
                              interpolation: Interpolation)
                              -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let center = (image.width() as f32 / 2f32, image.height() as f32 / 2f32);
    rotate(image, center, theta, interpolation)
}

/// Rotate an image clockwise about provided center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to default.
pub fn rotate_with_default<I>(image: &I,
                              center: (f32, f32),
                              theta: f32,
                              default: I::Pixel,
                              interpolation: Interpolation)
                              -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    match interpolation {
        Interpolation::Nearest => rotate_nearest(image, center, theta, default),
        Interpolation::Bilinear => rotate_bilinear(image, center, theta, default),
    }
}

fn rotate_nearest<I>(image: &I,
                     center: (f32, f32),
                     theta: f32,
                     default: I::Pixel)
                     -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let (width, height) = image.dimensions();
    let no_channel = I::Pixel::channel_count() as u32;
    let mut out = Vec::with_capacity((width * height * no_channel) as usize);

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let center_x = center.0;
    let center_y = center.1;

    for y in 0..height {
        let dy = y as f32 - center_y;
        let mut px = center_x + sin_theta * dy - cos_theta * center_x;
        let mut py = center_y + cos_theta * dy + sin_theta * center_x;

        for _ in 0..width {

            let pix = nearest(image, px, py, width, height, default);
            for c in pix.channels() {
                out.push(*c);
            }

            px += cos_theta;
            py -= sin_theta;
        }
    }

    ImageBuffer::from_raw(width, height, out).unwrap()
}

fn rotate_bilinear<I>(image: &I,
                      center: (f32, f32),
                      theta: f32,
                      default: I::Pixel)
                      -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let (width, height) = image.dimensions();
    let no_channel = I::Pixel::channel_count() as u32;
    let mut out = Vec::with_capacity((width * height * no_channel) as usize);

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let center_x = center.0;
    let center_y = center.1;

    for y in 0..height {
        let dy = y as f32 - center_y;
        let mut px = center_x + sin_theta * dy - cos_theta * center_x;
        let mut py = center_y + cos_theta * dy + sin_theta * center_x;

        for _ in 0..width {

            let pix = interpolate(image, px, py, width, height, default);
            for c in pix.channels() {
                out.push(*c);
            }

            px += cos_theta;
            py -= sin_theta;
        }
    }

    ImageBuffer::from_raw(width, height, out).unwrap()
}

/// Translates the input image by t. Note that image coordinates increase from
/// top left to bottom right. Output pixels whose pre-image are not in the input
/// image are set to the boundary pixel in the input image nearest to their pre-image.
// TODO: it's possibly confusing that this has different behaviour to
// TODO: attempting the equivalent transformation via the affine function
pub fn translate<I>(image: &I, t: (i32, i32)) -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    use std::cmp;

    let (width, height) = image.dimensions();
    let no_channel = I::Pixel::channel_count() as u32;
    let mut out = Vec::with_capacity((width * height * no_channel) as usize);

    let w = (width - 1) as i32;
    let h = (height - 1) as i32;

    for y in 0..height {
        for x in 0..width {
            let x_in = cmp::max(0, cmp::min(x as i32 - t.0, w));
            let y_in = cmp::max(0, cmp::min(y as i32 - t.1, h));
            let p = image.get_pixel(x_in as u32, y_in as u32);
            for c in p.channels() {
                out.push(*c);
            }
        }
    }

    ImageBuffer::from_raw(width, height, out).unwrap()
}

fn blend<P>(top_left: P,
            top_right: P,
            bottom_left: P,
            bottom_right: P,
            right_weight: f32,
            bottom_weight: f32)
            -> P
    where P: Pixel,
          P::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let top = top_left.map2(&top_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    let bottom = bottom_left.map2(&bottom_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    top.map2(&bottom,
             |u, v| P::Subpixel::clamp((1f32 - bottom_weight) * cast(u) + bottom_weight * cast(v)))
}

fn interpolate<I>(image: &I, x: f32, y: f32, width: u32, height: u32, default: I::Pixel) -> I::Pixel
    where I: GenericImage,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let left = x.floor();
    let right = left + 1f32;
    let top = y.floor();
    let bottom = top + 1f32;

    let right_weight = x - left;
    let bottom_weight = y - top;

    let x_out_of_bounds = left < 0f32 || right >= width as f32;
    let y_out_of_bounds = top < 0f32 || bottom >= height as f32;

    if x_out_of_bounds || y_out_of_bounds {
        return default;
    } else {
        return blend(image.get_pixel(left as u32, top as u32),
                     image.get_pixel(right as u32, top as u32),
                     image.get_pixel(left as u32, bottom as u32),
                     image.get_pixel(right as u32, bottom as u32),
                     right_weight,
                     bottom_weight);
    }
}

fn nearest<I>(image: &I, x: f32, y: f32, width: u32, height: u32, default: I::Pixel) -> I::Pixel
    where I: GenericImage
{
    let rx = x.round();
    let ry = y.round();

    if out_of_bounds(rx, ry, width, height) {
        return default;
    } else {
        let source = image.get_pixel(rx as u32, ry as u32);
        return source;
    }
}

fn out_of_bounds(x: f32, y: f32, width: u32, height: u32) -> bool {
    let x_out_of_bounds = x < 0f32 || x >= width as f32;
    let y_out_of_bounds = y < 0f32 || y >= height as f32;
    x_out_of_bounds || y_out_of_bounds
}

#[cfg(test)]
mod test {

    use super::{affine, rotate_bilinear, rotate_nearest, translate, Interpolation};
    use utils::gray_bench_image;
    use image::{GenericImage, GrayImage, ImageBuffer, Luma};
    use math::{Affine2, Mat2, Vec2};
    use test;

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_rotate_nearest_zero_radians() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let rotated = rotate_nearest(&image, (1f32, 0f32), 0f32, Luma([99u8]));
        assert_pixels_eq!(rotated, image);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn text_rotate_nearest_quarter_turn_clockwise() {
        use std::f32;

        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            11, 01, 99,
            12, 02, 99]).unwrap();

        let rotated
            = rotate_nearest(&image, (1f32, 0f32), f32::consts::PI / 2f32, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn text_rotate_nearest_half_turn_anticlockwise() {
        use std::f32;

        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            12, 11, 10,
            02, 01, 00]).unwrap();

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

    #[bench]
    fn bench_rotate_bilinear(b: &mut test::Bencher) {
        let mut image: GrayImage = ImageBuffer::new(200, 200);
        for pix in image.pixels_mut() {
            *pix = Luma([15u8]);
        }

        b.iter(|| {
            let rotated = rotate_bilinear(&image, (3f32, 3f32), 1f32, Luma([0u8]));
            test::black_box(rotated);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_positive_x_positive_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 01, 02,
            10, 11, 12,
            20, 21, 22,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 00, 01,
            00, 00, 01,
            10, 10, 11,]).unwrap();

        let translated = translate(&image, (1, 1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_positive_x_negative_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 01, 02,
            10, 11, 12,
            20, 21, 22,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            10, 10, 11,
            20, 20, 21,
            20, 20, 21,]).unwrap();

        let translated = translate(&image, (1, -1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_large_x_large_y() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 01, 02,
            10, 11, 12,
            20, 21, 22,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 00, 00,
            00, 00, 00,
            00, 00, 00,]).unwrap();

        // Translating by more than the image width and height
        let translated = translate(&image, (5, 5));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn bench_translate(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let translated = translate(&image, (30, 30));
            test::black_box(translated);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_affine_translation() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 01, 02,
            10, 11, 12,
            20, 21, 22,]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            00, 00, 00,
            00, 00, 01,
            00, 10, 11,]).unwrap();

        let aff = Affine2::new(
            Mat2::new(1f32, 0f32, 0f32, 1f32),
            Vec2::new(1f32, 1f32));

        if let Some(translated) = affine(&image, aff, Interpolation::Nearest) {
            assert_pixels_eq!(translated, expected);
        }
        else {
            assert!(false, "Affine transformation returned None");
        }
    }
}
