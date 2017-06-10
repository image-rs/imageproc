//! Functions for affine transformations of images.

use image::{Pixel, GenericImage, ImageBuffer};
use definitions::{Clamp, HasBlack, Image};
use math::cast;
use nalgebra::{Affine2,Point2};
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
pub fn affine<P>(image: &Image<P>,
                 affine: Affine2<f32>,
                 interpolation: Interpolation)
                 -> Option<Image<P>>
    where P: Pixel + HasBlack + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    affine_with_default(image, affine, P::black(), interpolation)
}

/// Applies an affine transformation to an image, or None if the provided
/// transformation is not invertible.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to default.
pub fn affine_with_default<P>(image: &Image<P>,
                              affine: Affine2<f32>,
                              default: P,
                              interpolation: Interpolation)
                              -> Option<Image<P>>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let inverse: Affine2<f32>;
    match affine.try_inverse() {
        None => return None,
        Some(inv) => inverse = inv,
    }

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {

            let preimage = inverse*Point2::new(x as f32, y as f32);
            let px = preimage[0];
            let py = preimage[1];

            let pix = match interpolation {
                Interpolation::Nearest => {
                    nearest(image, px, py, default)
                }
                Interpolation::Bilinear => {
                    interpolate(image, px, py, default)
                }
            };
            unsafe { out.unsafe_put_pixel(x, y, pix); }
        }
    }

    Some(out)
}

/// Rotate an image clockwise about provided center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are black.
pub fn rotate<P>(image: &Image<P>,
                 center: (f32, f32),
                 theta: f32,
                 interpolation: Interpolation)
                 -> Image<P>
    where P: Pixel + HasBlack + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    rotate_with_default(image,
                        center,
                        theta,
                        <P as HasBlack>::black(),
                        interpolation)
}

/// Rotate an image clockwise about its center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are black.
pub fn rotate_about_center<P>(image: &Image<P>,
                              theta: f32,
                              interpolation: Interpolation)
                              -> Image<P>
    where P: Pixel + HasBlack + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let center = (image.width() as f32 / 2f32, image.height() as f32 / 2f32);
    rotate(image, center, theta, interpolation)
}

/// Rotate an image clockwise about provided center by theta radians.
/// The output image has the same dimensions as the input. Output pixels
/// whose pre-image lies outside the input image are set to default.
pub fn rotate_with_default<P>(image: &Image<P>,
                              center: (f32, f32),
                              theta: f32,
                              default: P,
                              interpolation: Interpolation)
                              -> Image<P>
    where P: Pixel + HasBlack + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    match interpolation {
        Interpolation::Nearest => rotate_nearest(image, center, theta, default),
        Interpolation::Bilinear => rotate_bilinear(image, center, theta, default),
    }
}

fn rotate_nearest<P>(image: &Image<P>,
                     center: (f32, f32),
                     theta: f32,
                     default: P)
                     -> Image<P>
    where P: Pixel + 'static
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let center_x = center.0;
    let center_y = center.1;

    for y in 0..height {
        let dy = y as f32 - center_y;
        let mut px = center_x + sin_theta * dy - cos_theta * center_x;
        let mut py = center_y + cos_theta * dy + sin_theta * center_x;

        for x in 0..width {

            unsafe {
                let pix = nearest(image, px, py, default);
                out.unsafe_put_pixel(x, y, pix);
            }

            px += cos_theta;
            py -= sin_theta;
        }
    }

    out
}

fn rotate_bilinear<P>(image: &Image<P>,
                      center: (f32, f32),
                      theta: f32,
                      default: P)
                      -> Image<P>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let center_x = center.0;
    let center_y = center.1;

    for y in 0..height {
        let dy = y as f32 - center_y;
        let mut px = center_x + sin_theta * dy - cos_theta * center_x;
        let mut py = center_y + cos_theta * dy + sin_theta * center_x;

        for x in 0..width {

            let pix = interpolate(image, px, py, default);
            unsafe { out.unsafe_put_pixel(x, y, pix); }

            px += cos_theta;
            py -= sin_theta;
        }
    }

    out
}

/// Translates the input image by t. Note that image coordinates increase from
/// top left to bottom right. Output pixels whose pre-image are not in the input
/// image are set to the boundary pixel in the input image nearest to their pre-image.
// TODO: it's possibly confusing that this has different behaviour to
// TODO: attempting the equivalent transformation via the affine function
pub fn translate<P>(image: &Image<P>, t: (i32, i32)) -> Image<P>
    where P: Pixel + 'static
{
    use std::cmp;

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let w = width as i32;
    let h = height as i32;

    for y in 0..height {
        for x in 0..width {
            let x_in = cmp::max(0, cmp::min(x as i32 - t.0, w - 1));
            let y_in = cmp::max(0, cmp::min(y as i32 - t.1, h - 1));
            // (x_in, y_in) and (x, y) are guaranteed to be in bounds
            unsafe {
                let p = image.unsafe_get_pixel(x_in as u32, y_in as u32);
                out.unsafe_put_pixel(x, y, p);
            }
        }
    }

    out
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

fn interpolate<P>(image: &Image<P>, x: f32, y: f32, default: P) -> P
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let left = x.floor();
    let right = left + 1f32;
    let top = y.floor();
    let bottom = top + 1f32;

    let right_weight = x - left;
    let bottom_weight = y - top;

    // default if out of bound
    let (width, height) = image.dimensions();
    if left < 0f32 || right >= width as f32 || top < 0f32 || bottom >= height as f32 {
        default
    } else {
        let (tl, tr, bl, br) = unsafe {
            (image.unsafe_get_pixel(left as u32, top as u32),
             image.unsafe_get_pixel(right as u32, top as u32),
             image.unsafe_get_pixel(left as u32, bottom as u32),
             image.unsafe_get_pixel(right as u32, bottom as u32),)
        };
        blend(tl, tr, bl, br, right_weight, bottom_weight)
    }
}

fn nearest<P: Pixel + 'static>(image: &Image<P>, x: f32, y: f32, default: P) -> P
{
    let rx = x.round();
    let ry = y.round();

    // default if out of bound
    let (width, height) = image.dimensions();
    if rx < 0f32 || rx >= width as f32 || ry < 0f32 || ry >= height as f32 {
        default
    } else {
        unsafe { image.unsafe_get_pixel(rx as u32, ry as u32) }
    }
}

#[cfg(test)]
mod test {

    use super::{affine, rotate_bilinear, rotate_nearest, translate, Interpolation};
    use utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use nalgebra::{Affine2,Matrix3};
    use test;

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_rotate_nearest_zero_radians() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let rotated = rotate_nearest(&image, (1f32, 0f32), 0f32, Luma([99u8]));
        assert_pixels_eq!(rotated, image);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn text_rotate_nearest_quarter_turn_clockwise() {
        use std::f32;

        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            11, 01, 99;
            12, 02, 99);

        let rotated
            = rotate_nearest(&image, (1f32, 0f32), f32::consts::PI / 2f32, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn text_rotate_nearest_half_turn_anticlockwise() {
        use std::f32;

        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            12, 11, 10;
            02, 01, 00);

        let rotated = rotate_nearest(&image, (1f32, 0.5f32), -f32::consts::PI, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[bench]
    fn bench_rotate_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        b.iter(|| {
            let rotated = rotate_nearest(&image, (3f32, 3f32), 1f32, Luma([0u8]));
            test::black_box(rotated);
        });
    }

    #[bench]
    fn bench_rotate_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        b.iter(|| {
            let rotated = rotate_bilinear(&image, (3f32, 3f32), 1f32, Luma([0u8]));
            test::black_box(rotated);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_positive_x_positive_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 01;
            00, 00, 01;
            10, 10, 11);

        let translated = translate(&image, (1, 1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_positive_x_negative_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            10, 10, 11;
            20, 20, 21;
            20, 20, 21);

        let translated = translate(&image, (1, -1));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_translate_large_x_large_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 00;
            00, 00, 00);

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
    fn test_affine() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 01;
            00, 10, 11);

        let aff = Affine2::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ));

        if let Some(translated) = affine(&image, aff, Interpolation::Nearest) {
            assert_pixels_eq!(translated, expected);
        }
        else {
            assert!(false, "Affine transformation returned None");
        }
    }

    #[bench]
    fn bench_affine_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Affine2::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ));

        b.iter(|| {
            let transformed = affine(&image, aff, Interpolation::Nearest);
            test::black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Affine2::from_matrix_unchecked(Matrix3::new(
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ));

        b.iter(|| {
            let transformed = affine(&image, aff, Interpolation::Bilinear);
            test::black_box(transformed);
        });
    }
}
