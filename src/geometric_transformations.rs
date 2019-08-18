//! Geometric transformations of images.
//! Functions defined in this module do not change the pixel content, but deform the pixel grid and
//! copy corresponding pixels from the source image to their new destination.

use image::{Pixel, GenericImageView, ImageBuffer};
use crate::definitions::{Clamp, Image};

use crate::math::cast;
use conv::ValueInto;
use std::ops::Mul;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
enum TransformationClass {
    Translation,
    Affine,
    Projection
}

/// A 2d Projective transformation, stored as a row major 3x3 matrix.
/// Transformations combine by multiplication.
///
/// I.e. to define rotation around image center for a VGA image.
/// ```
/// use imageproc::geometric_transformations::*;
/// let (cx, cy) = (320.0, 240.0);
/// let c_rotation = Projection::translate(cx, cy)*Projection::rotate(30f32.to_radians())*Projection::translate(-cx, -cy);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Projection {
    transform: [f32; 9],
    inverse:   [f32; 9],
    class: TransformationClass,
}

impl Projection {
    /// Create a 2d projective transform from a row-major 3x3 matrix in homogeneous coordinates.
    /// Matrix must be invertible, otherwise it does not define a Projection (by definition).
    pub fn from_matrix(transform: [f32; 9]) -> Option<Projection> {
        let transform = normalize(transform);
        let class = class_from_matrix(transform);
        try_inverse(&transform)
            .map(|inverse| 
                 Projection { transform, inverse, class })
    }

    /// Defines a translation by (tx, ty).
    pub fn translate(tx: f32, ty: f32) -> Projection {
        Projection { 
            transform: [
                1.0, 0.0, tx,
                0.0, 1.0, ty,
                0.0, 0.0, 1.0],
            inverse: [
                1.0, 0.0, -tx,
                0.0, 1.0, -ty,
                0.0, 0.0, 1.0],
            class: TransformationClass::Translation,
        }
    }

    /// Defines a rotation around the origin by angle theta radians. Origin is top left corner and
    /// positive direction is clockwise (because y grows downwards).
    pub fn rotate(theta: f32) -> Projection {
        let s = theta.sin();
        let c = theta.cos();
        Projection { 
            transform: [
                c,  -s,   0.0,
                s,   c,   0.0,
                0.0, 0.0, 1.0],
            inverse: [
                c,   s,   0.0,
                -s,  c,   0.0,
                0.0, 0.0, 1.0],
            class: TransformationClass::Affine,
        }
    }

    /// Creates an anisotropic scaling (sx, sy).
    pub fn scale(sx: f32, sy: f32) -> Projection {
        Projection { 
            transform: [
                sx,  0.0, 0.0,
                0.0, sy,  0.0,
                0.0, 0.0, 1.0],
            inverse: [
                1.0/sx, 0.0, 0.0,
                0.0, 1.0/sy, 0.0,
                0.0, 0.0, 1.0],
            class: TransformationClass::Affine,
        }
    }

    /// Inverts the transformation.
    pub fn invert(self) -> Projection {
        Projection { transform: self.inverse, inverse: self.transform, class: self.class }
    }

    /// Calculates a homography from a set of four control point pairs.
    pub fn from_control_points(from: [(f32, f32); 4], to: [(f32, f32); 4]) -> Option<Projection> {
        use rulinalg::matrix::*;

        let (xf1, yf1, xf2, yf2, xf3, yf3, xf4, yf4) = 
            (from[0].0 as f64, from[0].1 as f64,
             from[1].0 as f64, from[1].1 as f64,
             from[2].0 as f64, from[2].1 as f64,
             from[3].0 as f64, from[3].1 as f64);

        let (x1, y1, x2, y2, x3, y3, x4, y4) =
            (to[0].0 as f64, to[0].1 as f64,
             to[1].0 as f64, to[1].1 as f64,
             to[2].0 as f64, to[2].1 as f64,
             to[3].0 as f64, to[3].1 as f64);

        let a = Matrix::new(9, 9, vec![
            0f64, 0.0, 0.0, -xf1, -yf1, -1.0, y1 * xf1, y1 * yf1, y1,
            xf1, yf1, 1.0, 0.0, 0.0, 0.0,-x1*xf1,-x1*yf1,-x1,
            0.0, 0.0, 0.0,-xf2,-yf2,-1.0, y2*xf2, y2*yf2, y2,
            xf2, yf2, 1.0, 0.0, 0.0, 0.0,-x2*xf2,-x2*yf2,-x2,
            0.0, 0.0, 0.0,-xf3,-yf3,-1.0, y3*xf3, y3*yf3, y3,
            xf3, yf3, 1.0, 0.0, 0.0, 0.0,-x3*xf3,-x3*yf3,-x3,
            0.0, 0.0, 0.0,-xf4,-yf4,-1.0, y4*xf4, y4*yf4, y4,
            xf4, yf4, 1.0, 0.0, 0.0, 0.0,-x4*xf4,-x4*yf4,-x4,
            xf4, yf4, 1.0, 0.0, 0.0, 0.0,-x4*xf4,-x4*yf4,-x4]);
                
        match a.svd() {
            Err(_) => None,
            Ok((s, _, v)) => {
                if s[[8,8]].abs() > 0.0001 || // rank(a) must be 8
                   s[[7,7]].abs() < 0.0001 {  // but not 7
                    None
                } else {
                    let h = v.col(8).into_matrix().into_vec();
                    let mut transform = 
                        [h[0] as f32, h[1] as f32, h[2] as f32,
                         h[3] as f32, h[4] as f32, h[5] as f32,
                         h[6] as f32, h[7] as f32, h[8] as f32];
                    transform = normalize(transform);
                    let class = class_from_matrix(transform);

                    try_inverse(&transform)
                        .map(|inverse| { Projection { transform, inverse, class }})
                }
            }
        }
    }

    // Helper functions used as optimization in warp.
    #[inline(always)]
    fn map_projective(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        let d = t[6]*x + t[7]*y + t[8];

        ((t[0]*x + t[1]*y + t[2])/d,
         (t[3]*x + t[4]*y + t[5])/d)
    }

    #[inline(always)]
    fn map_affine(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;

        ((t[0]*x + t[1]*y + t[2]),
         (t[3]*x + t[4]*y + t[5]))
    }

    #[inline(always)]
    fn map_translation(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;
        let tx = t[2];
        let ty = t[5];

        (x + tx,
         y + ty)
    }
}

/// Combines two projective transformations.
/// (matrix multiplication)
impl<'a, 'b> Mul<&'b Projection> for &'a Projection {
    type Output = Projection;

    fn mul(self, rhs: &Projection) -> Projection {
        use TransformationClass as TC;
        let t = mul3x3(self.transform, rhs.transform);
        let i = mul3x3(rhs.inverse, self.inverse);

        let class = match (self.class, rhs.class) {
            (TC::Translation, TC::Translation) => TC::Translation,
            (TC::Translation, TC::Affine) => TC::Affine,
            (TC::Affine, TC::Translation) => TC::Affine,
            (TC::Affine, TC::Affine)      => TC::Affine,
            (_, _) => TC::Projection
        };

        Projection { transform: t, inverse: i, class: class }
    }
}

/// Combines two projective transformations.
/// (matrix multiplication)
impl Mul<Projection> for Projection {
    type Output = Projection;

    fn mul(self, rhs: Projection) -> Projection {
        &self * &rhs
    }
}

impl<'a, 'b> Mul<&'b (f32, f32)> for &'a Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: &(f32, f32)) -> (f32, f32) {
        let (x, y) = *rhs;
        self.map_projective(x, y)
    }
}

impl Mul<(f32, f32)> for Projection {
    type Output = (f32, f32);

    fn mul(self, rhs: (f32, f32)) -> (f32, f32) {
        &self * &rhs
    }
}

/// Performs projective transformation to an image. Allocates an output image with same
/// dimensions as the input image. Output pixels outside the input image are set to `default`.
/// Projection `homography` defines a mapping from coordinates in the `output` image to coordinates of
/// the `image`.
pub fn warp<P>(
    homography: &Projection,
    image: &Image<P>,
    interpolation: Interpolation,
    default: P
) -> Image<P> 
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    warp_into(homography, image, interpolation, default, &mut out);

    out
}

/// Performs projective transformation `homography` mapping pixels from
/// `image` into `&mut output`.
/// Projection `homograpy` defines coordinate mapping from `out` to `image`.
pub fn warp_into<P>(
    homography: &Projection,
    image: &Image<P>,
    interpolation: Interpolation,
    default: P,
    out: &mut Image<P>
)
where
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32> + Sync,
{
    let nn = |x, y| nearest(image, x, y, default);
    let bl = |x, y| interpolate(image, x, y, default);
    let wt = |x, y| homography.map_projective(x, y);
    let wa = |x, y| homography.map_affine(x, y);
    let wp = |x, y| homography.map_translation(x, y);
    use TransformationClass as TC;
    use Interpolation as I;

    match (interpolation, homography.class) {
        (I::Nearest,  TC::Translation) => warp_inner(out, wt, nn),
        (I::Nearest,  TC::Affine)      => warp_inner(out, wa, nn),
        (I::Nearest,  TC::Projection)  => warp_inner(out, wp, nn),
        (I::Bilinear, TC::Translation) => warp_inner(out, wt, bl),
        (I::Bilinear, TC::Affine)      => warp_inner(out, wa, bl),
        (I::Bilinear, TC::Projection)  => warp_inner(out, wp, bl),
    }
}

/// Transforms input `image` into output image of the same size.
/// Fm `mapping` is the coordinate mapping function, mapping from out to in.
pub fn warp_with<P, Fm>(
    mapping: Fm,
    image: &Image<P>,
    interpolation: Interpolation,
    default: P
) -> Image<P> 
where
    Fm: Fn(f32, f32) -> (f32, f32) + Sync + Send,
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    warp_into_with(mapping, image, interpolation, default, &mut out);

    out
}

/// Warp image with custom function.
/// This enables the user to define a custom mapping such as a wave pattern:
/// ```
/// use image::{ImageBuffer, Luma};
/// use imageproc::utils::gray_bench_image;
/// use imageproc::geometric_transformations::*;
///
/// let img = gray_bench_image(300, 300);
/// let mut out = ImageBuffer::new(300, 300);
/// warp_into_with(|x, y| (x, y+(x/30.0).sin()), &img, Interpolation::Nearest, Luma([0u8]), &mut
/// out);
/// ```
pub fn warp_into_with<P, Fm>(
    mapping: Fm,
    image: &Image<P>,
    interpolation: Interpolation,
    default: P,
    out: &mut Image<P>
)
where
    Fm: Fn(f32, f32) -> (f32, f32) + Send + Sync,
    P: Pixel + Send + Sync + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,

{
    let nn = |x, y| nearest(image, x, y, default);
    let bl = |x, y| interpolate(image, x, y, default);
    use Interpolation as I;

    match interpolation {
        I::Nearest => warp_inner(out, mapping, nn),
        I::Bilinear => warp_inner(out, mapping, bl),
    }
}

// Work horse of all warp functions
// TODO: make faster by avoiding boundary checks in inner section of src image
fn warp_inner<P,Fc,Fi>(
    out: &mut Image<P>,
    mapping: Fc,
    get_pixel: Fi,
    )
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    Fc: Fn(f32, f32) -> (f32, f32) + Send + Sync,
    Fi: Fn(f32, f32) -> P + Send + Sync,
{
    let width = out.width();
    let raw_out = out.as_mut();
    let pitch = P::CHANNEL_COUNT as usize * width as usize;

    raw_out.par_chunks_mut(pitch).enumerate().for_each(|(y, row)| { 
        for (x, slice) in row.chunks_mut(P::CHANNEL_COUNT as usize).enumerate() {
            let (px, py) = mapping(x as f32, y as f32);
            *P::from_slice_mut(slice) = get_pixel(px, py);
        }
    });
        
}

// Classifies transformation by looking up transformation matrix coefficients
fn class_from_matrix(mx: [f32; 9]) -> TransformationClass {

    if (mx[6]-0.0).abs() < 1e-10 &&
       (mx[7]-0.0).abs() < 1e-10 &&
       (mx[8]-1.0).abs() < 1e-10
    {  
        if (mx[0]-1.0).abs() < 1e-10 &&
           (mx[1]-0.0).abs() < 1e-10 &&
           (mx[3]-0.0).abs() < 1e-10 &&
           (mx[4]-1.0).abs() < 1e-10 
        {
            TransformationClass::Translation 
        } else { 
            TransformationClass::Affine
        }
    } else {
        TransformationClass::Projection
    }
}

fn normalize(mx: [f32; 9]) -> [f32; 9] {
    [mx[0]/mx[8], mx[1]/mx[8], mx[2]/mx[8],
     mx[3]/mx[8], mx[4]/mx[8], mx[5]/mx[8],
     mx[6]/mx[8], mx[7]/mx[8], mx[8]/mx[8]]
}

fn try_inverse(t: &[f32; 9]) -> Option<[f32; 9]> {
    let (
        t00, t01, t02,
        t10, t11, t12,
        t20, t21, t22
    ) = (
        t[0], t[1], t[2],
        t[3], t[4], t[5],
        t[6], t[7], t[8]
    );

    let m00 = t11 as f64 * t22 as f64 - t12 as f64 * t21 as f64;
    let m01 = t10 as f64 * t22 as f64 - t12 as f64 * t20 as f64;
    let m02 = t10 as f64 * t21 as f64 - t11 as f64 * t20 as f64;

    let det = t00 as f64 * m00 - t01 as f64 * m01 + t02 as f64 * m02;

    if (det).abs() < 1e-10 {
        return None;
    }

    let m10 = t01 as f64 * t22 as f64 - t02 as f64 * t21 as f64;
    let m11 = t00 as f64 * t22 as f64 - t02 as f64 * t20 as f64;
    let m12 = t00 as f64 * t21 as f64 - t01 as f64 * t20 as f64;
    let m20 = t01 as f64 * t12 as f64 - t02 as f64 * t11 as f64;
    let m21 = t00 as f64 * t12 as f64 - t02 as f64 * t10 as f64;
    let m22 = t00 as f64 * t11 as f64 - t01 as f64 * t10 as f64;

    let inv = [
        (m00 / det) as f32,  (-m10 / det) as f32, (m20 / det) as f32,
        (-m01 / det) as f32, (m11 / det) as f32,  (-m21 / det) as f32,
        (m02 / det) as f32,  (-m12 / det) as f32, (m22 / det) as f32
    ];

    Some(normalize(inv))
}

fn mul3x3(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let (
        a11, a12, a13,
        a21, a22, a23,
        a31, a32, a33
        ) = (
        a[0], a[1], a[2],
        a[3], a[4], a[5], 
        a[6], a[7], a[8]
        );
    let (
        b11, b12, b13,
        b21, b22, b23,
        b31, b32, b33
        ) = (
        b[0], b[1], b[2],
        b[3], b[4], b[5], 
        b[6], b[7], b[8]
        );

    [a11*b11 + a12*b21 + a13*b31, a11*b12 + a12*b22 + a13*b32, a11*b13 + a12*b23 +a13*b33,
     a21*b11 + a22*b21 + a23*b31, a21*b12 + a22*b22 + a23*b32, a21*b13 + a22*b23 + a23*b33,
     a31*b11 + a32*b21 + a33*b31, a31*b12 + a32*b22 + a33*b32, a31*b13 + a32*b23 + a33*b33]
}

fn blend<P>(
    top_left: P,
    top_right: P,
    bottom_left: P,
    bottom_right: P,
    right_weight: f32,
    bottom_weight: f32,
) -> P
where
    P: Pixel,
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let top = top_left.map2(&top_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    let bottom = bottom_left.map2(&bottom_right, |u, v| {
        P::Subpixel::clamp((1f32 - right_weight) * cast(u) + right_weight * cast(v))
    });

    top.map2(&bottom, |u, v| {
        P::Subpixel::clamp((1f32 - bottom_weight) * cast(u) + bottom_weight * cast(v))
    })
}


#[allow(deprecated)]
fn interpolate<P>(image: &Image<P>, x: f32, y: f32, default: P) -> P
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
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
            (
                image.unsafe_get_pixel(left as u32, top as u32),
                image.unsafe_get_pixel(right as u32, top as u32),
                image.unsafe_get_pixel(left as u32, bottom as u32),
                image.unsafe_get_pixel(right as u32, bottom as u32),
            )
        };
        blend(tl, tr, bl, br, right_weight, bottom_weight)
    }
}

#[inline(always)]
#[allow(deprecated)]
fn nearest<P: Pixel + 'static>(image: &Image<P>, x: f32, y: f32, default: P) -> P {
    let rx = x.round() as u32;
    let ry = y.round() as u32;
    let (width, height) = image.dimensions();
    if x.round() < 0f32 ||  y.round() < 0f32 {
        default
    } else if rx >= width || ry >= height {
        default
    } else {
        unsafe { image.unsafe_get_pixel(rx, ry) }
    }
}

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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use ::test;

    #[test]
    fn test_rotate_nearest_zero_radians() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);
        let rot = Projection::translate(1.0, 0.0)*Projection::rotate(0.0)*Projection::translate(-1.0, 0.0);

        let rotated = warp(&rot.invert(), &image, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, image);
    }

    #[test]
    fn text_rotate_nearest_quarter_turn_clockwise() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            11, 01, 99;
            12, 02, 99);
        let c = Projection::translate(1.0, 0.0);
        let rot = c*Projection::rotate(90f32.to_radians())*c.invert();

        let rotated = warp(&rot.invert(), &image, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[test]
    fn text_rotate_nearest_half_turn_anticlockwise() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);

        let expected = gray_image!(
            12, 11, 10;
            02, 01, 00);
        let c = Projection::translate(1.0, 0.5);

        let rot = c*Projection::rotate((-180f32).to_radians())*c.invert();

        let rotated = warp(&rot.invert(), &image, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[bench]
    fn bench_rotate_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Projection::translate(3.0, 3.0);
        let rot = c*Projection::rotate(1f32.to_degrees())*c.invert();
        b.iter(|| {
            let rotated = warp(&rot.invert(), &image, Interpolation::Nearest, Luma([98u8]));
            test::black_box(rotated);
        });
    }

    #[bench]
    fn bench_rotate_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Projection::translate(3.0, 3.0);
        let rot = c*Projection::rotate(1f32.to_degrees())*c.invert();
        b.iter(|| {
            let rotated = warp(&rot.invert(), &image, Interpolation::Bilinear, Luma([98u8]));
            test::black_box(rotated);
        });
    }

    #[test]
    fn test_translate_positive_x_positive_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 01;
            00, 10, 11);

        let translated = warp(&Projection::translate(-1.0,-1.0), &image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
    fn test_translate_positive_x_negative_y() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 10, 11;
            00, 20, 21;
            00, 00, 00);

        let translated = warp(&Projection::translate(-1.0,1.0), &image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[test]
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
        let translated = warp(&Projection::translate(-5.0, -5.0), &image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_translate(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        let t = Projection::translate(-30.0, -30.0);

        b.iter(|| {
            let translated = warp(&t, &image, Interpolation::Nearest, Luma([0u8]));
            test::black_box(translated);
        });
    }

    #[bench]
    fn bench_translate_with(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);

        b.iter(|| {
            let (width, height) = image.dimensions();
            let mut out = ImageBuffer::new(width, height);
            warp_into_with(|x, y| (x-30.0, y-30.0), &image, Interpolation::Nearest, Luma([0u8]), &mut out);
            test::black_box(out);
        });
    }

    #[test]
    fn test_affine() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let expected = gray_image!(
            00, 00, 00;
            00, 00, 01;
            00, 10, 11);

        let aff = Projection::from_matrix([
            1.0, 0.0, -1.0,
            0.0, 1.0, -1.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        let translated = warp(&aff, &image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_affine_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Projection::from_matrix([
            1.0, 0.0, -1.0,
            0.0, 1.0, -1.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        b.iter(|| {
            let transformed = warp(&aff, &image, Interpolation::Nearest, Luma([0u8]));
            test::black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Projection::from_matrix([
            1.8, -0.2, 5.0,
            0.2, 1.9, 6.0,
            0.0002, 0.0003, 1.0,
        ]).unwrap();

        b.iter(|| {
            let transformed = warp(&aff, &image, Interpolation::Bilinear, Luma([0u8]));
            test::black_box(transformed);
        });
    }

    #[test]
    fn test_from_cp_translate() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(10f32, 5.0), (60.0, 55.0), (60.0, 5.0), (10.0, 55.0)];

        let p = Projection::from_control_points(from, to);
        assert!(p.is_some());

        let out = p.unwrap()*(0f32, 0f32);

        assert_approx_eq!(out.0, 10.0, 1e-10);
        assert_approx_eq!(out.1, 5.0, 1e-10);
    }

    #[test]
    fn test_from_cp() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(16f32, 20.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];

        let p = Projection::from_control_points(from, to);
        assert!(p.is_some());

        let out = p.unwrap()*(0f32, 0f32);

        assert_approx_eq!(out.0, 16.0, 1e-10);
        assert_approx_eq!(out.1, 20.0, 1e-10);
    }

    #[test]
    fn test_from_cp_trans_colinear() {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(0f32, 5.0), (0.0, 55.0), (0.0, 5.0), (10.0, 55.0)];

        let p = Projection::from_control_points(from, to);
        // Should fail if 3 points are colinear
        assert!(p.is_none());
    }

    #[bench]
    fn bench_from_cp(b: &mut test::Bencher) {
        let from = [(0f32, 0.0), (50.0, 50.0), (50.0, 0.0), (0.0, 50.0)];
        let to = [(10f32, 5.0), (60.0, 55.0), (60.0, 5.0), (10.0, 55.0)];

        b.iter(|| {
            let proj = Projection::from_control_points(from, to);
            test::black_box(proj);
        });
    }
}
