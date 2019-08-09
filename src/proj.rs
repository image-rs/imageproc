//! Defines projections on images

use image::{Pixel, /*GenericImage, */GenericImageView, ImageBuffer};
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
#[derive(Copy, Clone, Debug)]
pub struct Proj {
    transform: [f32; 10], // 3x3 matrix + 1 for SIMD alignment
    inverse: [f32; 10],
    class: TransformationClass,
}

impl Proj {
    /// Create a 2d projective transform from a row-major 3x3 matrix in homogeneous coordinates.
    /// Matrix must be invertible, otherwise it does not define a Projection (by definition).
    pub fn from_matrix(tform: [f32; 9]) -> Option<Proj> {
        let t = &tform;
        let transform: [f32; 10] = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], 0.0];
        let transform = normalize(transform);
        let class = class_from_matrix(transform);
        try_inverse(&transform)
            .map(|inverse| 
                 Proj { transform, inverse, class })
    }

    /// Defines a translation by (tx, ty)
    pub fn trans(tx: f32, ty: f32) -> Proj {
        Proj { 
            transform: [
                1.0, 0.0, tx,
                0.0, 1.0, ty,
                0.0, 0.0, 1.0, 0.0],
            inverse: [
                1.0, 0.0, -tx,
                0.0, 1.0, -ty,
                0.0, 0.0, 1.0, 0.0],
            class: TransformationClass::Translation,
        }
    }

    /// Defines a rotation around the origin by angle theta degrees. Origin is (0,0) pixel
    /// coordinate, usually top left corner. To rotate around image center combine rotation with 
    /// two translations: T*Rotation*T^-1.
    pub fn rot(theta: f32) -> Proj {
        let theta = theta.to_radians();
        let s = theta.sin();
        let c = theta.cos();
        Proj { 
            transform: [
                c,  -s,   0.0,
                s,   c,   0.0,
                0.0, 0.0, 1.0, 0.0],
            inverse: [
                c,   s,   0.0,
                -s,  c,   0.0,
                0.0, 0.0, 1.0, 0.0],
            class: TransformationClass::Affine,
        }
    }

    /// Creates an anisotropic scaling (sx,sy).
    pub fn scale(sx: f32, sy: f32) -> Proj {
        Proj { 
            transform: [
                sx,  0.0, 0.0,
                0.0, sy,  0.0,
                0.0, 0.0, 1.0, 0.0],
            inverse: [
                1.0/sx, 0.0, 0.0,
                0.0, 1.0/sy, 0.0,
                0.0, 0.0, 1.0, 0.0],
            class: TransformationClass::Affine,
        }
    }

    /// Inverts the transformation.
    pub fn inv(self) -> Proj {
        Proj { transform: self.inverse, inverse: self.transform, class: self.class }
    }

    /// Performs projective transformation tform to an image. Allocates an output image with same
    /// dimensions as the input image. Output pixels outside the input image are set to `default`.
    pub fn warp_new<P>(&self,
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
            self.warp(image, interpolation, default, &mut out);

            out
        }

    /// Performs projective transformation `tfrom` mapping pixels from
    /// `image` into `&mut output`
    pub fn warp<P>(&self,
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
            let tform = self.inv();

            let nn = |x,y| nearest(image, x, y, default);
            let bl = |x,y| interpolate(image, x, y, default);
            let w_t = |x,y| tform.warp_t(x,y);
            let w_a = |x,y| tform.warp_a(x,y);
            let w_p = |x,y| tform.warp_p(x,y);
            use TransformationClass as TC;
            use Interpolation as I;

            match (interpolation, tform.class) {
                (I::Nearest,  TC::Translation) => warp_inner(out, w_t, nn),
                (I::Nearest,  TC::Affine)      => warp_inner(out, w_a, nn),
                (I::Nearest,  TC::Projection)  => warp_inner(out, w_p, nn),
                (I::Bilinear, TC::Translation) => warp_inner(out, w_t, bl),
                (I::Bilinear, TC::Affine)      => warp_inner(out, w_a, bl),
                (I::Bilinear, TC::Projection)  => warp_inner(out, w_p, bl),
            }
        }

    // Helper functions used as optiomization in warp
    #[inline(always)]
    fn warp_p(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;

        let d = t[6]*x + t[7]*y + t[8];

        ((t[0]*x + t[1]*y + t[2])/d,
         (t[3]*x + t[4]*y + t[5])/d)
    }

    #[inline(always)]
    fn warp_a(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;

        ((t[0]*x + t[1]*y + t[2]),
         (t[3]*x + t[4]*y + t[5]))
    }

    #[inline(always)]
    fn warp_t(&self, x: f32, y: f32) -> (f32, f32) {
        let t = &self.transform;

        ((x + t[2]),
         (y + t[5]))
    }
}

fn warp_inner<P,Fc,Fi>(
    out: &mut Image<P>,
    coord_tf: Fc,
    get_pixel: Fi,
    )
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: Send + Sync,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    Fc: Fn(f32, f32) -> (f32, f32),
    Fc: Send + Sync,
    Fi: Fn(f32, f32) -> P,
    Fi: Send + Sync,
{
    let width = out.width();
    let raw_out = out.as_mut();
    let pitch = P::CHANNEL_COUNT as usize * width as usize;

    raw_out.par_chunks_mut(pitch).enumerate().for_each(|(y, row)| { 
        for (x, slice) in row.chunks_mut(P::CHANNEL_COUNT as usize).enumerate() {
            let (px, py) = coord_tf(x as f32, y as f32);
            *P::from_slice_mut(slice) = get_pixel(px, py);
        }
    });
        
}

fn class_from_matrix(mx: [f32; 10]) -> TransformationClass {

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

fn normalize(mx: [f32; 10]) -> [f32; 10] {
    [mx[0]/mx[8], mx[1]/mx[8], mx[2]/mx[8],
     mx[3]/mx[8], mx[4]/mx[8], mx[5]/mx[8],
     mx[6]/mx[8], mx[7]/mx[8], mx[8]/mx[8], 0.0]
}

fn try_inverse(t: &[f32; 10]) -> Option<[f32; 10]> {
    let (
        t00, t01, t02,
        t10, t11, t12,
        t20, t21, t22
    ) = (
        t[0], t[1], t[2],
        t[3], t[4], t[5],
        t[6], t[7], t[8]
    );

    let m00 = t11 * t22 - t12 * t21;
    let m01 = t10 * t22 - t12 * t20;
    let m02 = t10 * t21 - t11 * t20;

    let det = t00 * m00 - t01 * m01 + t02 * m02;

    if (det).abs() < 1e-10 {
        return None;
    }

    let m10 = t01 * t22 - t02 * t21;
    let m11 = t00 * t22 - t02 * t20;
    let m12 = t00 * t21 - t01 * t20;
    let m20 = t01 * t12 - t02 * t11;
    let m21 = t00 * t12 - t02 * t10;
    let m22 = t00 * t11 - t01 * t10;

    let inv = [
        m00 / det, -m10 / det,  m20 / det,
       -m01 / det,  m11 / det, -m21 / det,
        m02 / det, -m12 / det,  m22 / det, 0.0
    ];

    Some(normalize(inv))
}

fn mul3x3(a: [f32; 10], b: [f32; 10]) -> [f32; 10] {
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
     a31*b11 + a32*b21 + a33*b31, a31*b12 + a32*b22 + a33*b32, a31*b13 + a32*b23 + a33*b33, 0.0]
}

/// Basically matrix multiplication
impl Mul<Proj> for Proj {
    type Output = Proj;

    fn mul(self, rhs: Proj) -> Proj {
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

        Proj { transform: t, inverse: i, class: class }
    }
}

/// Basically matrix multiplication
impl<'a, 'b> Mul<&Proj> for &'a Proj {
    type Output = Proj;

    fn mul(self, rhs: &Proj) -> Proj {
        use TransformationClass as TC;
        let t = mul3x3(self.transform, rhs.transform);
        let i = mul3x3(rhs.inverse, self.inverse);
        let class = match (self.class, rhs.class) {
            (TC::Translation, TC::Translation) => TC::Translation,
            (TC::Translation, TC::Affine) => TC::Affine,
            (TC::Affine, TC::Translation) => TC::Affine,
            (_, _) => TC::Projection
        };

        Proj { transform: t, inverse: i, class: class }
    }
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
    use image::{GrayImage, Luma, RgbImage, Rgb, Rgba};
    use ::test;

    #[test]
    fn test_rotate_nearest_zero_radians() {
        let image = gray_image!(
            00, 01, 02;
            10, 11, 12);
        let rot = Proj::trans(1.0, 0.0)*Proj::rot(0.0)*Proj::trans(-1.0, 0.0);

        let rotated = rot.warp_new(&image, Interpolation::Nearest, Luma([99u8]));
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
        let c = Proj::trans(1.0, 0.0);
        let rot = c*Proj::rot(90.0)*c.inv();

        let rotated = rot.warp_new(&image, Interpolation::Nearest, Luma([99u8]));
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
        let c = Proj::trans(1.0, 0.5);

        let rot = c*Proj::rot(-180.0)*c.inv();

        let rotated = rot.warp_new(&image, Interpolation::Nearest, Luma([99u8]));
        assert_pixels_eq!(rotated, expected);
    }

    #[bench]
    fn bench_rotate_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Proj::trans(3.0, 3.0);
        let rot = c*Proj::rot(1f32.to_degrees())*c.inv();
        b.iter(|| {
            let rotated = rot.warp_new(&image, Interpolation::Nearest, Luma([98u8]));
            test::black_box(rotated);
        });
    }

    #[bench]
    fn bench_rotate_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));
        let c = Proj::trans(3.0, 3.0);
        let rot = c*Proj::rot(1f32.to_degrees())*c.inv();
        b.iter(|| {
            let rotated = rot.warp_new(&image, Interpolation::Bilinear, Luma([98u8]));
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

        let translated = Proj::trans(1.0,1.0).warp_new(&image, Interpolation::Nearest, Luma([0u8]));
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

        let translated = Proj::trans(1.0,-1.0).warp_new(&image, Interpolation::Nearest, Luma([0u8]));
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
        let translated = Proj::trans(5.0, 5.0).warp_new(&image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_translate(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        let t = Proj::trans(30.0, 30.0);

        b.iter(|| {
            let translated = t.warp_new(&image, Interpolation::Nearest, Luma([0u8]));
            test::black_box(translated);
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

        let aff = Proj::from_matrix([
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        let translated = aff.warp_new(&image, Interpolation::Nearest, Luma([0u8]));
        assert_pixels_eq!(translated, expected);
    }

    #[bench]
    fn bench_affine_nearest(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Proj::from_matrix([
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        b.iter(|| {
            let transformed = aff.warp_new(&image, Interpolation::Nearest, Luma([0u8]));
            test::black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_bilinear(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(200, 200, Luma([15u8]));

        let aff = Proj::from_matrix([
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        b.iter(|| {
            let transformed = aff.warp_new(&image, Interpolation::Bilinear, Luma([0u8]));
            test::black_box(transformed);
        });
    }

    #[bench]
    fn bench_affine_nearest_fhd(b: &mut test::Bencher) {
        let image = RgbImage::from_pixel(1920, 1080, Rgb([15u8, 16, 17]));

        let aff = Proj::from_matrix([
            1.8, -0.2, 5.0,
            0.2, 1.9, 6.0,
            0.0002, 0.0003, 1.0,
        ]).unwrap();

        let t = Proj::trans(3.0, 3.0);

        b.iter(|| {
            let transformed = aff.warp_new(&image, Interpolation::Nearest, Rgb([0u8, 0, 0]));
            test::black_box(transformed);
        });
    }

    /*
    #[bench]
    fn bench_nearest_plain(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(1920, 1080, Luma([13u8]));

        b.iter(|| {
            for y in 0..1080 {
                for x in 0..1920 {
                    test::black_box(nearest(&image, x as f32, y as f32, Luma([99u8])));
                }
            }
        });
    }

    #[bench]
    fn bench_nearest_fast(b: &mut test::Bencher) {
        let image = GrayImage::from_pixel(1920, 1080, Luma([13u8]));

        b.iter(|| {
            for y in 0..1080 {
                for x in 0..1920 {
                    test::black_box(nearest_nocheck(&image, x as f32, y as f32, Luma([99u8])));
                }
            }
        });
    }
    */

}
