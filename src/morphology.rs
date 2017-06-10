//! Functions for computing [morphological operators].
//!
//! [morphological operators]: http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

use std::cmp::min;
use std::u8;
use image::{GenericImage, GrayImage, Luma};

/// How to measure distance when performing dilations and erosions.
/// See the [`distance_transform`](fn.distance_transform.html) documentation for examples.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Defines d((x1, y1), (x2, y2)) to be abs(x1 - x2) + abs(y1 - y2).
    /// Also known as the Manhattan or city block norm.
    L1,
    /// Defines d((x1, y1), (x2, y2)) to be max(abs(x1 - x2), abs(y1 - y2)).
    /// Also known as the chessboard norm.
    LInf
}

/// Sets all pixels within distance `k` of a foreground pixel to white.
///
/// A pixel is treated as belonging to the foreground if it has non-zero intensity.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{dilate, Norm};
///
/// let image = GrayImage::from_raw(5, 5, vec![
///     0,   0,   0,   0,   0,
///     0,   0,   0,   0,   0,
///     0,   0, 255,   0,   0,
///     0,   0,   0,   0,   0,
///     0,   0,   0,   0,   0
/// ]).unwrap();
///
/// // L1 norm
/// let l1_dilated = GrayImage::from_raw(5, 5, vec![
///     0,   0,   0,   0,   0,
///     0,   0, 255,   0,   0,
///     0, 255, 255, 255,   0,
///     0,   0, 255,   0,   0,
///     0,   0,   0,   0,   0
/// ]).unwrap();
/// assert_pixels_eq!(dilate(&image, Norm::L1, 1), l1_dilated);
///
/// // LInf norm
/// let linf_dilated = GrayImage::from_raw(5, 5, vec![
///    0,   0,   0,   0,   0,
///    0, 255, 255, 255,   0,
///    0, 255, 255, 255,   0,
///    0, 255, 255, 255,   0,
///    0,   0,   0,   0,   0
/// ]).unwrap();
///
/// assert_pixels_eq!(dilate(&image, Norm::LInf, 1), linf_dilated);
/// # }
/// ```
pub fn dilate(image: &GrayImage, norm: Norm, k: u8) -> GrayImage {
    let mut out = image.clone();
    dilate_mut(&mut out, norm, k);
    out
}

/// Sets all pixels within distance `k` of a foreground pixel to white.
///
/// A pixel is treated as belonging to the foreground if it has non-zero intensity.
///
/// See the [`dilate`](fn.dilate.html) documentation for examples.
pub fn dilate_mut(image: &mut GrayImage, norm: Norm, k: u8) {
    distance_transform_mut(image, norm);
    for p in image.iter_mut() {
        *p = if *p <= k { 255 } else { 0 };
    }
}

/// Returns an image showing the distance of each pixel from a foreground pixel in the original image.
///
/// A pixel belongs to the foreground if it has non-zero intensity. As the image
/// has a bit-depth of 8, distances saturate at 255.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{distance_transform, Norm};
///
/// let image = GrayImage::from_raw(5, 5, vec![
///     0,   0,   0,   0,   0,
///     0,   0,   0,   0,   0,
///     0,   0,   1,   0,   0,
///     0,   0,   0,   0,   0,
///     0,   0,   0,   0,   0
/// ]).unwrap();
///
/// // L1 norm
/// let l1_distances = GrayImage::from_raw(5, 5, vec![
///     4,   3,   2,   3,   4,
///     3,   2,   1,   2,   3,
///     2,   1,   0,   1,   2,
///     3,   2,   1,   2,   3,
///     4,   3,   2,   3,   4
/// ]).unwrap();
/// assert_pixels_eq!(distance_transform(&image, Norm::L1), l1_distances);
///
/// // LInf norm
/// let linf_distances = GrayImage::from_raw(5, 5, vec![
///     2,   2,   2,   2,   2,
///     2,   1,   1,   1,   2,
///     2,   1,   0,   1,   2,
///     2,   1,   1,   1,   2,
///     2,   2,   2,   2,   2
/// ]).unwrap();
///
/// assert_pixels_eq!(distance_transform(&image, Norm::LInf), linf_distances);
/// # }
/// ```
pub fn distance_transform(image: &GrayImage, norm: Norm) -> GrayImage {
    let mut out = image.clone();
    distance_transform_mut(&mut out, norm);
    out
}

/// Updates an image in place so that each pixel contains its distance from a foreground pixel in the original image.
///
/// A pixel belongs to the foreground if it has non-zero intensity. As the image has a bit-depth of 8,
/// distances saturate at 255.
///
/// See the [`distance_transform`](fn.distance_transform.html) documentation for examples.
pub fn distance_transform_mut(image: &mut GrayImage, norm: Norm) {
    let max_distance = Luma([min(image.width() + image.height(), 255u32) as u8]);

    unsafe {
        // Top-left to bottom-right
        for y in 0..image.height() {
            for x in 0..image.width() {
                if image.unsafe_get_pixel(x, y)[0] > 0u8 {
                    image.unsafe_put_pixel(x, y, Luma([0u8]));
                    continue;
                }

                image.unsafe_put_pixel(x, y, max_distance);

                if x > 0 {
                    check(image, x, y, x - 1, y);
                }

                if y > 0 {
                    check(image, x, y, x, y - 1);

                    if norm == Norm::LInf {
                        if x > 0 {
                            check(image, x, y, x - 1, y - 1);
                        }
                        if x < image.width() - 1 {
                            check(image, x, y, x + 1, y - 1);
                        }
                    }
                }
            }
        }

        // Bottom-right to top-left
        for y in (0..image.height()).rev() {
            for x in (0..image.width()).rev() {
                if x < image.width() - 1 {
                    check(image, x, y, x + 1, y);
                }

                if y < image.height() - 1 {
                    check(image, x, y, x, y + 1);

                    if norm == Norm::LInf {
                        if x < image.width() - 1 {
                            check(image, x, y, x + 1, y + 1);
                        }
                        if x > 0 {
                            check(image, x, y, x - 1, y + 1);
                        }
                    }
                }
            }
        }
    }
}

// Sets image[current_x, current_y] to min(image[current_x, current_y], image[candidate_x, candidate_y] + 1).
// We avoid overflow by performing the arithmetic at type u16. We could use u8::saturating_add instead, but
// (based on the benchmarks tests) this appears to be considerably slower.
unsafe fn check(image: &mut GrayImage, current_x: u32, current_y: u32, candidate_x: u32, candidate_y: u32) {
    let current = image.unsafe_get_pixel(current_x, current_y)[0] as u16;
    let candidate_incr = image.unsafe_get_pixel(candidate_x, candidate_y)[0] as u16 + 1;
    if candidate_incr < current {
        image.unsafe_put_pixel(current_x, current_y, Luma([candidate_incr as u8]));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use image::{GrayImage, Luma};
    use test::*;
    use std::cmp::max;

    #[test]
    fn test_distance_transform_saturation() {
        // A single foreground pixel in the top-right
        let image = GrayImage::from_fn(5, 5, |x, y| {
            match (x, y) {
                (0, 0) => Luma([255u8]),
                _ => Luma([0u8])
            }
        });

        // Distances should not overflow
        let expected = GrayImage::from_fn(5, 5, |x, y| {
            Luma([min(255, max(x, y)) as u8])
        });

        let distances = distance_transform(&image, Norm::LInf);
        assert_pixels_eq!(distances, expected);
    }

    #[test]
    fn test_dilate_point_l1_1() {
        let image = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0, 255,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0
        ]).unwrap();
        let dilated = dilate(&image, Norm::L1, 1);

        let expected = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0,   0, 255,   0,   0,
              0, 255, 255, 255,   0,
              0,   0, 255,   0,   0,
              0,   0,   0,   0,   0
        ]).unwrap();

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l1_2() {
        let image = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0, 255,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0
        ]).unwrap();
        let dilated = dilate(&image, Norm::L1, 2);

        let expected = GrayImage::from_raw(5, 5, vec![
              0,   0, 255,   0,   0,
              0, 255, 255, 255,   0,
            255, 255, 255, 255, 255,
              0, 255, 255, 255,   0,
              0,   0, 255,   0,   0
        ]).unwrap();

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_1() {
        let image = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0, 255,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0
        ]).unwrap();
        let dilated = dilate(&image, Norm::LInf, 1);

        let expected = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0, 255, 255, 255,   0,
              0, 255, 255, 255,   0,
              0, 255, 255, 255,   0,
              0,   0,   0,   0,   0
        ]).unwrap();

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_2() {
        let image = GrayImage::from_raw(5, 5, vec![
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,
              0,   0, 255,   0,   0,
              0,   0,   0,   0,   0,
              0,   0,   0,   0,   0
        ]).unwrap();
        let dilated = dilate(&image, Norm::LInf, 2);

        let expected = GrayImage::from_raw(5, 5, vec![
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255,
            255, 255, 255, 255, 255
        ]).unwrap();

        assert_pixels_eq!(dilated, expected);
    }

    fn square() -> GrayImage {
        GrayImage::from_fn(500, 500, |x, y|{
            if min(x, y) > 100 && max(x, y) < 300  {
                Luma([255u8])
            } else {
                Luma([0u8])
            }
        })
    }

    #[bench]
    fn bench_dilate_l1_5(b: &mut Bencher) {
        let image = square();
        b.iter(|| {
            let dilated = dilate(&image, Norm::L1, 5);
            black_box(dilated);
        })
    }

    #[bench]
    fn bench_dilate_linf_5(b: &mut Bencher) {
        let image = square();
        b.iter(|| {
            let dilated = dilate(&image, Norm::LInf, 5);
            black_box(dilated);
        })
    }
}
