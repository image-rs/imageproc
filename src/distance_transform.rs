//! Functions for computing distance transforms - the distance of each pixel in an
//! image from the nearest pixel of interest.

use std::cmp::min;
use std::u8;
use image::{GenericImage, GrayImage, Luma};

/// How to measure distance between coordinates.
/// See the [`distance_transform`](fn.distance_transform.html) documentation for examples.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Defines d((x1, y1), (x2, y2)) to be abs(x1 - x2) + abs(y1 - y2).
    /// Also known as the Manhattan or city block norm.
    L1,
    /// Defines d((x1, y1), (x2, y2)) to be max(abs(x1 - x2), abs(y1 - y2)).
    /// Also known as the chessboard norm.
    LInf,
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
/// use imageproc::distance_transform::{distance_transform, Norm};
///
/// let image = gray_image!(
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0,   1,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0
/// );
///
/// // L1 norm
/// let l1_distances = gray_image!(
///     4,   3,   2,   3,   4;
///     3,   2,   1,   2,   3;
///     2,   1,   0,   1,   2;
///     3,   2,   1,   2,   3;
///     4,   3,   2,   3,   4
/// );
///
/// assert_pixels_eq!(distance_transform(&image, Norm::L1), l1_distances);
///
/// // LInf norm
/// let linf_distances = gray_image!(
///     2,   2,   2,   2,   2;
///     2,   1,   1,   1,   2;
///     2,   1,   0,   1,   2;
///     2,   1,   1,   1,   2;
///     2,   2,   2,   2,   2
/// );
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
    distance_transform_impl(image, norm, DistanceFrom::Foreground);
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub(crate) enum DistanceFrom {
    Foreground,
    Background,
}

pub(crate) fn distance_transform_impl(image: &mut GrayImage, norm: Norm, from: DistanceFrom) {
    let max_distance = Luma([min(image.width() + image.height(), 255u32) as u8]);

    unsafe {
        // Top-left to bottom-right
        for y in 0..image.height() {
            for x in 0..image.width() {
                if from == DistanceFrom::Foreground {
                    if image.unsafe_get_pixel(x, y)[0] > 0u8 {
                        image.unsafe_put_pixel(x, y, Luma([0u8]));
                        continue;
                    }
                } else {
                    if image.unsafe_get_pixel(x, y)[0] == 0u8 {
                        image.unsafe_put_pixel(x, y, Luma([0u8]));
                        continue;
                    }
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
unsafe fn check(
    image: &mut GrayImage,
    current_x: u32,
    current_y: u32,
    candidate_x: u32,
    candidate_y: u32,
) {
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
    use std::cmp::max;

    #[test]
    fn test_distance_transform_saturation() {
        // A single foreground pixel in the top-left
        let image = GrayImage::from_fn(300, 300, |x, y| match (x, y) {
            (0, 0) => Luma([255u8]),
            _ => Luma([0u8]),
        });

        // Distances should not overflow
        let expected = GrayImage::from_fn(300, 300, |x, y| Luma([min(255, max(x, y)) as u8]));

        let distances = distance_transform(&image, Norm::LInf);
        assert_pixels_eq!(distances, expected);
    }
}
