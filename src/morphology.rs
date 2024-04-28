//! Functions for computing [morphological operators].
//!
//! [morphological operators]: https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

use crate::distance_transform::{
    distance_transform_impl, distance_transform_mut, DistanceFrom, Norm,
};
use image::{GenericImageView, GrayImage, Luma};
use num::pow::Pow;

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
/// use imageproc::morphology::dilate;
/// use imageproc::distance_transform::Norm;
///
/// let image = gray_image!(
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0, 255,   0,   0;
///     0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0
/// );
///
/// // L1 norm
/// let l1_dilated = gray_image!(
///     0,   0,   0,   0,   0;
///     0,   0, 255,   0,   0;
///     0, 255, 255, 255,   0;
///     0,   0, 255,   0,   0;
///     0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(dilate(&image, Norm::L1, 1), l1_dilated);
///
/// // L2 norm
/// // (note that L2 behaves identically to L1 for distances of 2 or less)
/// let l2_dilated = gray_image!(
///    0,   0,   0,   0,   0;
///    0,   0, 255,   0,   0;
///    0, 255, 255, 255,   0;
///    0,   0, 255,   0,   0;
///    0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(dilate(&image, Norm::L2, 1), l2_dilated);
///
/// // LInf norm
/// let linf_dilated = gray_image!(
///    0,   0,   0,   0,   0;
///    0, 255, 255, 255,   0;
///    0, 255, 255, 255,   0;
///    0, 255, 255, 255,   0;
///    0,   0,   0,   0,   0
/// );
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

/// Sets all pixels within distance `k` of a background pixel to black.
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
/// use imageproc::morphology::erode;
/// use imageproc::distance_transform::Norm;
///
/// let image = gray_image!(
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0, 255, 255, 255,   0, 255, 255, 255,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0, 255, 255, 255, 255, 255, 255, 255,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0
/// );
///
/// // L1 norm - the outermost foreground pixels are eroded,
/// // as well as those horizontally and vertically adjacent
/// // to the centre background pixel.
/// let l1_eroded = gray_image!(
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0, 255, 255,   0, 255, 255,   0,  0;
///     0,   0, 255,   0,   0,   0, 255,   0,  0;
///     0,   0, 255, 255,   0, 255, 255,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0
/// );
///
/// assert_pixels_eq!(erode(&image, Norm::L1, 1), l1_eroded);
///
/// // L2 norm - all foreground pixels within a distance of n or less
/// // of a background pixel are eroded.
/// // (note that L2 behaves identically to L1 for distances of 2 or less)
/// let l2_eroded = gray_image!(
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0, 255, 255,   0, 255, 255,   0,  0;
///     0,   0, 255,   0,   0,   0, 255,   0,  0;
///     0,   0, 255, 255,   0, 255, 255,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0
/// );
///
/// assert_pixels_eq!(erode(&image, Norm::L2, 1), l2_eroded);
///
/// // LInf norm - all pixels eroded using the L1 norm are eroded,
/// // as well as the pixels diagonally adjacent to the centre pixel.
/// let linf_eroded = gray_image!(
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0, 255,   0,   0,   0, 255,   0,  0;
///     0,   0, 255,   0,   0,   0, 255,   0,  0;
///     0,   0, 255,   0,   0,   0, 255,   0,  0;
///     0,   0, 255, 255, 255, 255, 255,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0;
///     0,   0,   0,   0,   0,   0,   0,   0,  0
/// );
///
/// assert_pixels_eq!(erode(&image, Norm::LInf, 1), linf_eroded);
/// # }
/// ```
pub fn erode(image: &GrayImage, norm: Norm, k: u8) -> GrayImage {
    let mut out = image.clone();
    erode_mut(&mut out, norm, k);
    out
}

/// Sets all pixels within distance `k` of a background pixel to black.
///
/// A pixel is treated as belonging to the foreground if it has non-zero intensity.
///
/// See the [`erode`](fn.erode.html) documentation for examples.
pub fn erode_mut(image: &mut GrayImage, norm: Norm, k: u8) {
    distance_transform_impl(image, norm, DistanceFrom::Background);
    for p in image.iter_mut() {
        *p = if *p <= k { 0 } else { 255 };
    }
}

/// Erosion followed by dilation.
///
/// See the [`erode`](fn.erode.html) and [`dilate`](fn.dilate.html)
/// documentation for definitions of dilation and erosion.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::morphology::open;
/// use imageproc::distance_transform::Norm;
///
/// // Isolated regions of foreground pixels are removed.
/// let cross = gray_image!(
///       0,   0,   0,   0,   0;
///       0,   0, 255,   0,   0;
///       0, 255, 255, 255,   0;
///       0,   0, 255,   0,   0;
///       0,   0,   0,   0,   0
/// );
///
/// let opened_cross = gray_image!(
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(
///     open(&cross, Norm::LInf, 1),
///     opened_cross
/// );
///
/// // Large blocks survive unchanged.
/// let blob = gray_image!(
///       0,   0,   0,   0,   0;
///       0, 255, 255, 255,   0;
///       0, 255, 255, 255,   0;
///       0, 255, 255, 255,   0;
///       0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(
///     open(&blob, Norm::LInf, 1),
///     blob
/// );
/// # }
/// ```
pub fn open(image: &GrayImage, norm: Norm, k: u8) -> GrayImage {
    let mut out = image.clone();
    open_mut(&mut out, norm, k);
    out
}

/// Erosion followed by dilation.
///
/// See the [`open`](fn.open.html) documentation for examples,
/// and the [`erode`](fn.erode.html) and [`dilate`](fn.dilate.html)
/// documentation for definitions of dilation and erosion.
pub fn open_mut(image: &mut GrayImage, norm: Norm, k: u8) {
    erode_mut(image, norm, k);
    dilate_mut(image, norm, k);
}

/// Dilation followed by erosion.
///
/// See the [`erode`](fn.erode.html) and [`dilate`](fn.dilate.html)
/// documentation for definitions of dilation and erosion.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::morphology::close;
/// use imageproc::distance_transform::Norm;
///
/// // Small holes are closed - hence the name.
/// let small_hole = gray_image!(
///     255, 255, 255, 255;
///     255,   0,   0, 255;
///     255,   0,   0, 255;
///     255, 255, 255, 255
/// );
///
/// let closed_small_hole = gray_image!(
///     255, 255, 255, 255;
///     255, 255, 255, 255;
///     255, 255, 255, 255;
///     255, 255, 255, 255
/// );
///
/// assert_pixels_eq!(
///     close(&small_hole, Norm::LInf, 1),
///     closed_small_hole
/// );
///
/// // Large holes survive unchanged.
/// let large_hole = gray_image!(
///     255, 255, 255, 255, 255;
///     255,   0,   0,   0, 255;
///     255,   0,   0,   0, 255;
///     255,   0,   0,   0, 255;
///     255, 255, 255, 255, 255
/// );
///
/// assert_pixels_eq!(
///     close(&large_hole, Norm::LInf, 1),
///     large_hole
/// );
///
/// // A dot gains a layer of foreground pixels
/// // when dilated and loses them again when eroded,
/// // resulting in no change.
/// let dot = gray_image!(
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0, 255,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(
///     close(&dot, Norm::LInf, 1),
///     dot
/// );
///
/// // A dot near the boundary gains pixels in the top-left
/// // of the image which are not within distance 1 of any
/// // background pixels, so are not removed by erosion.
/// let dot_near_boundary = gray_image!(
///       0,   0,   0,   0,   0;
///       0, 255,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0
/// );
///
/// let closed_dot_near_boundary = gray_image!(
///     255, 255,   0,   0,   0;
///     255, 255,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0;
///       0,   0,   0,   0,   0
/// );
///
/// assert_pixels_eq!(
///     close(&dot_near_boundary, Norm::LInf, 1),
///     closed_dot_near_boundary
/// );
/// # }
/// ```
pub fn close(image: &GrayImage, norm: Norm, k: u8) -> GrayImage {
    let mut out = image.clone();
    close_mut(&mut out, norm, k);
    out
}

/// Dilation followed by erosion.
///
/// See the [`close`](fn.close.html) documentation for examples,
/// and the [`erode`](fn.erode.html) and [`dilate`](fn.dilate.html)
/// documentation for definitions of dilation and erosion.
pub fn close_mut(image: &mut GrayImage, norm: Norm, k: u8) {
    dilate_mut(image, norm, k);
    erode_mut(image, norm, k);
}

/// A struct representing a mask used in morphological operations
///
/// the mask is represented by a list of the positions of its pixels
/// relative to its center, with a maximum distance of 255
/// along each axis.
/// This means that in the most extreme case, the mask could have
/// a size of 513 by 513 pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    elements: Vec<(i16, i16)>,
}

impl Mask {
    /// todo!()
    pub fn square(radius: u8) -> Self {
        Self {
            elements: (-(radius as i16)..=(radius as i16))
                .flat_map(|x| (-(radius as i16)..=(radius as i16)).map(move |y| (x, y)))
                .collect(),
        }
    }

    /// todo!()
    pub fn diamond(radius: u8) -> Self {
        Self {
            elements: (-(radius as i16)..=(radius as i16))
                .flat_map(|x| {
                    ((x.abs() - radius as i16)..=(radius as i16 - x.abs())).map(move |y| (x, y))
                })
                .collect(),
        }
    }

    /// todo!()
    pub fn disk(radius: u8) -> Self {
        Self {
            elements: (-(radius as i16)..=(radius as i16))
                .flat_map(|x| {
                    (-(radius as i16)..=(radius as i16))
                        .filter(move |&y| (x.abs() as u32).pow(2) + (y.abs() as u32).pow(2) <= (radius as u32).pow(2) )
                        .map(move |y| (x, y))
                })
                .collect(),
        }
    }

    /// todo!()
    pub fn from_image(image: &GrayImage, center_x: u8, center_y: u8) -> Self {
        assert!(
            (image.width() - center_x as u32) < (u8::MAX as u32),
            "all pixels of the mask must be at most {} pixels from the center",
            u8::MAX
        );
        assert!(
            (image.height() - center_y as u32) < (u8::MAX as u32),
            "all pixels of the mask must be at most {} pixels from the center",
            u8::MAX
        );
        Self {
            elements: (0..image.width() as i16)
                .flat_map(|x| {
                    (0..(image.height() as i16))
                        .filter(move |&y| image.get_pixel(x as u32, y as u32).0[0] != 0)
                        .map(move |y| (x - center_x as i16, y - center_y as i16))
                })
                .collect(),
        }
    }

    fn apply<'a, 'b: 'a, 'c: 'a>(
        &'c self,
        image: &'b GrayImage,
        x: u32,
        y: u32,
    ) -> impl Iterator<Item = &'a Luma<u8>> {
        self.elements
            .iter()
            .map(move |(i, j)| (x as i64 + *i as i64, y as i64 + *j as i64))
            .filter(move |(i, j)| {
                0 < *i && *i < image.width() as i64 && 0 < *j && *j < image.height() as i64
            })
            .map(move |(i, j)| image.get_pixel(i as u32, j as u32))
    }
}

/// todo!()
pub fn grayscale_dilate(image: &GrayImage, mask: &Mask) -> GrayImage {
    #[cfg(feature = "rayon")]
    let result = GrayImage::from_par_fn(image.width(), image.height(), |x, y| {
        Luma([mask
            .apply(image, x, y)
            .map(|l| l.0[0])
            .max()
            .unwrap_or(u8::MAX)])
    });
    #[cfg(not(feature = "rayon"))]
    let result = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        Luma([mask
            .apply(image, x, y)
            .map(|l| l.0[0])
            .max()
            .unwrap_or(u8::MAX)])
    });
    result
}

/// todo!()
pub fn grayscale_dilate_mut(image: &mut GrayImage, mask: &Mask) {
    let dilated = grayscale_dilate(image, mask);
    image
        .iter_mut()
        .zip(dilated.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// todo!()
pub fn grayscale_erode(image: &GrayImage, mask: &Mask) -> GrayImage {
    #[cfg(feature = "rayon")]
    let result = GrayImage::from_par_fn(image.width(), image.height(), |x, y| {
        Luma([mask
            .apply(image, x, y)
            .map(|l| l.0[0])
            .min()
            .unwrap_or(u8::MAX)])
    });
    #[cfg(not(feature = "rayon"))]
    let result = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        Luma([mask
            .apply(image, x, y)
            .map(|l| l.0[0])
            .min()
            .unwrap_or(u8::MAX)])
    });
    result
}

/// todo!()
pub fn grayscale_erode_mut(image: &mut GrayImage, mask: &Mask) {
    let dilated = grayscale_dilate(image, mask);
    image
        .iter_mut()
        .zip(dilated.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// todo!()
pub fn grayscale_open(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_dilate(&grayscale_erode(image, mask), mask)
}

/// todo!()
pub fn grayscale_open_mut(image: &mut GrayImage, mask: &Mask) {
    let opened = grayscale_open(image, mask);
    image
        .iter_mut()
        .zip(opened.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// todo!()
pub fn grayscale_close(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_erode(&grayscale_dilate(image, mask), mask)
}

/// todo!()
pub fn grayscale_close_mut(image: &mut GrayImage, mask: &Mask) {
    let closed = grayscale_erode(&grayscale_dilate(image, mask), mask);
    image
        .iter_mut()
        .zip(closed.iter())
        .for_each(|(dst, src)| *dst = *src);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::test::*;
    use image::{GrayImage, Luma};
    use std::cmp::{max, min};

    #[test]
    fn test_dilate_point_l1_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L1, 0);

        let expected = image;

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l1_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L1, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0, 255, 255, 255,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l1_2() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L1, 2);

        let expected = gray_image!(
              0,   0, 255,   0,   0;
              0, 255, 255, 255,   0;
            255, 255, 255, 255, 255;
              0, 255, 255, 255,   0;
              0,   0, 255,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l1_4() {
        let image = gray_image!(
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0, 255,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0;
              0,   0,   0,   0,   0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L1, 4);

        let expected = gray_image!(
              0,   0,   0,   0, 255,   0,   0,   0,   0;
              0,   0,   0, 255, 255, 255,   0,   0,   0;
              0,   0, 255, 255, 255, 255, 255,   0,   0;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
              0,   0, 255, 255, 255, 255, 255,   0,   0;
              0,   0,   0, 255, 255, 255,   0,   0,   0;
              0,   0,   0,   0, 255,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l2_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L2, 0);

        let expected = image;

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l2_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L2, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0, 255, 255, 255,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l2_2() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L2, 2);

        let expected = gray_image!(
              0,   0, 255,   0,   0;
              0, 255, 255, 255,   0;
            255, 255, 255, 255, 255;
              0, 255, 255, 255,   0;
              0,   0, 255,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_l2_4() {
        let image = gray_image!(
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0, 255,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::L2, 4);

        let expected = gray_image!(
              0,   0,   0,   0, 255,   0,   0,   0,   0;
              0,   0, 255, 255, 255, 255, 255,   0,   0;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
              0, 255, 255, 255, 255, 255, 255, 255,   0;
              0,   0, 255, 255, 255, 255, 255,   0,   0;
              0,   0,   0,   0, 255,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::LInf, 0);

        let expected = image;

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::LInf, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0, 255, 255, 255,   0;
              0, 255, 255, 255,   0;
              0, 255, 255, 255,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_2() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::LInf, 2);

        let expected = gray_image!(
            255, 255, 255, 255, 255;
            255, 255, 255, 255, 255;
            255, 255, 255, 255, 255;
            255, 255, 255, 255, 255;
            255, 255, 255, 255, 255
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_dilate_point_linf_4() {
        let image = gray_image!(
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0, 255,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0;
            0,   0,   0,   0,   0,   0,   0,   0,   0
        );
        let dilated = dilate(&image, Norm::LInf, 4);

        let expected = gray_image!(
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255;
            255, 255, 255, 255, 255, 255, 255, 255, 255
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_erode_point_l1_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::L1, 0);

        let expected = image;

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_point_l1_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::L1, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_dented_wall_l1_4() {
        let image = gray_image!(
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255,   0,   0,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0
        );
        let dilated = erode(&image, Norm::L1, 4);

        let expected = gray_image!(
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_erode_point_l2_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::L2, 0);

        let expected = image;

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_point_l2_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::L2, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_dented_wall_l2_4() {
        let image = gray_image!(
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255,   0,   0,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0
        );
        let dilated = erode(&image, Norm::L2, 4);

        let expected = gray_image!(
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255, 255,   0,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0;
            255, 255, 255, 255,   0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    #[test]
    fn test_erode_point_linf_0() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::LInf, 0);

        let expected = image;

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_point_linf_1() {
        let image = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0, 255,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );
        let eroded = erode(&image, Norm::LInf, 1);

        let expected = gray_image!(
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0;
              0,   0,   0,   0,   0
        );

        assert_pixels_eq!(eroded, expected);
    }

    #[test]
    fn test_erode_dented_wall_linf_4() {
        let image = gray_image!(
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255,   0,   0,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0;
            255, 255, 255, 255, 255, 255, 255, 255,   0
        );
        let dilated = erode(&image, Norm::LInf, 4);

        let expected = gray_image!(
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0;
            255, 255,   0,   0,   0,   0,   0,   0,   0
        );

        assert_pixels_eq!(dilated, expected);
    }

    fn square() -> GrayImage {
        GrayImage::from_fn(500, 500, |x, y| {
            if min(x, y) > 100 && max(x, y) < 300 {
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
    fn bench_dilate_l2_5(b: &mut Bencher) {
        let image = square();
        b.iter(|| {
            let dilated = dilate(&image, Norm::L2, 5);
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
