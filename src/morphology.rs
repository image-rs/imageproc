//! Functions for computing [morphological operators].
//!
//! [morphological operators]: https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

use crate::distance_transform::{
    distance_transform_impl, distance_transform_mut, DistanceFrom, Norm,
};
use image::{GenericImageView, GrayImage, Luma};
use itertools::Itertools;
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
    /// creates a mask from a grayscale image
    ///
    /// a pixel is part of the mask if and only if it is non-zero
    ///
    /// `center_x` and `center_y` define the coordinates of the center of the mask.
    /// They may take any value, including outside of the bounds of the input image,
    /// but all pixels of the mask must be at most 255 pixels away from the center.
    ///
    /// For example, if `center_x` was 10 and center `center_y` was 40,
    /// the width of the image would have to be at most 266, and its height at most 296
    ///
    /// # Panics
    /// if there is a pixel which is 256 pixels or more away from the center along either direction
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    ///
    /// let ring_mask_base = gray_image!(
    ///     100,  75, 255, 222;
    ///      84,   0,   0,   1;
    ///      99,   0,   0,  22;
    ///     255,   7, 255,  20
    /// );
    ///
    /// let ring_mask_inside = Mask::from_image(&ring_mask_base, 1, 1);
    ///
    /// // two different images with identical zeroes will create the same mask
    /// let other_ring_mask_base = gray_image!(
    ///      10, 172,  13,   5;
    ///      45,   0,   0, 101;
    ///     222,   0,   0,  93;
    ///       1,   7, 212,  35
    /// );
    ///
    /// assert_eq!(ring_mask_inside, Mask::from_image(&other_ring_mask_base, 1, 1));
    ///
    /// // using two identical images with different centers will usually make different masks
    /// assert_ne!(Mask::from_image(&ring_mask_base, 1, 1), Mask::from_image(&ring_mask_base, 2, 2));
    ///
    /// // the center may be out of the image bound
    /// let ring_mask_outside = Mask::from_image(&ring_mask_base, 8, 8);
    ///
    /// // but all pixels must be at most 255 pixels away from the center
    /// // the code below will panic :
    /// // let some_mask = Mask::from_image(&GrayImage::new(300, 300), 2, 2);
    /// // this one won't :
    /// // let some_mask = Mask::from_image(&GrayImage::new(300, 300), 200, 200);
    /// # }
    /// ```
    pub fn from_image(image: &GrayImage, center_x: u8, center_y: u8) -> Self {
        assert!(
            (image.width() as i64 - center_x as i64) < (u8::MAX as i64),
            "all pixels of the mask must be at most 255 pixels from the center"
        );
        assert!(
            (image.height() as i64 - center_y as i64) < (u8::MAX as i64),
            "all pixels of the mask must be at most 255 pixels from the center"
        );
        Self {
            elements: (0..image.width() as i16)
                .cartesian_product(0..(image.height() as i16))
                .filter(|(x, y)| image.get_pixel(*x as u32, *y as u32).0[0] != 0)
                .map(|(x, y)| (x - center_x as i16, y - center_y as i16))
                .collect(),
        }
    }

    /// creates a square-shaped mask
    ///
    /// the mask contains exactly all the pixels `radius` pixels or less
    /// away from the center according to the `Linf` norm.
    ///
    /// therefore, `square(0)` will make a 1x1 square, `square(1)` a 3x3 square, etc...
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// let single_pixel = gray_image!(100);
    ///
    /// assert_eq!(Mask::square(0), Mask::from_image(&single_pixel, 0, 0));
    ///
    /// let three_by_three_mask_base = gray_image!(
    ///     100, 154, 222;
    ///     184, 184, 211;
    ///     255, 127, 255
    /// );
    ///
    /// assert_eq!(Mask::square(1), Mask::from_image(&three_by_three_mask_base, 1, 1));
    /// # }
    /// ```
    pub fn square(radius: u8) -> Self {
        let range = -(radius as i16)..=(radius as i16);
        Self {
            elements: range.clone().cartesian_product(range).collect(),
        }
    }

    /// creates a diamond-shaped mask
    ///
    /// the mask contains exactly all the pixels `radius` pixels or less
    /// away from the center according to the `L1` norm.
    ///
    /// therefore, `diamond(0)` will make a 1x1 square, `diamond(1)` a 3x3 cross,
    /// `diamond(2)` a 5x5 diamond, `diamond(3)` a 7x7 diamond, etc...
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// let single_pixel = gray_image!(100);
    ///
    /// assert_eq!(Mask::diamond(0), Mask::from_image(&single_pixel, 0, 0));
    ///
    /// let three_by_three_mask_base = gray_image!(
    ///       0, 255,   0;
    ///      84, 204, 101;
    ///       0, 217,   0
    /// );
    ///
    /// assert_eq!(Mask::diamond(1), Mask::from_image(&three_by_three_mask_base, 1, 1));
    ///
    /// let five_by_five_mask_base = gray_image!(
    ///       0,   0, 255,   0,   0;
    ///       0, 231, 204, 101,   0;
    ///     149, 193, 188, 137, 199;
    ///       0, 222, 182, 114,   0;
    ///       0,   0, 217,   0,   0
    /// );
    ///
    /// assert_eq!(Mask::diamond(2), Mask::from_image(&five_by_five_mask_base, 2, 2));
    /// # }
    /// ```
    pub fn diamond(radius: u8) -> Self {
        Self {
            elements: (-(radius as i16)..=(radius as i16))
                .flat_map(|x| {
                    ((x.abs() - radius as i16)..=(radius as i16 - x.abs())).map(move |y| (x, y))
                })
                .collect(),
        }
    }

    /// creates a disk-shaped mask
    ///
    /// the mask contains exactly all the pixels `radius` pixels or less
    /// away from the center according to the `L2` norm.
    ///
    /// When computing distances using the L2 norm we take the ceiling of the true values.
    /// This means that using the L2 norm gives the same results as the L1 norm for `radius <= 2`.
    ///
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// let single_pixel = gray_image!(100);
    ///
    /// assert_eq!(Mask::disk(0), Mask::from_image(&single_pixel, 0, 0));
    ///
    /// let three_by_three_mask_base = gray_image!(
    ///       0, 255,   0;
    ///      84, 204, 101;
    ///       0, 217,   0
    /// );
    ///
    /// assert_eq!(Mask::disk(1), Mask::from_image(&three_by_three_mask_base, 1, 1));
    ///
    /// let five_by_five_mask_base = gray_image!(
    ///       0,   0, 255,   0,   0;
    ///       0, 231, 204, 101,   0;
    ///     149, 193, 188, 137, 199;
    ///       0, 222, 182, 114,   0;
    ///       0,   0, 217,   0,   0
    /// );
    ///
    /// assert_eq!(Mask::disk(2), Mask::from_image(&five_by_five_mask_base, 2, 2));
    ///
    /// // disk() finally separates from diamond() at a radius of 3
    ///
    /// let seven_by_seven_mask_base = gray_image!(
    ///       0,   0,   0, 255,   0,   0,   0;
    ///       0, 217, 188, 101, 222, 137,   0;
    ///       0, 231, 204, 255, 182, 193,   0;
    ///     149, 193, 101, 188, 217, 149, 114;
    ///       0, 217, 188, 231, 222, 137,   0;
    ///       0, 101, 204, 222, 255, 193,   0;
    ///       0,   0,   0, 182,   0,   0,   0
    /// );
    ///
    /// assert_eq!(Mask::disk(3), Mask::from_image(&seven_by_seven_mask_base, 3, 3));
    ///
    /// # }
    /// ```
    pub fn disk(radius: u8) -> Self {
        let range = -(radius as i16)..=(radius as i16);
        Self {
            elements: range
                .clone()
                .cartesian_product(range)
                .filter(|(x, y)| {
                    (x.unsigned_abs() as u32).pow(2) + (y.unsigned_abs() as u32).pow(2)
                        <= (radius as u32).pow(2)
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
                0 <= *i && *i < image.width() as i64 && 0 <= *j && *j < image.height() as i64
            })
            .map(move |(i, j)| image.get_pixel(i as u32, j as u32))
    }
}

/// computes the morphologic dilation of the input image with the given mask
///
/// for each input pixel, the output pixel will be the maximum of all pixels included
/// in the mask at that position.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{Mask, grayscale_dilate};
///
/// let image = gray_image!(
///     7,   0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0,   0;
///     0,   0,  99,   0,   0,   0;
///     0,   0,   0,   0,   0,   0;
///     0,   0,   0,   0,   0, 222
/// );
///
/// // using a diamond mask
/// let diamond_dilated = gray_image!(
///     7,   7,   0,   0,   0,   0;
///     7,   0,  99,   0,   0,   0;
///     0,  99,  99,  99,   0,   0;
///     0,   0,  99,   0,   0, 222;
///     0,   0,   0,   0, 222, 222
/// );
///
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::diamond(1)), diamond_dilated);
///
/// // using a disk mask
/// let disk_dilated = gray_image!(
///    99,  99,  99,  99,  99,   0;
///    99,  99,  99,  99,  99, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99, 222, 222, 222, 222
/// );
///
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::disk(3)), disk_dilated);
///
/// // using a square mask
/// let square_dilated = gray_image!(
///     7,   7,   0,   0,   0,   0;
///     7,  99,  99,  99,   0,   0;
///     0,  99,  99,  99,   0,   0;
///     0,  99,  99,  99, 222, 222;
///     0,   0,   0,   0, 222, 222
/// );
///
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::square(1)), square_dilated);
///
///
/// // using an arbitrary mask
/// // the center of the mask is the '4' here, cause it's coordinates are (0,1)
/// let column_mask = Mask::from_image(
///     &gray_image!(
///         8;
///         1;
///         9;
///         2
///     ),  
///     0,  1
/// );
///
/// let column_dilated = gray_image!(
///     7,   0,  99,   0,   0,   0;
///     7,   0,  99,   0,   0,   0;
///     0,   0,  99,   0,   0, 222;
///     0,   0,  99,   0,   0, 222;
///     0,   0,   0,   0,   0, 222
/// );
///
/// assert_pixels_eq!(grayscale_dilate(&image, &column_mask), column_dilated);
/// # }
/// ```
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

/// applies a morphologic dilation on the image with the given mask
///
/// set each pixel to the maximum of all pixels included in the mask at its position.
///
/// See the [`grayscale_dilate`](fn.grayscale_dilate.html) documentation for examples.
///
/// note that unlike `dilate_mut`, this performs as many allocations as it's non-mut counterpart
/// and is thus neither more efficient nor faster
pub fn grayscale_dilate_mut(image: &mut GrayImage, mask: &Mask) {
    let dilated = grayscale_dilate(image, mask);
    image
        .iter_mut()
        .zip(dilated.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// computes the morphologic erosion of the input image with the given mask
///
/// for each input pixel, the output pixel will be the minimum of all pixels included
/// in the mask at that position.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{Mask, grayscale_erode};
///
/// let image = gray_image!(
///     7,  99,  99,  99,  99, 222;
///    99,  99,  99,  99,  99, 222;
///    99,  99,  99,  99, 222, 222;
///     7,  99,  99,  99, 222, 222;
///    99,  99,  99, 222, 222, 222
/// );
///
/// // using a diamond mask
/// let diamond_eroded = gray_image!(
///     7,   7,  99,  99,  99,  99;
///     7,  99,  99,  99,  99,  99;
///     7,  99,  99,  99,  99, 222;
///     7,   7,  99,  99,  99, 222;
///     7,  99,  99,  99, 222, 222
/// );
///
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::diamond(1)), diamond_eroded);
///
/// // using a disk mask
/// let disk_eroded = gray_image!(
///     7,   7,   7,   7,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,   7,  99,  99;
///     7,   7,   7,  99,  99,  99
/// );
///
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::disk(3)), disk_eroded);
///
/// // using a square mask
/// let square_eroded = gray_image!(
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99, 222;
///     7,   7,  99,  99,  99, 222
/// );
///
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::square(1)), square_eroded);
///
///
/// // using an arbitrary mask
/// // the center of the mask is the '4' here, cause it's coordinates are (0,1)
/// let column_mask = Mask::from_image(
///     &gray_image!(
///         8;
///         4;
///         9;
///         2
///     ),  
///     0,  1
/// );
///
/// let column_eroded = gray_image!(
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99, 222, 222;
///     7,  99,  99,  99, 222, 222
/// );
///
/// assert_pixels_eq!(grayscale_erode(&image, &column_mask), column_eroded);
/// # }
/// ```
pub fn grayscale_erode(image: &GrayImage, mask: &Mask) -> GrayImage {
    #[cfg(feature = "rayon")]
    let result = GrayImage::from_par_fn(image.width(), image.height(), |x, y| {
        Luma([mask.apply(image, x, y).map(|l| l.0[0]).min().unwrap_or(0)])
    });
    #[cfg(not(feature = "rayon"))]
    let result = GrayImage::from_fn(image.width(), image.height(), |x, y| {
        Luma([mask.apply(image, x, y).map(|l| l.0[0]).min().unwrap_or(0)])
    });
    result
}

/// applies a morphologic erosion on the image with the given mask
///
/// set each pixel to the minimum of all pixels included in the mask at its position.
///
/// See the [`grayscale_erode`](fn.grayscale_erode.html) documentation for examples.
///
/// note that unlike `erode_mut`, this performs as many allocations as it's non-mut counterpart
/// and is thus neither more efficient nor faster
pub fn grayscale_erode_mut(image: &mut GrayImage, mask: &Mask) {
    let dilated = grayscale_dilate(image, mask);
    image
        .iter_mut()
        .zip(dilated.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// Grayscale erosion followed by grayscale dilation.
///
/// See the [`grayscale_dilate`](fn.grayscale_dilate.html)
/// and [`grayscale_erode`](fn.grayscale_erode.html)
/// documentation for definitions of dilation and erosion.
///
////// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{Mask, grayscale_open};
///
/// // Isolated regions of foreground pixels are removed,
/// // while isolated zones of background pixels are maintaned
/// let image = gray_image!(
///   100,  99,  99,  99, 222,  99;
///    99,  99,  99, 222, 222, 222;
///    99,   7,  99,  99, 222,  99;
///     7,   7,   7,  99,  99,  99;
///    99,   7,  99,  99,  99,  99
/// );
///
/// // Isolated regions of foreground pixels are removed,
/// // while isolated zones of background are maintaned
/// let image_opened = gray_image!(
///    99,  99,  99,  99,  99,  99;
///    99,  99,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,  99,  99,  99
/// );
///
/// assert_pixels_eq!(grayscale_open(&image, &Mask::square(1)), image_opened);
///
/// // because it is a morhological operator, applying the same
/// // opening a second time doesn't do anything
/// assert_pixels_eq!(grayscale_open(&image_opened, &Mask::square(1)), image_opened);
///
/// // which regions are removed depends on the mask used
/// let image_opened_diamond = gray_image!(
///    99,  99,  99,  99, 222,  99;
///    99,  99,  99, 222, 222, 222;
///    99,   7,  99,  99, 222,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,  99,  99,  99,  99
/// );
///
/// assert_pixels_eq!(grayscale_open(&image, &Mask::diamond(1)), image_opened_diamond);
/// # }
/// ```
pub fn grayscale_open(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_dilate(&grayscale_erode(image, mask), mask)
}

/// Grayscale erosion followed by grayscale dilation.
///
/// See the [`grayscale_open`](fn.grayscale_open.html) documentation for examples,
/// and the [`grayscale_dilate`](fn.grayscale_dilate.html)
/// and [`grayscale_erode`](fn.grayscale_erode.html)
/// documentation for definitions of dilation and erosion.
pub fn grayscale_open_mut(image: &mut GrayImage, mask: &Mask) {
    let opened = grayscale_open(image, mask);
    image
        .iter_mut()
        .zip(opened.iter())
        .for_each(|(dst, src)| *dst = *src);
}

/// Grayscale dilation followed by grayscale erosion.
///
/// See the [`grayscale_dilate`](fn.grayscale_dilate.html)
/// and [`grayscale_erode`](fn.grayscale_erode.html)
/// documentation for definitions of dilation and erosion.
///
////// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::GrayImage;
/// use imageproc::morphology::{Mask, grayscale_close};
///
/// let image = gray_image!(
///    50,  99,  99,  99, 222,  99;
///    99,  99,  99, 222, 222, 222;
///    99,   7,  99,  99, 222,  99;
///     7,   7,   7,  99,  99,  99;
///    99,   7,  99,  99,  99,  99
/// );
///
/// // Isolated regions of background pixels are removed,
/// // while isolated zones of foreground pixels are maintaned
/// let image_closed = gray_image!(
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99,  99, 222, 222;
///    99,  99,  99,  99,  99,  99;
///    99,  99,  99,  99,  99,  99
/// );
///
/// assert_pixels_eq!(grayscale_close(&image, &Mask::square(1)), image_closed);
///
/// // because it is a morphological operator, applying the same
/// // closing a second time doesn't do anything
/// assert_pixels_eq!(grayscale_close(&image_closed, &Mask::square(1)), image_closed);
///
/// // which regions are removed depends on the mask used
/// let image_closed_diamond = gray_image!(
///    99,  99,  99,  99, 222, 222;
///    99,  99,  99, 222, 222, 222;
///    99,   7,  99,  99, 222,  99;
///     7,   7,   7,  99,  99,  99;
///    99,   7,  99,  99,  99,  99
/// );
///
/// assert_pixels_eq!(grayscale_close(&image, &Mask::diamond(1)), image_closed_diamond);
/// # }
/// ```
pub fn grayscale_close(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_erode(&grayscale_dilate(image, mask), mask)
}

/// Grayscale dilation followed by grayscale erosion.
///
/// See the [`grayscale_close`](fn.grayscale_close.html) documentation for examples,
/// and the [`grayscale_erode`](fn.grayscale_erode.html)
/// and [`grayscale_dilate`](fn.grayscale_dilate.html)
/// documentation for definitions of dilation and erosion.
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

    #[test]
    fn test_mask_from_image_equality() {
        let ring_mask_base = gray_image!(
             100,  75, 255, 222;
              84,   0,   0,   1;
              99,   0,   0,  22;
             255,   7, 255,  20
        );

        let other_ring_mask_base = gray_image!(
             18, 172,  13,   5;
             45,   0,   0, 101;
            222,   0,   0,  93;
              1,   9, 212,  35
        );

        assert_eq!(
            Mask::from_image(&ring_mask_base, 1, 1),
            Mask::from_image(&other_ring_mask_base, 1, 1)
        );
    }

    #[test]
    fn test_mask_from_image_displacement_inequality() {
        let mask_base = gray_image!(
             100,  75, 255, 222;
              84,   0,   0,   1;
              99,   0,   0,  22;
             255,   7, 255,  20
        );

        assert_ne!(
            Mask::from_image(&mask_base, 1, 1),
            Mask::from_image(&mask_base, 2, 2)
        );
    }

    #[test]
    fn test_mask_from_image_empty() {
        let mask_base = gray_image!(0);
        assert!(Mask::from_image(&mask_base, 1, 1).elements.is_empty())
    }

    /// this tests that it doesn't panic
    #[test]
    fn test_mask_from_image_outside() {
        let mask_base = gray_image!(
             100,  75, 255, 222;
              84,   0,   0,   1;
              99,   0,   0,  22;
             255,   7, 255,  20
        );
        let _ = Mask::from_image(&mask_base, 20, 20);
    }

    #[test]
    #[should_panic]
    fn test_mask_from_image_out_of_bounds() {
        let mask_base = GrayImage::new(300, 300);
        Mask::from_image(&mask_base, 5, 5);
    }

    #[test]
    fn test_masks_0() {
        let mask_base = gray_image!(72);
        assert_eq!(Mask::from_image(&mask_base, 0, 0), Mask::square(0));
        assert_eq!(Mask::from_image(&mask_base, 0, 0), Mask::diamond(0));
        assert_eq!(Mask::from_image(&mask_base, 0, 0), Mask::disk(0))
    }

    #[test]
    fn test_mask_square_1() {
        let mask_base = gray_image!(
            72,  31, 148;
             2, 219, 173;
            48,   7, 200
        );
        assert_eq!(Mask::from_image(&mask_base, 1, 1), Mask::square(1));
    }

    #[test]
    fn test_mask_square_2() {
        let mask_base = gray_image!(
            217, 188, 101, 222, 137;
            231, 204, 255, 182, 193;
            193, 101, 188, 217, 149;
            217, 188, 231, 222, 137;
            101, 204, 222, 255, 193
        );
        assert_eq!(Mask::from_image(&mask_base, 2, 2), Mask::square(2));
    }

    #[test]
    fn test_mask_square_3() {
        let mask_base = gray_image!(
            217, 188, 101, 222, 137, 101, 222;
            231, 204, 255, 222, 137, 222, 255;
            193, 101, 188, 217, 217, 222, 188;
            217, 188, 231, 222, 137, 217, 149;
            193, 255, 193, 188, 231, 222, 188;
            217, 188, 231, 149, 101, 188, 149;
            101, 204, 222, 255, 193, 255, 182
        );
        assert_eq!(Mask::from_image(&mask_base, 3, 3), Mask::square(3));
    }

    #[test]
    fn test_mask_diamond_1() {
        let mask_base = gray_image!(
             0,  31,   0;
             2, 219, 173;
             0,   7,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 1, 1), Mask::diamond(1));
    }

    #[test]
    fn test_mask_diamond_2() {
        let mask_base = gray_image!(
              0,   0, 255,   0,   0;
              0, 231, 204, 101,   0;
            149, 193, 188, 137, 199;
              0, 222, 182, 114,   0;
              0,   0, 217,   0,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 2, 2), Mask::diamond(2));
    }

    #[test]
    fn test_mask_diamond_3() {
        let mask_base = gray_image!(
              0,   0,   0, 222,   0,   0,   0;
              0,   0, 255, 222, 137,   0,   0;
              0, 101, 188, 217, 217, 222,   0;
            217, 188, 231, 222, 137, 217, 149;
              0, 255, 193, 188, 231, 222,   0;
              0,   0, 231, 149, 101,   0,   0;
              0,   0,   0, 255,   0,   0,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 3, 3), Mask::diamond(3));
    }

    #[test]
    fn test_mask_disk_1() {
        let mask_base = gray_image!(
             0,  31,   0;
             2, 219, 173;
             0,   7,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 1, 1), Mask::disk(1));
    }

    #[test]
    fn test_mask_disk_2() {
        let mask_base = gray_image!(
              0,   0, 255,   0,   0;
              0, 231, 204, 101,   0;
            149, 193, 188, 137, 199;
              0, 222, 182, 114,   0;
              0,   0, 217,   0,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 2, 2), Mask::disk(2));
    }

    #[test]
    fn test_mask_disk_3() {
        let mask_base = gray_image!(
              0,   0,   0, 222,   0,   0,   0;
              0, 149, 255, 222, 137, 101,   0;
              0, 101, 188, 217, 217, 222,   0;
            217, 188, 231, 222, 137, 217, 149;
              0, 255, 193, 188, 231, 222,   0;
              0, 137, 231, 149, 101, 188,   0;
              0,   0,   0, 255,   0,   0,   0
        );
        assert_eq!(Mask::from_image(&mask_base, 3, 3), Mask::disk(3));
    }

    #[test]
    fn test_grayscale_dilate_0() {
        let image = gray_image!(
            217, 188, 101, 222, 137, 101, 222;
            231, 204, 255, 222, 137, 222, 255;
            193, 101, 188, 217, 217, 222, 188;
            217, 188, 231, 222, 137, 217, 149;
            193, 255, 193, 188, 231, 222, 188;
            217, 188, 231, 149, 101, 188, 149;
            101, 204, 222, 255, 193, 255, 182
        );
        assert_eq!(grayscale_dilate(&image, &Mask::square(0)), image.clone());
    }

    #[test]
    fn test_grayscale_dilate_diamond_1() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80,  80,  80,  80, 212,  80;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212,  80;
             0,  80,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::diamond(1)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_diamond_3() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80,  80, 212, 212, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212,  80;
            80,  80,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::diamond(3)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_square_1() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80,  80,  80,  80;
             0,  80,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::square(1)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_square_3() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80, 212, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80,  80,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::square(3)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_disk_1() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80,  80,  80,  80, 212,  80;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212,  80;
             0,  80,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::disk(1)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_disk_3() {
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80, 212, 212, 212, 212, 212;
            80,  80, 212, 212, 212, 212, 212;
            80, 212, 212, 212, 212, 212, 212;
            80,  80, 212, 212, 212, 212, 212;
            80,  80, 212, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212,  80;
            80,  80,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &Mask::disk(3)), dilated);
    }

    #[test]
    fn test_grayscale_dilate_arbitrary() {
        let mask = Mask::from_image(
            &gray_image!(
                15, 7;
                0, 17;
                0, 253;
                0, 22
            ),
            1,
            2,
        );
        let image = gray_image!(
             80,  80,  80,  80,  80,  80,  80;
             80,  80,  80,  80,  80, 212,  80;
             80,  80,  80,  80, 212, 212,  80;
              0,  80,  80,  80,  80,  80,  80;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let dilated = gray_image!(
            80,  80,  80,  80,  80, 212,  80;
            80,  80,  80,  80, 212, 212,  80;
            80,  80,  80,  80, 212, 212,  80;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80,  80,  80;
             0,   0,  80,  80,  80,  80,  80
        );
        assert_eq!(grayscale_dilate(&image, &mask), dilated);
    }

    #[test]
    fn test_grayscale_erode_0() {
        let image = gray_image!(
            217, 188, 101, 222, 137, 101, 222;
            231, 204, 255, 222, 137, 222, 255;
            193, 101, 188, 217, 217, 222, 188;
            217, 188, 231, 222, 137, 217, 149;
            193, 255, 193, 188, 231, 222, 188;
            217, 188, 231, 149, 101, 188, 149;
            101, 204, 222, 255, 193, 255, 182
        );
        assert_eq!(grayscale_erode(&image, &Mask::square(0)), image.clone());
    }

    #[test]
    fn test_grayscale_erode_diamond_1() {
        let image = gray_image!(
             80,  80,  80, 212, 212, 212, 212;
             80,  80,  80,  80, 212, 212, 212;
             80,  80,  80,  80, 212, 212, 212;
              0,  80,  80,  80,  80, 212, 212;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80,  80, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::diamond(1)), eroded);
    }

    #[test]
    fn test_grayscale_erode_diamond_3() {
        let image = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
             0,  80,  80,  80,  80,  80, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,   0,  80,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::diamond(3)), eroded);
    }

    #[test]
    fn test_grayscale_erode_square_1() {
        let image = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
            80,  80,  80,  80,  80, 212, 212;
            80,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80, 212;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::square(1)), eroded);
    }

    #[test]
    fn test_grayscale_erode_square_3() {
        let image = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,   0,  80,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::square(3)), eroded);
    }

    #[test]
    fn test_grayscale_erode_disk_1() {
        let image = gray_image!(
             80,  80,  80, 212, 212, 212, 212;
             80,  80,  80,  80, 212, 212, 212;
             80,  80,  80,  80, 212, 212, 212;
              0,  80,  80,  80,  80, 212, 212;
              0,   0,  80,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80;
              0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80,  80, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::disk(1)), eroded);
    }

    #[test]
    fn test_grayscale_erode_disk_3() {
        let image = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
             0,  80,  80,  80,  80,  80, 212;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80;
             0,   0,   0,   0,   0,  80,  80;
             0,   0,   0,   0,   0,  80,  80;
             0,   0,   0,   0,   0,   0,  80;
             0,   0,   0,   0,   0,   0,  80
        );
        assert_eq!(grayscale_erode(&image, &Mask::disk(3)), eroded);
    }

    #[test]
    fn test_grayscale_erode_arbitrary() {
        let mask = Mask::from_image(
            &gray_image!(
                15, 7;
                1, 17;
                0, 253;
                0, 22
            ),
            1,
            2,
        );
        let image = gray_image!(
            80,  80,  80, 212, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80
        );
        let eroded = gray_image!(
            80,  80,  80,  80, 212, 212, 212;
            80,  80,  80,  80, 212, 212, 212;
             0,  80,  80,  80,  80, 212, 212;
             0,   0,  80,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,  80,  80,  80,  80;
             0,   0,   0,   0,  80,  80,  80
        );
        assert_eq!(grayscale_erode(&image, &mask), eroded);
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
    fn bench_erode_l1_5(b: &mut Bencher) {
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
