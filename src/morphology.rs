//! Functions for computing [morphological operators].
//!
//! [morphological operators]: https://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm

use crate::{
    distance_transform::{distance_transform_impl, distance_transform_mut, DistanceFrom, Norm},
    point::Point,
};
use image::{GrayImage, Luma};
use itertools::Itertools;

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
/// //
/// // When computing distances using the L2 norm we take the ceiling of the true values.
/// // This means that using the L2 norm gives the same results as the L1 norm for `k <= 2`.
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
/// // L2 norm
/// //
/// // When computing distances using the L2 norm we take the ceiling of the true values.
/// // This means that using the L2 norm gives the same results as the L1 norm for `k <= 2`.
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

/// A mask used in grayscale morphological operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    /// For any optimisation/arithmetic purposes, it is guaranteed that:
    /// - all the integer values will be strictly between -512 and 512
    /// - all tuples will be sorted in reverse lexicographic order, line by line ((-1,-1),(0,-1),(1,-1),(-1,0),(0,0),...)
    /// - no tuple shall appear twice
    elements: Vec<Point<i16>>,
}

macro_rules! lines {
    ($mask:expr) => {
        $mask
            .elements
            .iter()
            .group_by(|p| p.y)
            .into_iter()
            .map(|(y, line)| (y, line.map(|p| p.x)))
    };
}

impl Mask {
    /// Creates a mask containing the points `(x, y) - (center_x, center_y)` for each `(x, y)` with non-zero intensity in `image`.
    ///
    /// Mask contents are represented using signed integers, so `(center_x, center_y)` is not required to be within bounds for `image`. However, `image`
    /// is restricted to have a side length of at most 511 pixels.
    ///
    /// # Panics
    /// If `image.width() >= 512` or `image.height() >= 512`.
    pub fn from_image(image: &GrayImage, center_x: u8, center_y: u8) -> Self {
        assert!(
            image.width() < 512,
            "the input image must be at most 511 pixels wide"
        );
        assert!(
            image.height() < 512,
            "the input image must be at most 511 pixels high"
        );
        let center = Point::new(center_x, center_y).to_i16();
        let elements = image
            .enumerate_pixels()
            .filter(|(_, _, &p)| p[0] != 0)
            .map(|(x, y, _)| Point::new(x, y).to_i16())
            .map(|p| p - center)
            .collect();
        Self::new(elements)
    }

    fn new(elements: Vec<Point<i16>>) -> Self {
        assert!(elements.len() <= (511 * 511) as usize);
        debug_assert!(elements.iter().tuple_windows().all(|(a, b)| {
            if a.y == b.y {
                a.x < b.x
            } else {
                a.y < b.y
            }
        }));
        Self { elements }
    }

    /// Creates a square mask of side length `2 * radius + 1`.
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// // Mask::square(1) is a 3x3 square mask centered at the origin.
    /// let square = gray_image!(
    ///     255, 255, 255;
    ///     255, 255, 255;
    ///     255, 255, 255
    /// );
    /// assert_eq!(Mask::square(1), Mask::from_image(&square, 1, 1));
    /// # }
    /// ```
    pub fn square(radius: u8) -> Self {
        let radius = i16::from(radius);
        let range = -radius..=radius;
        let elements = range
            .clone()
            .cartesian_product(range)
            .map(|(y, x)| Point::new(x, y))
            .collect();
        Self::new(elements)
    }

    /// Creates a diamond-shaped mask containing all points with `L1` norm at most `radius`.
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// // Mask::diamond(1) is a 3x3 cross centered at the origin.
    /// let diamond_1 = gray_image!(
    ///       0, 255,   0;
    ///     255, 255, 255;
    ///       0, 255,   0
    /// );
    /// assert_eq!(Mask::diamond(1), Mask::from_image(&diamond_1, 1, 1));
    ///
    /// // Mask::diamond(2) is a 5x5 diamond centered at the origin.
    /// let diamond_2 = gray_image!(
    ///       0,   0, 255,   0,   0;
    ///       0, 255, 255, 255,   0;
    ///     255, 255, 255, 255, 255;
    ///       0, 255, 255, 255,   0;
    ///       0,   0, 255,   0,   0
    /// );
    /// assert_eq!(Mask::diamond(2), Mask::from_image(&diamond_2, 2, 2));
    /// # }
    /// ```
    pub fn diamond(radius: u8) -> Self {
        let cap = 1 + 2 * usize::from(radius) * (usize::from(radius) + 1);
        let mut elements = Vec::with_capacity(cap);
        let radius = i16::from(radius);
        let points = (-radius..=radius)
            .flat_map(|y| ((y.abs() - radius)..=(radius - y.abs())).map(move |x| Point::new(x, y)));
        elements.extend(points);
        Self::new(elements)
    }

    /// Creates a disk-shaped mask containing all points with `L2` norm at most `radius`.
    ///
    /// When computing distances using the L2 norm we take the ceiling of the true values.
    /// This means that using the L2 norm gives the same results as the `L1` norm for `radius <= 2`.
    ///
    /// # Example
    /// ```
    /// # extern crate image;
    /// # #[macro_use]
    /// # extern crate imageproc;
    /// # fn main() {
    /// use imageproc::morphology::Mask;
    ///
    /// // For radius <= 2, Mask::disk(radius) is the same as Mask::diamond(radius).
    /// let disk_2 = gray_image!(
    ///       0,   0, 255,   0,   0;
    ///       0, 231, 204, 101,   0;
    ///     149, 193, 188, 137, 199;
    ///       0, 222, 182, 114,   0;
    ///       0,   0, 217,   0,   0
    /// );
    /// assert_eq!(Mask::disk(2), Mask::from_image(&disk_2, 2, 2));
    ///
    /// // Mask::disk(3) is a filled circle of radius 3.
    /// let disk_3 = gray_image!(
    ///       0,   0,   0, 255,   0,   0,   0;
    ///       0, 255, 255, 255, 255, 255,   0;
    ///       0, 255, 255, 255, 255, 255,   0;
    ///     255, 255, 255, 255, 255, 255, 255;
    ///       0, 255, 255, 255, 255, 255,   0;
    ///       0, 255, 255, 255, 255, 255,   0;
    ///       0,   0,   0, 255,   0,   0,   0
    /// );
    /// assert_eq!(Mask::disk(3), Mask::from_image(&disk_3, 3, 3));
    /// # }
    /// ```
    pub fn disk(radius: u8) -> Self {
        let radius_squared = u32::from(radius).pow(2);
        let half_widths_per_height = std::iter::successors(
            Some((-i16::from(radius), 0u8)),
            |&(last_height, last_half_width)| {
                if last_height == i16::from(radius) {
                    return None;
                };
                let next_height = last_height + 1;
                let height_squared = (u32::from(next_height.unsigned_abs())).pow(2);
                let next_half_width = if next_height <= 0 {
                    // upper part of the circle => increasing width
                    (u32::from(last_half_width)..)
                        .find(|x| (x + 1).pow(2) + height_squared > radius_squared)?
                } else {
                    // lower part of the circle => decreasing width
                    (0u32..=last_half_width.into())
                        .rev()
                        .find(|&x| x.pow(2) + height_squared <= radius_squared)?
                };
                Some((next_height, next_half_width.try_into().unwrap()))
            },
        );
        let cap = half_widths_per_height
            .clone()
            .map(|(_, half_width)| 2 * usize::from(half_width) + 1)
            .sum();
        let mut elements = Vec::with_capacity(cap);
        let points = half_widths_per_height.flat_map(|(y, half_width)| {
            (-i16::from(half_width)..=i16::from(half_width)).map(move |x| Point::new(x, y))
        });
        elements.extend(points);
        Self::new(elements)
    }
}

fn mask_reduce<F: Fn(u8, u8) -> u8>(
    image: &GrayImage,
    mask: &Mask,
    neutral: u8,
    operator: F,
) -> GrayImage {
    let mut result = GrayImage::from_pixel(image.width(), image.height(), Luma([neutral]));
    for (y, line_group) in lines!(mask) {
        let y = i64::from(y);
        let line = line_group.collect::<Vec<_>>();
        let input_rows = image
            .chunks(image.width() as usize)
            .skip(y.try_into().unwrap_or(0));
        let output_rows = result
            .chunks_mut(image.width() as usize)
            .skip((-y).try_into().unwrap_or(0));
        for (input_row, output_row) in input_rows.zip(output_rows) {
            for x in line.iter().copied() {
                let inputs = input_row.iter().skip(x.try_into().unwrap_or(0));
                let outputs = output_row.iter_mut().skip((-x).try_into().unwrap_or(0));
                for (&input, output) in inputs.zip(outputs) {
                    *output = operator(input, *output);
                }
            }
        }
    }
    result
}

/// Computes the morphologic dilation of `image` with the given mask.
///
/// For each input pixel, the output pixel will be the maximum of all pixels included
/// in the mask at that position. If the mask doesn't intersect any input pixel at some point,
/// it will default to a value of [`u8::MIN`].
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
/// // Using a diamond mask
/// let diamond_dilated = gray_image!(
///     7,   7,   0,   0,   0,   0;
///     7,   0,  99,   0,   0,   0;
///     0,  99,  99,  99,   0,   0;
///     0,   0,  99,   0,   0, 222;
///     0,   0,   0,   0, 222, 222
/// );
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::diamond(1)), diamond_dilated);
///
/// // Using a disk mask
/// let disk_dilated = gray_image!(
///    99,  99,  99,  99,  99,   0;
///    99,  99,  99,  99,  99, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99, 222, 222, 222, 222
/// );
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::disk(3)), disk_dilated);
///
/// // Using a square mask
/// let square_dilated = gray_image!(
///     7,   7,   0,   0,   0,   0;
///     7,  99,  99,  99,   0,   0;
///     0,  99,  99,  99,   0,   0;
///     0,  99,  99,  99, 222, 222;
///     0,   0,   0,   0, 222, 222
/// );
/// assert_pixels_eq!(grayscale_dilate(&image, &Mask::square(1)), square_dilated);
///
/// // Using an arbitrary mask
/// let column_mask = Mask::from_image(
///     &gray_image!(
///       255;
///       255;
///       255;
///       255
///     ),  
///     0,  1
/// );
/// let column_dilated = gray_image!(
///     7,   0,  99,   0,   0,   0;
///     7,   0,  99,   0,   0,   0;
///     0,   0,  99,   0,   0, 222;
///     0,   0,  99,   0,   0, 222;
///     0,   0,   0,   0,   0, 222
/// );
/// assert_pixels_eq!(grayscale_dilate(&image, &column_mask), column_dilated);
/// # }
/// ```
pub fn grayscale_dilate(image: &GrayImage, mask: &Mask) -> GrayImage {
    mask_reduce(image, mask, u8::MIN, u8::max)
}

/// Computes the morphologic erosion of `image` with the given mask.
///
/// For each input pixel, the output pixel will be the minimum of all pixels included
/// in the mask at that position. If the mask doesn't intersect any input pixel at some point,
/// it will default to a value of [`u8::MAX`].
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
/// // Using a diamond mask
/// let diamond_eroded = gray_image!(
///     7,   7,  99,  99,  99,  99;
///     7,  99,  99,  99,  99,  99;
///     7,  99,  99,  99,  99, 222;
///     7,   7,  99,  99,  99, 222;
///     7,  99,  99,  99, 222, 222
/// );
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::diamond(1)), diamond_eroded);
///
/// // Using a disk mask
/// let disk_eroded = gray_image!(
///     7,   7,   7,   7,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,   7,  99,  99;
///     7,   7,   7,  99,  99,  99
/// );
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::disk(3)), disk_eroded);
///
/// // Using a square mask
/// let square_eroded = gray_image!(
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,  99,  99,  99, 222;
///     7,   7,  99,  99,  99, 222
/// );
/// assert_pixels_eq!(grayscale_erode(&image, &Mask::square(1)), square_eroded);
///
/// // Using an arbitrary mask
/// let column_mask = Mask::from_image(
///     &gray_image!(
///       255;
///       255;
///       255;
///       255
///     ),  
///     0,  1
/// );
/// let column_eroded = gray_image!(
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99,  99, 222;
///     7,  99,  99,  99, 222, 222;
///     7,  99,  99,  99, 222, 222
/// );
/// assert_pixels_eq!(grayscale_erode(&image, &column_mask), column_eroded);
/// # }
/// ```
pub fn grayscale_erode(image: &GrayImage, mask: &Mask) -> GrayImage {
    mask_reduce(image, mask, u8::MAX, u8::min)
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
/// let image = gray_image!(
///   100,  99,  99,  99, 222,  99;
///    99,  99,  99, 222, 222, 222;
///    99,   7,  99,  99, 222,  99;
///     7,   7,   7,  99,  99,  99;
///    99,   7,  99,  99,  99,  99
/// );
///
/// // Isolated regions of foreground pixels are removed,
/// // while isolated zones of background are maintained
/// let image_opened = gray_image!(
///    99,  99,  99,  99,  99,  99;
///    99,  99,  99,  99,  99,  99;
///     7,   7,  99,  99,  99,  99;
///     7,   7,   7,  99,  99,  99;
///     7,   7,   7,  99,  99,  99
/// );
/// assert_pixels_eq!(grayscale_open(&image, &Mask::square(1)), image_opened);
///
/// // grayscale_open is idempotent - applying it a second time has no effect.
/// assert_pixels_eq!(grayscale_open(&image_opened, &Mask::square(1)), image_opened);
/// # }
/// ```
pub fn grayscale_open(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_dilate(&grayscale_erode(image, mask), mask)
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
/// // while isolated zones of foreground pixels are maintained
/// let image_closed = gray_image!(
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99, 222, 222, 222;
///    99,  99,  99,  99, 222, 222;
///    99,  99,  99,  99,  99,  99;
///    99,  99,  99,  99,  99,  99
/// );
/// assert_pixels_eq!(grayscale_close(&image, &Mask::square(1)), image_closed);
///
/// // grayscale_close is idempotent - applying it a second time has no effect.
/// assert_pixels_eq!(grayscale_close(&image_closed, &Mask::square(1)), image_closed);
/// # }
/// ```
pub fn grayscale_close(image: &GrayImage, mask: &Mask) -> GrayImage {
    grayscale_erode(&grayscale_dilate(image, mask), mask)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_pixels_eq!(dilated, image);
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
        assert_pixels_eq!(dilated, image);
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
        assert_pixels_eq!(dilated, image);
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
        assert_pixels_eq!(eroded, image);
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
        assert_pixels_eq!(eroded, image);
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
        assert_pixels_eq!(eroded, image);
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
        let mask_base = GrayImage::new(600, 5);
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
    fn test_grayscale_dilate_default_value() {
        let mask = Mask::from_image(&gray_image!(), 0, 0);
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
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN;
            u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN, u8::MIN
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

    #[test]
    fn test_grayscale_erode_default_value() {
        let mask = Mask::from_image(&gray_image!(), 0, 0);
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
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX;
            u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX, u8::MAX
        );
        assert_eq!(grayscale_erode(&image, &mask), dilated);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::proptest_utils::{arbitrary_image, arbitrary_image_with};
    use proptest::prelude::*;

    fn reference_mask_disk(radius: u8) -> Mask {
        let range = -(radius as i16)..=(radius as i16);
        let elements = range
            .clone()
            .cartesian_product(range)
            .filter(|(y, x)| {
                (x.unsigned_abs() as u32).pow(2) + (y.unsigned_abs() as u32).pow(2)
                    <= (radius as u32).pow(2)
            })
            .map(|(y, x)| Point::new(x, y))
            .collect();
        Mask::new(elements)
    }

    proptest! {
        #[test]
        fn proptest_mask_from_white_image(
            img in arbitrary_image_with(Just(255), 0..=511, 0..=511),
            x in any::<u8>(),
            y in any::<u8>(),
        ) {
            Mask::from_image(&img, x, y);
        }

        #[test]
        fn proptest_mask_from_image(
            img in arbitrary_image(0..=511, 0..=511),
            x in any::<u8>(),
            y in any::<u8>(),
        ) {
            Mask::from_image(&img, x, y);
        }

        #[test]
        fn proptest_mask_square(radius in any::<u8>()) {
            Mask::square(radius);
        }

        #[test]
        fn proptest_mask_diamond(radius in any::<u8>()) {
            Mask::diamond(radius);
        }

        #[test]
        fn proptest_mask_disk(radius in any::<u8>()) {
            let actual = Mask::disk(radius);
            let expected = reference_mask_disk(radius);
            assert_eq!(actual, expected);
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use ::test::*;
    use image::{GrayImage, Luma};
    use std::cmp::{max, min};

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

    #[bench]
    fn bench_grayscale_mask_from_image(b: &mut Bencher) {
        let image = GrayImage::from_fn(200, 200, |x, y| Luma([(x + y % 3) as u8]));
        b.iter(|| {
            let mask = Mask::from_image(&image, 100, 100);
            black_box(mask);
        })
    }

    macro_rules! bench_grayscale_mask {
        ($name:ident, $f:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                b.iter(|| {
                    let mask = $f(100);
                    black_box(mask);
                })
            }
        };
    }

    bench_grayscale_mask!(bench_grayscale_square_mask, Mask::square);
    bench_grayscale_mask!(bench_grayscale_diamond_mask, Mask::diamond);
    bench_grayscale_mask!(bench_grayscale_disk_mask, Mask::disk);

    macro_rules! bench_grayscale_operator {
        ($name:ident, $f:expr, $mask:expr, $img_size:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image =
                    GrayImage::from_fn($img_size, $img_size, |x, y| Luma([(x + y % 3) as u8]));
                let mask = $mask;
                b.iter(|| {
                    let processed = $f(&image, &mask);
                    black_box(processed);
                })
            }
        };
    }

    bench_grayscale_operator!(
        bench_grayscale_op_erode_small_image_point,
        grayscale_erode,
        Mask::diamond(0),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_medium_image_point,
        grayscale_erode,
        Mask::diamond(0),
        200
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_big_image_point,
        grayscale_erode,
        Mask::diamond(0),
        1000
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_small_image_diamond,
        grayscale_erode,
        Mask::diamond(5),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_medium_image_diamond,
        grayscale_erode,
        Mask::diamond(5),
        200
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_big_image_diamond,
        grayscale_erode,
        Mask::diamond(5),
        1000
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_small_image_large_square,
        grayscale_erode,
        Mask::square(25),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_erode_medium_image_large_square,
        grayscale_erode,
        Mask::square(25),
        200
    );

    bench_grayscale_operator!(
        bench_grayscale_op_dilate_small_image_point,
        grayscale_dilate,
        Mask::diamond(0),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_medium_image_point,
        grayscale_dilate,
        Mask::diamond(0),
        200
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_big_image_point,
        grayscale_dilate,
        Mask::diamond(0),
        1000
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_small_image_diamond,
        grayscale_dilate,
        Mask::diamond(5),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_medium_image_diamond,
        grayscale_dilate,
        Mask::diamond(5),
        200
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_big_image_diamond,
        grayscale_dilate,
        Mask::diamond(5),
        1000
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_small_image_large_square,
        grayscale_dilate,
        Mask::square(25),
        50
    );
    bench_grayscale_operator!(
        bench_grayscale_op_dilate_medium_image_large_square,
        grayscale_dilate,
        Mask::square(25),
        200
    );
}
