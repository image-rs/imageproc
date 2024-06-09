//! Functions for composing one or more images.

use image::math::Rect;
use image::Pixel;

use crate::definitions::Image;

/// Crops an image to a given rectangle.
///
/// # Panics
///
/// - If `rect.x + rect.width > image.width()`
/// - If `rect.y + rect.height > image.height()`
///
/// # Examples
/// ```
/// use imageproc::compose::crop;
/// use image::math::Rect;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0);
///
/// let cropped = crop(&image, Rect {x: 3, y: 1, width: 2, height: 3});
///
/// assert_eq!(cropped, gray_image!(
///     1, 1;
///     1, 1;
///     1, 1));
/// ```
pub fn crop<P>(image: &Image<P>, rect: Rect) -> Image<P>
where
    P: Pixel,
{
    assert!(rect.x + rect.width <= image.width());
    assert!(rect.y + rect.height <= image.height());

    Image::from_fn(rect.width, rect.height, |x, y| {
        *image.get_pixel(rect.x + x, rect.y + y)
    })
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("crop")]
pub fn crop_parallel<P>(image: &Image<P>, rect: Rect) -> Image<P>
where
    P: Pixel + Send + Sync,
    P::Subpixel: Send + Sync,
{
    assert!(rect.x + rect.width <= image.width());
    assert!(rect.y + rect.height <= image.height());

    Image::from_par_fn(rect.width, rect.height, |x, y| {
        *image.get_pixel(rect.x + x, rect.y + y)
    })
}

/// Horizontally flips an image.
///
/// # Examples
/// ```
/// use imageproc::compose::flip_horizontal;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     1, 2, 3, 4, 5, 6;
///     1, 2, 3, 4, 5, 6;
///     1, 2, 3, 4, 5, 6;
///     1, 2, 3, 4, 5, 6;
///     1, 2, 3, 4, 5, 6;
///     1, 2, 3, 4, 5, 6);
///
/// let flipped = flip_horizontal(&image);
///
/// assert_eq!(flipped, gray_image!(
///     6, 5, 4, 3, 2, 1;
///     6, 5, 4, 3, 2, 1;
///     6, 5, 4, 3, 2, 1;
///     6, 5, 4, 3, 2, 1;
///     6, 5, 4, 3, 2, 1;
///     6, 5, 4, 3, 2, 1));
/// ```
pub fn flip_horizontal<P>(image: &Image<P>) -> Image<P>
where
    P: Pixel,
{
    let mut out = image.clone();
    flip_horizontal_mut(&mut out);
    out
}
#[doc=generate_mut_doc_comment!("flip_horizontal")]
pub fn flip_horizontal_mut<P>(image: &mut Image<P>)
where
    P: Pixel,
{
    for y in 0..image.height() {
        for x in 0..(image.width() / 2) {
            let flipped_x = image.width() - x - 1;

            let pixel = *image.get_pixel(x, y);
            let flipped_pixel = *image.get_pixel(flipped_x, y);

            image.put_pixel(x, y, flipped_pixel);
            image.put_pixel(flipped_x, y, pixel);
        }
    }
}

/// Vertically flips an image.
///
/// # Examples
/// ```
/// use imageproc::compose::flip_vertical;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     1, 1, 1, 1, 1, 1;
///     2, 2, 2, 2, 2, 2;
///     3, 3, 3, 3, 3, 3;
///     4, 4, 4, 4, 4, 4;
///     5, 5, 5, 5, 5, 5;
///     6, 6, 6, 6, 6, 6);
///
/// let flipped = flip_vertical(&image);
///
/// assert_eq!(flipped, gray_image!(
///     6, 6, 6, 6, 6, 6;
///     5, 5, 5, 5, 5, 5;
///     4, 4, 4, 4, 4, 4;
///     3, 3, 3, 3, 3, 3;
///     2, 2, 2, 2, 2, 2;
///     1, 1, 1, 1, 1, 1));
/// ```
pub fn flip_vertical<P>(image: &Image<P>) -> Image<P>
where
    P: Pixel,
{
    let mut out = image.clone();
    flip_vertical_mut(&mut out);
    out
}
#[doc=generate_mut_doc_comment!("flip_vertical")]
pub fn flip_vertical_mut<P>(image: &mut Image<P>)
where
    P: Pixel,
{
    for y in 0..(image.height() / 2) {
        for x in 0..image.width() {
            let flipped_y = image.height() - y - 1;

            let pixel = *image.get_pixel(x, y);
            let flipped_pixel = *image.get_pixel(x, flipped_y);

            image.put_pixel(x, y, flipped_pixel);
            image.put_pixel(x, flipped_y, pixel);
        }
    }
}
