//! Functions for composing one or more images.

use std::cmp::min;

use image::Pixel;
use image::math::Rect;

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

/// Replaces the pixels in the `bottom` image with the pixels from the top image starting from the
/// given `(x, y)` coordinates in the `bottom` image and starting from `(0, 0)` in the `top` image.
///
/// # Panics
///
/// - If `x >= bottom.width()`
/// - If `y >= bottom.height()`
///
/// # Examples
/// ```
/// use imageproc::compose::replace;
/// use imageproc::gray_image;
///
/// let bottom = gray_image!(
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0);
///
/// let top = gray_image!(
///     1, 1;
///     1, 1;
///     1, 1);
///
/// let replaced = replace(&bottom, &top, 3, 1);
///
/// assert_eq!(replaced, gray_image!(
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 1, 1, 0;
///     0, 0, 0, 0, 0, 0;
///     0, 0, 0, 0, 0, 0));
/// ```
pub fn replace<P>(bottom: &Image<P>, top: &Image<P>, x: u32, y: u32) -> Image<P>
where
    P: Pixel,
{
    let mut bottom = bottom.clone();
    replace_mut(&mut bottom, top, x, y);
    bottom
}
#[doc=generate_mut_doc_comment!("replace")]
pub fn replace_mut<P>(bottom: &mut Image<P>, top: &Image<P>, x: u32, y: u32)
where
    P: Pixel,
{
    assert!(x < bottom.width());
    assert!(y < bottom.height());

    let x_end = min(bottom.width(), x.saturating_add(top.width()));
    let y_end = min(bottom.height(), y.saturating_add(top.height()));

    for y_bot in y..y_end {
        for x_bot in x..x_end {
            bottom.put_pixel(x_bot, y_bot, *top.get_pixel(x_bot - x, y_bot - y));
        }
    }
}

/// Blends the pixels in the `bottom` image with the pixels from the top image starting from the
/// given `(x, y)` coordinates in the `bottom` image and starting from `(0, 0)` in the `top` image.
///
/// # Panics
///
/// - If `x >= bottom.width()`
/// - If `y >= bottom.height()`
///
/// # Examples
/// ```
/// use imageproc::compose::overlay;
/// use imageproc::definitions::Image;
/// use image::LumaA;
///
/// let bottom = Image::from_pixel(4, 4, LumaA([0u8, 255]));
/// let top = Image::from_pixel(4, 4, LumaA([255u8, 0]));
///
/// let overlay = overlay(&bottom, &top, 0, 0);
///
/// assert_eq!(overlay, bottom);
/// ```
pub fn overlay<P>(bottom: &Image<P>, top: &Image<P>, x: u32, y: u32) -> Image<P>
where
    P: Pixel,
{
    let mut bottom = bottom.clone();
    overlay_mut(&mut bottom, top, x, y);
    bottom
}
#[doc=generate_mut_doc_comment!("overlay")]
pub fn overlay_mut<P>(bottom: &mut Image<P>, top: &Image<P>, x: u32, y: u32)
where
    P: Pixel,
{
    assert!(x < bottom.width());
    assert!(y < bottom.height());

    let x_end = min(bottom.width(), x.saturating_add(top.width()));
    let y_end = min(bottom.height(), y.saturating_add(top.height()));

    for y_bot in y..y_end {
        for x_bot in x..x_end {
            let mut bottom_pixel = *bottom.get_pixel(x_bot, y_bot);
            let top_pixel = top.get_pixel(x_bot - x, y_bot - y);

            bottom_pixel.blend(top_pixel);

            bottom.put_pixel(x_bot, y_bot, bottom_pixel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_extracts_requested_rectangle() {
        let image = gray_image!(
            0, 1, 2, 3;
            4, 5, 6, 7;
            8, 9, 10, 11
        );
        let out = crop(
            &image,
            Rect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
        );

        assert_pixels_eq!(out, gray_image!(5, 6; 9, 10));
    }

    #[test]
    #[should_panic]
    fn crop_panics_when_rectangle_is_outside_image() {
        let image = gray_image!(0, 1; 2, 3);
        let _ = crop(
            &image,
            Rect {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
        );
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn crop_parallel_matches_crop() {
        let image = gray_image!(
            0, 1, 2, 3;
            4, 5, 6, 7;
            8, 9, 10, 11
        );
        let rect = Rect {
            x: 1,
            y: 0,
            width: 2,
            height: 3,
        };

        assert_pixels_eq!(crop_parallel(&image, rect), crop(&image, rect));
    }

    #[test]
    fn flip_horizontal_matches_expected() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6
        );
        let expected = gray_image!(
            3, 2, 1;
            6, 5, 4
        );

        assert_pixels_eq!(flip_horizontal(&image), expected);

        let mut image_mut = image.clone();
        flip_horizontal_mut(&mut image_mut);
        assert_pixels_eq!(image_mut, expected);
    }

    #[test]
    fn flip_vertical_matches_expected() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        );
        let expected = gray_image!(
            7, 8, 9;
            4, 5, 6;
            1, 2, 3
        );

        assert_pixels_eq!(flip_vertical(&image), expected);

        let mut image_mut = image.clone();
        flip_vertical_mut(&mut image_mut);
        assert_pixels_eq!(image_mut, expected);
    }

    #[test]
    fn replace_copies_top_with_clipping() {
        let bottom = gray_image!(
            0, 0, 0;
            0, 0, 0;
            0, 0, 0
        );
        let top = gray_image!(
            1, 2;
            3, 4
        );
        let expected = gray_image!(
            0, 0, 0;
            0, 0, 0;
            0, 0, 1
        );

        assert_pixels_eq!(replace(&bottom, &top, 2, 2), expected);

        let mut bottom_mut = bottom.clone();
        replace_mut(&mut bottom_mut, &top, 2, 2);
        assert_pixels_eq!(bottom_mut, expected);
    }

    #[test]
    #[should_panic]
    fn replace_panics_when_start_x_is_outside_bottom() {
        let mut bottom = gray_image!(0, 0; 0, 0);
        let top = gray_image!(1);
        replace_mut(&mut bottom, &top, 2, 0);
    }

    #[test]
    fn overlay_blends_top_over_bottom() {
        let bottom = gray_image!(
            0, 0, 0;
            0, 0, 0;
            0, 0, 0
        );
        let top = gray_image!(
            200, 200;
            200, 200
        );
        let expected = gray_image!(
            0, 0, 0;
            0, 0, 0;
            0, 0, 200
        );

        assert_pixels_eq!(overlay(&bottom, &top, 2, 2), expected);

        let mut bottom_mut = bottom.clone();
        overlay_mut(&mut bottom_mut, &top, 2, 2);
        assert_pixels_eq!(bottom_mut, expected);
    }

    #[test]
    #[should_panic]
    fn overlay_panics_when_start_y_is_outside_bottom() {
        let mut bottom = gray_image!(
            0, 0;
            0, 0
        );
        let top = gray_image!(1);
        overlay_mut(&mut bottom, &top, 0, 2);
    }
}
