//! Functions for rotating images.

use image::Pixel;

use crate::definitions::Image;

/// Rotates an image 90 degrees clockwise.
///
/// # Examples
/// ```
/// use imageproc::rotate::rotate90;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     1, 2, 0, 0;
///     3, 4, 0, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 0);
///
/// let rotated = rotate90(&image);
///
/// assert_eq!(rotated, gray_image!(
///     0, 0, 3, 1;
///     0, 0, 4, 2;
///     0, 0, 0, 0;
///     0, 0, 0, 0));
/// ```
pub fn rotate90<P>(image: &Image<P>) -> Image<P>
where
    P: Pixel,
{
    let (width, height) = image.dimensions();

    let mut rotated = Image::new(height, width);

    for y in 0..height {
        for x in 0..width {
            rotated.put_pixel(height - y - 1, x, *image.get_pixel(x, y));
        }
    }

    rotated
}

/// Rotates an image 270 degrees clockwise.
///
/// # Examples
/// ```
/// use imageproc::rotate::rotate270;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     1, 2, 0, 0;
///     3, 4, 0, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 0);
///
/// let rotated = rotate270(&image);
///
/// assert_eq!(rotated, gray_image!(
///     0, 0, 0, 0;
///     0, 0, 0, 0;
///     2, 4, 0, 0;
///     1, 3, 0, 0));
/// ```
pub fn rotate270<P>(image: &Image<P>) -> Image<P>
where
    P: Pixel,
{
    let (width, height) = image.dimensions();

    let mut rotated = Image::new(height, width);

    for y in 0..height {
        for x in 0..width {
            rotated.put_pixel(y, width - x - 1, *image.get_pixel(x, y));
        }
    }

    rotated
}

/// Rotates an image 180 degrees clockwise.
///
/// # Examples
/// ```
/// use imageproc::rotate::rotate180;
/// use imageproc::gray_image;
///
/// let image = gray_image!(
///     1, 2, 0, 0;
///     3, 4, 0, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 0);
///
/// let rotated = rotate180(&image);
///
/// assert_eq!(rotated, gray_image!(
///     0, 0, 0, 0;
///     0, 0, 0, 0;
///     0, 0, 4, 3;
///     0, 0, 2, 1));
/// ```
pub fn rotate180<P>(image: &Image<P>) -> Image<P>
where
    P: Pixel,
{
    let mut out = image.clone();
    rotate180_mut(&mut out);
    out
}
#[doc=generate_mut_doc_comment!("rotate180")]
pub fn rotate180_mut<P>(image: &mut Image<P>)
where
    P: Pixel,
{
    let (width, height) = image.dimensions();

    for y in 0..height / 2 {
        for x in 0..width {
            let x180 = width - x - 1;
            let y180 = height - y - 1;

            let p = *image.get_pixel(x, y);
            let p180 = *image.get_pixel(x180, y180);

            image.put_pixel(x, y, p180);
            image.put_pixel(x180, y180, p);
        }
    }

    if height % 2 != 0 {
        let y_middle = height / 2;

        for x in 0..width / 2 {
            let x180 = width - x - 1;

            let p = *image.get_pixel(x, y_middle);
            let p180 = *image.get_pixel(x180, y_middle);

            image.put_pixel(x, y_middle, p180);
            image.put_pixel(x180, y_middle, p);
        }
    }
}
