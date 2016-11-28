//! Functions for computing gradients of image intensities.

use image::{
    GenericImage,
    ImageBuffer,
    Luma
};

use definitions::{
    VecBuffer
};

use filter::{
    filter3x3
};

/// Sobel filter for vertical gradients.
static VERTICAL_SOBEL: [i32; 9] = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1];

/// Sobel filter for horizontal gradients.
static HORIZONTAL_SOBEL: [i32; 9] = [
     -1, 0, 1,
     -2, 0, 2,
     -1, 0, 1];

/// Convolves with the horizontal Sobel kernel to detect horizontal
/// gradients in an image.
pub fn horizontal_sobel<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    filter3x3(image, &HORIZONTAL_SOBEL)
}

/// Convolves with the vertical Sobel kernel to detect vertical
/// gradients in an image.
pub fn vertical_sobel<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    filter3x3(image, &VERTICAL_SOBEL)
}

/// Prewitt filter for vertical gradients.
static VERTICAL_PREWITT: [i32; 9] = [
    -1, -1, -1,
     0,  0,  0,
     1,  1,  1];

/// Prewitt filter for horizontal gradients.
static HORIZONTAL_PREWITT: [i32; 9] = [
     -1, 0, 1,
     -1, 0, 1,
     -1, 0, 1];

/// Convolves with the horizontal Prewitt kernel to detect horizontal
/// gradients in an image.
pub fn horizontal_prewitt<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    filter3x3(image, &HORIZONTAL_PREWITT)
}

/// Convolves with the vertical Prewitt kernel to detect vertical
/// gradients in an image.
pub fn vertical_prewitt<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    filter3x3(image, &VERTICAL_PREWITT)
}

/// Returns the magnitudes of gradients in an image using Sobel filters.
pub fn sobel_gradients<I>(image: &I) -> VecBuffer<Luma<u16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    gradients(image, &HORIZONTAL_SOBEL, &VERTICAL_SOBEL)
}

/// Returns the magnitudes of gradients in an image using Prewitt filters.
pub fn prewitt_gradients<I>(image: &I) -> VecBuffer<Luma<u16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    gradients(image, &HORIZONTAL_PREWITT, &VERTICAL_PREWITT)
}

// TODO: Returns directions as well as magnitudes.
// TODO: Support filtering without allocating a fresh image - filtering functions could
// TODO: take some kind of pixel-sink. This would allow us to compute gradient magnitudes
// TODO: and directions without allocating intermediates for vertical and horizontal gradients.
fn gradients<I>(image: &I, horizontal_kernel: &[i32; 9], vertical_kernel: &[i32; 9])
    -> VecBuffer<Luma<u16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static
{
    let horizontal: ImageBuffer<Luma<i16>, Vec<i16>> = filter3x3(image, horizontal_kernel);
    let vertical: ImageBuffer<Luma<i16>, Vec<i16>> = filter3x3(image, vertical_kernel);

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let h = horizontal.unsafe_get_pixel(x, y)[0] as f32;
                let v = vertical.unsafe_get_pixel(x, y)[0] as f32;
                let m = (h.powi(2) + v.powi(2)).sqrt() as u16;
                out.unsafe_put_pixel(x, y, Luma([m]));
            }
        }
    }

    out
}

#[cfg(test)]
mod test {

    use super::{
        horizontal_sobel,
        vertical_sobel,
        horizontal_prewitt,
        vertical_prewitt
    };

    use image::{
        GrayImage,
        ImageBuffer,
        Luma
    };

    #[test]
    fn test_gradients_constant_image() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([15u8]));
        let expected = ImageBuffer::from_pixel(5, 5, Luma([0i16]));
        assert_pixels_eq!(horizontal_sobel(&image), expected);
        assert_pixels_eq!(vertical_sobel(&image), expected);
        assert_pixels_eq!(horizontal_prewitt(&image), expected);
        assert_pixels_eq!(vertical_prewitt(&image), expected);
    }

    #[test]
    fn test_horizontal_sobel_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 2, 1,
            6, 5, 4,
            9, 8, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16]).unwrap();

        let filtered = horizontal_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_sobel_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 6, 9,
            2, 5, 8,
            1, 4, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -4i16, -4i16, -4i16,
            -8i16, -8i16, -8i16,
            -4i16, -4i16, -4i16]).unwrap();

        let filtered = vertical_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_prewitt_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 2, 1,
            6, 5, 4,
            9, 8, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -3i16, -6i16, -3i16,
            -3i16, -6i16, -3i16,
            -3i16, -6i16, -3i16]).unwrap();

        let filtered = horizontal_prewitt(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_prewitt_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 6, 9,
            2, 5, 8,
            1, 4, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -3i16, -3i16, -3i16,
            -6i16, -6i16, -6i16,
            -3i16, -3i16, -3i16]).unwrap();

        let filtered = vertical_prewitt(&image);
        assert_pixels_eq!(filtered, expected);
    }
}
