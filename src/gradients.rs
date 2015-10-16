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

/// Sobel filter for vertical edges.
static VERTICAL_SOBEL: [i32; 9] = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1];

/// Sobel filter for horizontal edges.
static HORIZONTAL_SOBEL: [i32; 9] = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1];

/// Convolves with the horizontal Sobel kernel to detect horizontal
/// edges in an image.
pub fn horizontal_sobel<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    filter3x3(image, &HORIZONTAL_SOBEL)
}

/// Convolves with the vertical Sobel kernel to detect vertical
/// edges in an image.
pub fn vertical_sobel<I>(image: &I) -> VecBuffer<Luma<i16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    filter3x3(image, &VERTICAL_SOBEL)
}

/// Returns the magnitudes of gradients in an image.
// TODO: Returns directions as well as magnitudes.
// TODO: Support filtering with allocating a fresh image - filtering functions could
// TODO: take some kind of pixel-sink. This would allow us to compute gradient magnitudes
// TODO: and directions without allocating intermediates for vertical and horizontal gradients.
pub fn sobel_gradients<I>(image: &I) -> VecBuffer<Luma<u16>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let horizontal = horizontal_sobel(image);
    let vertical = vertical_sobel(image);

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let h = horizontal.get_pixel(x, y)[0] as f32;
            let v = vertical.get_pixel(x, y)[0] as f32;
            let m = (h.powi(2) + v.powi(2)).sqrt() as u16;
            out.put_pixel(x, y, Luma([m]));
        }
    }

    out
}

#[cfg(test)]
mod test {

    use super::{
        horizontal_sobel,
        vertical_sobel
    };

    use image::{
        GrayImage,
        ImageBuffer,
        Luma
    };

    #[test]
    fn test_sobel_constant_image() {

        let image = ImageBuffer::from_fn(5, 5, |_, _| Luma([15u8]));
        let expected = ImageBuffer::from_fn(5, 5, |_, _| Luma([0i16]));
        assert_pixels_eq!(horizontal_sobel(&image), expected);
        assert_pixels_eq!(vertical_sobel(&image), expected);
    }

    #[test]
    fn test_vertical_sobel_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 2, 1,
            6, 5, 4,
            9, 8, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16]).unwrap();

        let filtered = vertical_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_sobel_gradient_image() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 6, 9,
            2, 5, 8,
            1, 4, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -4i16, -4i16, -4i16,
            -8i16, -8i16, -8i16,
            -4i16, -4i16, -4i16]).unwrap();

        let filtered = horizontal_sobel(&image);
        assert_pixels_eq!(filtered, expected);
    }
}
