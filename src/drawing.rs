//! Helpers for drawing basic shapes on images.

use image::{
    Pixel,
    GenericImage,
    ImageBuffer
};

/// Draws a colored cross on an image in place.
/// If (x, y) is within the image bounds then as much of the cross
/// is drawn as will fit. If (x, y) is outside the image bounds then
/// we draw nothing.
pub fn draw_cross_mut<I>(image: &mut I, color: I::Pixel, x: u32, y: u32)
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

    let (width, height) = image.dimensions();
    if x >= width || y >= height {
        return;
    }
    if y > 0 {
        image.put_pixel(x, y - 1, color);
    }
    if x > 0 {
        image.put_pixel(x - 1, y, color);
    }
    image.put_pixel(x, y, color);
    if x + 1 < width {
        image.put_pixel(x + 1, y, color);
    }
    if y + 1 < height {
        image.put_pixel(x, y + 1, color);
    }
}

/// Draws a colored cross on an image.
/// If (x, y) is within the image bounds then as much of the cross
/// is drawn as will fit. If (x, y) is outside the image bounds then
/// we draw nothing.
pub fn draw_cross<I>(image: &I, color: I::Pixel, x: u32, y: u32)
        -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_cross_mut(&mut out, color, x, y);
    out
}

#[cfg(test)]
mod test {

    use super::{
      draw_cross
    };
    use image::{
      GrayImage,
      ImageBuffer,
      Luma
    };

    #[test]
    fn test_draw_corner_inside_bounds() {
      let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1,
          1, 1, 1, 1, 1]).unwrap();

      let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
          1, 1, 1, 1, 1,
          1, 1, 2, 1, 1,
          1, 2, 2, 2, 1,
          1, 1, 2, 1, 1,
          1, 1, 1, 1, 1]).unwrap();

      assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 2, 2), expected);
    }

    #[test]
    fn test_draw_corner_partially_outside_left() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            2, 1, 1, 1, 1,
            2, 2, 1, 1, 1,
            2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 0, 2), expected);
    }

    #[test]
    fn test_draw_corner_partially_outside_right() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 2,
            1, 1, 1, 2, 2,
            1, 1, 1, 1, 2,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 4, 2), expected);
    }

    #[test]
    fn test_draw_corner_partially_outside_bottom() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 2, 1, 1,
            1, 2, 2, 2, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 2, 4), expected);
    }

    #[test]
    fn test_draw_corner_partially_outside_top() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 2, 2, 2, 1,
            1, 1, 2, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 2, 0), expected);
    }

    #[test]
    fn test_draw_corner_outside_bottom() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 0, 5), image);
    }

    #[test]
    fn test_draw_corner_outside_right() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 5, 0), image);
    }
}
