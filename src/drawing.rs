//! Helpers for drawing basic shapes on images.

use image::{
    Pixel,
    GenericImage,
    ImageBuffer
};

use definitions::{
    VecBuffer
};

/// Draws a colored cross on an image in place. Handles coordinates outside image bounds.
pub fn draw_cross_mut<I>(image: &mut I, color: I::Pixel, x: i32, y: i32)
    where I: GenericImage {

    let (width, height) = image.dimensions();
    let idx = |x, y| (3 * (y + 1) + x + 1) as usize;
    let stencil = [0u8, 1u8, 0u8,
                   1u8, 1u8, 1u8,
                   0u8, 1u8, 0u8];

    for sy in -1..2 {
        let iy = y + sy;
        if iy < 0 || iy >= height as i32 {
            continue;
        }

        for sx in -1..2 {
            let ix = x + sx;
            if ix < 0 || ix >= width as i32 {
                continue;
            }

            if stencil[idx(sx, sy)] == 1u8 {
                image.put_pixel(ix as u32, iy as u32, color);
            }
        }
    }
}

/// Draws a colored cross on an image. Handles coordinates outside image bounds.
pub fn draw_cross<I>(image: &I, color: I::Pixel, x: i32, y: i32)
        -> VecBuffer<I::Pixel>
    where I: GenericImage, I::Pixel: 'static {
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

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            2, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 0, 5), expected);
    }

    #[test]
    fn test_draw_corner_outside_right() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 2,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 5, 0), expected);
    }
}
