//! Helpers for drawing basic shapes on images.

use image::{
    Pixel,
    GenericImage,
    ImageBuffer
};
use definitions::{
    VecBuffer
};
use std::mem::swap;

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

/// Draws as much of the line segment between start and end as lies inside the image bounds.
pub fn draw_line_segment<I>(image: &I, start: (f32, f32), end: (f32, f32), color: I::Pixel)
        -> VecBuffer<I::Pixel>
    where I: GenericImage, I::Pixel: 'static{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_line_segment_mut(&mut out, start, end, color);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
pub fn draw_line_segment_mut<I>(image: &mut I, start: (f32, f32), end: (f32, f32), color: I::Pixel)
    where I: GenericImage, I::Pixel: 'static{

    let (width, height) = image.dimensions();
    let in_bounds = |x, y| x >= 0 && x < width as i32 && y >= 0 && y < height as i32;

    let (mut x0, mut y0) = (start.0, start.1);
    let (mut x1, mut y1) = (end.0, end.1);

    let is_steep = (y1 - y0).abs() > (x1 - x0).abs();

    if is_steep {
        swap(&mut x0, &mut y0);
        swap(&mut x1, &mut y1);
    }

    if x0 > x1 {
        swap(&mut x0, &mut x1);
        swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = (y1 - y0).abs();
    let mut error = dx / 2f32;

    let y_step = if y0 < y1 {1f32} else {-1f32};
    let mut y = y0 as i32;

    for x in x0 as i32..(x1 + 1f32) as i32 {
        if is_steep && in_bounds(y, x) {
            image.put_pixel(y as u32, x as u32, color);
        }
        else if in_bounds(x, y) {
            image.put_pixel(x as u32, y as u32, color);
        }
        error -= dy;
        if error < 0f32 {
            y = (y as f32 + y_step) as i32;
            error += dx;
        }
    }
}

#[cfg(test)]
mod test {

    use super::{
      draw_cross,
      draw_line_segment
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
            1, 1, 3, 1, 1,
            1, 3, 3, 3, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([3u8]), 2, 4), expected);
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
            1, 9, 9, 9, 1,
            1, 1, 9, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 2, 0), expected);
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
            9, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 0, 5), expected);
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
            1, 1, 1, 1, 9,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 5, 0), expected);
    }


    // Octants for line directions:
    //
    //   \ 5 | 6 /
    //   4 \ | / 7
    //   ---   ---
    //   3 / | \ 0
    //   / 2 | 1 \

    #[test]
    fn test_draw_line_segment_horizontal() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            4, 4, 4, 4, 4,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let right = draw_line_segment(&image, (-3f32, 1f32), (6f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(right, expected);

        let left = draw_line_segment(&image, (6f32, 1f32), (-3f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(left, expected);
    }

    #[test]
    fn test_draw_line_segment_oct0_and_oct4() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 9, 9, 1, 1,
            1, 1, 1, 9, 9,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let oct0 = draw_line_segment(&image, (1f32, 1f32), (4f32, 2f32), Luma([9u8]));
        assert_pixels_eq!(oct0, expected);

        let oct4 = draw_line_segment(&image, (4f32, 2f32), (1f32, 1f32), Luma([9u8]));
        assert_pixels_eq!(oct4, expected);
    }

    #[test]
    fn test_draw_line_segment_diagonal() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 6, 1, 1, 1,
            1, 1, 6, 1, 1,
            1, 1, 1, 6, 1,
            1, 1, 1, 1, 1]).unwrap();

        let down_right = draw_line_segment(&image, (1f32, 1f32), (3f32, 3f32), Luma([6u8]));
        assert_pixels_eq!(down_right, expected);

        let up_left = draw_line_segment(&image, (3f32, 3f32), (1f32, 1f32), Luma([6u8]));
        assert_pixels_eq!(up_left, expected);
    }

    #[test]
    fn test_draw_line_segment_oct1_and_oct5() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            5, 1, 1, 1, 1,
            5, 1, 1, 1, 1,
            5, 1, 1, 1, 1,
            1, 5, 1, 1, 1,
            1, 5, 1, 1, 1]).unwrap();

        let oct1 = draw_line_segment(&image, (0f32, 0f32), (1f32, 4f32), Luma([5u8]));
        assert_pixels_eq!(oct1, expected);

        let oct5 = draw_line_segment(&image, (1f32, 4f32), (0f32, 0f32), Luma([5u8]));
        assert_pixels_eq!(oct5, expected);
    }

    #[test]
    fn test_draw_line_segment_vertical() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 1, 1]).unwrap();

        let down = draw_line_segment(&image, (3f32, 1f32), (3f32, 3f32), Luma([8u8]));
        assert_pixels_eq!(down, expected);

        let up = draw_line_segment(&image, (3f32, 3f32), (3f32, 1f32), Luma([8u8]));
        assert_pixels_eq!(up, expected);
    }

    #[test]
    fn test_draw_line_segment_oct2_and_oct6() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 4, 1, 1,
            1, 1, 4, 1, 1,
            1, 4, 1, 1, 1,
            1, 4, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let oct2 = draw_line_segment(&image, (2f32, 0f32), (1f32, 3f32), Luma([4u8]));
        assert_pixels_eq!(oct2, expected);

        let oct6 = draw_line_segment(&image, (1f32, 3f32), (2f32, 0f32), Luma([4u8]));
        assert_pixels_eq!(oct6, expected);
    }

    #[test]
    fn test_draw_line_segment_oct3_and_oct7() {
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
            1, 1, 1, 2, 2,
            2, 2, 2, 1, 1]).unwrap();

        let oct3 = draw_line_segment(&image, (0f32, 4f32), (5f32, 3f32), Luma([2u8]));
        assert_pixels_eq!(oct3, expected);

        let oct7 = draw_line_segment(&image, (5f32, 3f32), (0f32, 4f32), Luma([2u8]));
        assert_pixels_eq!(oct7, expected);
    }
}
