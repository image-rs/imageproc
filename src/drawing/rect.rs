use image::{GenericImage, ImageBuffer};
use crate::definitions::Image;
use crate::rect::Rect;
use std::f32;
use crate::drawing::line::draw_line_segment_mut;

/// Draws as much of the boundary of a rectangle as lies inside the image bounds.
pub fn draw_hollow_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_hollow_rect_mut(&mut out, rect, color);
    out
}

/// Draws as much of the boundary of a rectangle as lies inside the image bounds.
pub fn draw_hollow_rect_mut<I>(image: &mut I, rect: Rect, color: I::Pixel)
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let left = rect.left() as f32;
    let right = rect.right() as f32;
    let top = rect.top() as f32;
    let bottom = rect.bottom() as f32;

    draw_line_segment_mut(image, (left, top), (right, top), color);
    draw_line_segment_mut(image, (left, bottom), (right, bottom), color);
    draw_line_segment_mut(image, (left, top), (left, bottom), color);
    draw_line_segment_mut(image, (right, top), (right, bottom), color);
}

/// Draw as much of a rectangle, including its boundary, as lies inside the image bounds.
pub fn draw_filled_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_filled_rect_mut(&mut out, rect, color);
    out
}

/// Draw as much of a rectangle, including its boundary, as lies inside the image bounds.
pub fn draw_filled_rect_mut<I>(image: &mut I, rect: Rect, color: I::Pixel)
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let image_bounds = Rect::at(0, 0).of_size(image.width(), image.height());
    if let Some(intersection) = image_bounds.intersect(rect) {
        for dy in 0..intersection.height() {
            for dx in 0..intersection.width() {
                let x = intersection.left() as u32 + dx;
                let y = intersection.top() as u32 + dy;
                unsafe {
                    image.unsafe_put_pixel(x, y, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::rect::Rect;
    use image::{GrayImage, Luma, RgbImage, Rgb};
    use test::{Bencher, black_box};

    #[bench]
    fn bench_draw_filled_rect_mut_rgb(b: &mut Bencher) {
        let mut image = RgbImage::new(200, 200);
        let color = Rgb([120u8, 60u8, 47u8]);
        let rect = Rect::at(50, 50).of_size(80, 90);
        b.iter(|| {
            draw_filled_rect_mut(&mut image, rect, color);
            black_box(&image);
        });
    }

    #[test]
    fn test_draw_hollow_rect() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 4, 4, 4;
            1, 1, 4, 1, 4;
            1, 1, 4, 4, 4);

        let actual = draw_hollow_rect(&image, Rect::at(2, 2).of_size(3, 3), Luma([4u8]));
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_draw_filled_rect() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 4, 4, 4, 1;
            1, 4, 4, 4, 1;
            1, 4, 4, 4, 1;
            1, 1, 1, 1, 1);

        let actual = draw_filled_rect(&image, Rect::at(1, 1).of_size(3, 3), Luma([4u8]));
        assert_pixels_eq!(actual, expected);
    }
}
