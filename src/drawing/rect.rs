use crate::definitions::Image;
use crate::drawing::line::draw_line_segment_mut;
use crate::drawing::Canvas;
use crate::rect_ext::{Rect, RectExt};
use image::GenericImage;
use std::f32;

/// Draws the outline of a rectangle on an image.
///
/// Draws as much of the boundary of the rectangle as lies inside the image bounds.
#[must_use = "the function does not modify the original image"]
pub fn draw_hollow_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = Image::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_hollow_rect_mut(&mut out, rect, color);
    out
}
#[doc=generate_mut_doc_comment!("draw_hollow_rect")]
pub fn draw_hollow_rect_mut<C>(canvas: &mut C, rect: Rect, color: C::Pixel)
where
    C: Canvas,
{
    let left_x = rect.left_x() as f32;
    let right_x = rect.right_x() as f32;
    let top_y = rect.top_y() as f32;
    let bottom_y = rect.bottom_y() as f32;

    draw_line_segment_mut(canvas, (left_x, top_y), (right_x, top_y), color);
    draw_line_segment_mut(canvas, (left_x, bottom_y), (right_x, bottom_y), color);
    draw_line_segment_mut(canvas, (left_x, top_y), (left_x, bottom_y), color);
    draw_line_segment_mut(canvas, (right_x, top_y), (right_x, bottom_y), color);
}

/// Draws a rectangle and its contents on an image.
///
/// Draws as much of the rectangle and its contents as lies inside the image bounds.
#[must_use = "the function does not modify the original image"]
pub fn draw_filled_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = Image::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_filled_rect_mut(&mut out, rect, color);
    out
}
#[doc=generate_mut_doc_comment!("draw_filled_rect")]
pub fn draw_filled_rect_mut<C>(canvas: &mut C, rect: Rect, color: C::Pixel)
where
    C: Canvas,
{
    let canvas_bounds = Rect {
        x: 0,
        y: 0,
        width: canvas.width(),
        height: canvas.height(),
    };
    if let Some(intersection) = canvas_bounds.intersect(&rect) {
        for dy in 0..intersection.height {
            for dx in 0..intersection.width {
                let x = intersection.left_x() + dx;
                let y = intersection.top_y() + dy;
                canvas.draw_pixel(x, y, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawing::Blend;
    use crate::rect_ext::Rect;
    use image::{GrayImage, Luma, Pixel, Rgba, RgbaImage};

    #[test]
    fn test_draw_hollow_rect() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 4, 4, 4;
            1, 1, 4, 1, 4;
            1, 1, 4, 4, 4);

        let actual = draw_hollow_rect(
            &image,
            Rect {
                x: 2,
                y: 2,
                width: 3,
                height: 3,
            },
            Luma([4u8]),
        );
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

        let actual = draw_filled_rect(
            &image,
            Rect {
                x: 1,
                y: 1,
                width: 3,
                height: 3,
            },
            Luma([4u8]),
        );
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_draw_blended_filled_rect() {
        // https://github.com/image-rs/imageproc/issues/261

        let white = Rgba([255u8, 255u8, 255u8, 255u8]);
        let blue = Rgba([0u8, 0u8, 255u8, 255u8]);
        let semi_transparent_red = Rgba([255u8, 0u8, 0u8, 127u8]);

        let mut image = Blend(RgbaImage::from_pixel(5, 5, white));

        draw_filled_rect_mut(
            &mut image,
            Rect {
                x: 1,
                y: 1,
                width: 3,
                height: 3,
            },
            blue,
        );
        draw_filled_rect_mut(
            &mut image,
            Rect {
                x: 2,
                y: 2,
                width: 1,
                height: 1,
            },
            semi_transparent_red,
        );

        // The central pixel should be blended
        let mut blended = blue;
        blended.blend(&semi_transparent_red);

        #[rustfmt::skip]
        let expected = [white, white,   white, white, white,
            white,  blue,    blue,  blue, white,
            white,  blue, blended,  blue, white,
            white,  blue,    blue,  blue, white,
            white, white,   white, white, white];
        let expected = RgbaImage::from_fn(5, 5, |x, y| expected[(y * 5 + x) as usize]);

        assert_pixels_eq!(image.0, expected);

        // Draw an opaque rectangle over the central pixel as a sanity check that
        // we're blending in the correct direction only.
        draw_filled_rect_mut(
            &mut image,
            Rect {
                x: 2,
                y: 2,
                width: 1,
                height: 1,
            },
            blue,
        );
        assert_eq!(*image.0.get_pixel(2, 2), blue);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::rect_ext::Rect;
    use image::{Rgb, RgbImage};
    use test::{black_box, Bencher};

    #[bench]
    fn bench_draw_filled_rect_mut_rgb(b: &mut Bencher) {
        let mut image = RgbImage::new(200, 200);
        let color = Rgb([120u8, 60u8, 47u8]);
        let rect = Rect {
            x: 50,
            y: 50,
            width: 80,
            height: 90,
        };
        b.iter(|| {
            draw_filled_rect_mut(&mut image, rect, color);
            black_box(&image);
        });
    }
}
