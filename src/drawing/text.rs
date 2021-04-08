use crate::definitions::{Clamp, Image};
use crate::drawing::Canvas;
use conv::ValueInto;
use image::{GenericImage, ImageBuffer, Pixel};
use std::f32;
use std::i32;

use crate::pixelops::weighted_sum;
use rusttype::{point, Font, PositionedGlyph, Rect, Scale};
use std::cmp::max;

fn layout_glyphs(
    scale: Scale,
    font: &Font,
    text: &str,
    mut f: impl FnMut(PositionedGlyph, Rect<i32>),
) -> (i32, i32) {
    let v_metrics = font.v_metrics(scale);

    let (mut w, mut h) = (0, 0);

    for g in font.layout(text, scale, point(0.0, v_metrics.ascent)) {
        if let Some(bb) = g.pixel_bounding_box() {
            w = max(w, bb.max.x);
            h = max(h, bb.max.y);
            f(g, bb);
        }
    }

    (w, h)
}

/// Get the width and height of the given text, rendered with the given font and scale. Note that this function *does not* support newlines, you must do this manually.
pub fn text_size(scale: Scale, font: &Font, text: &str) -> (i32, i32) {
    layout_glyphs(scale, font, text, |_, _| {})
}

/// Draws colored text on an image in place. `scale` is augmented font scaling on both the x and y axis (in pixels). Note that this function *does not* support newlines, you must do this manually.
pub fn draw_text_mut<'a, C>(
    canvas: &'a mut C,
    color: C::Pixel,
    x: i32,
    y: i32,
    scale: Scale,
    font: &'a Font<'a>,
    text: &'a str,
) where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let image_width = canvas.width() as i32;
    let image_height = canvas.height() as i32;

    layout_glyphs(scale, font, text, |g, bb| {
        g.draw(|gx, gy, gv| {
            let gx = gx as i32 + bb.min.x;
            let gy = gy as i32 + bb.min.y;

            let image_x = gx + x;
            let image_y = gy + y;

            if (0..image_width).contains(&image_x) && (0..image_height).contains(&image_y) {
                let pixel = canvas.get_pixel(image_x as u32, image_y as u32);
                let weighted_color = weighted_sum(pixel, color, 1.0 - gv, gv);
                canvas.draw_pixel(image_x as u32, image_y as u32, weighted_color);
            }
        })
    });
}

/// Draws colored text on an image in place. `scale` is augmented font scaling on both the x and y axis (in pixels). Note that this function *does not* support newlines, you must do this manually.
pub fn draw_text<'a, I>(
    image: &'a mut I,
    color: I::Pixel,
    x: i32,
    y: i32,
    scale: Scale,
    font: &'a Font<'a>,
    text: &'a str,
) -> Image<I::Pixel>
where
    I: GenericImage,
    <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_text_mut(&mut out, color, x, y, scale, font, text);
    out
}
