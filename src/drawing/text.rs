use crate::definitions::{Clamp, Image};
use crate::drawing::Canvas;
use conv::ValueInto;
use image::{GenericImage, ImageBuffer, Pixel};
use std::f32;

use crate::pixelops::weighted_sum;

use ab_glyph::{point, Font, GlyphId, OutlinedGlyph, PxScale, Rect, ScaleFont};

fn layout_glyphs(
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
    mut f: impl FnMut(OutlinedGlyph, Rect),
) -> (u32, u32) {
    let (mut w, mut h) = (0f32, 0f32);

    let font = font.as_scaled(scale);
    let mut last: Option<GlyphId> = None;

    for c in text.chars() {
        let glyph_id = font.glyph_id(c);
        let glyph = glyph_id.with_scale_and_position(scale, point(w, font.ascent()));
        w += font.h_advance(glyph_id);
        if let Some(g) = font.outline_glyph(glyph) {
            if let Some(last) = last {
                let kern_width = font.kern(glyph_id, last);
                w += kern_width;
            }
            last = Some(glyph_id);
            let bb = g.px_bounds();
            let g_height = bb.height();

            if g_height > h {
                h = g_height;
            }

            f(g, bb);
        }
    }

    (w as u32, h as u32)
}

/// Get the width and height of the given text, rendered with the given font and scale.
///
/// Note that this function *does not* support newlines, you must do this manually.
pub fn text_size(scale: impl Into<PxScale> + Copy, font: &impl Font, text: &str) -> (u32, u32) {
    layout_glyphs(scale, font, text, |_, _| {})
}

/// Draws colored text on an image in place.
///
/// `scale` is augmented font scaling on both the x and y axis (in pixels).
///
/// Note that this function *does not* support newlines, you must do this manually.
pub fn draw_text_mut<C>(
    canvas: &mut C,
    color: C::Pixel,
    x: u32,
    y: u32,
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let image_width = canvas.width();
    let image_height = canvas.height();

    layout_glyphs(scale, font, text, |g, bb| {
        g.draw(|gx, gy, gv| {
            let image_x = gx + x + bb.min.x.round() as u32;
            let image_y = gy + y + bb.min.y.round() as u32;

            if (0..image_width).contains(&image_x) && (0..image_height).contains(&image_y) {
                let pixel = canvas.get_pixel(image_x, image_y);
                let weighted_color = weighted_sum(pixel, color, 1.0 - gv, gv);
                canvas.draw_pixel(image_x, image_y, weighted_color);
            }
        })
    });
}

/// Draws colored text on a new copy of an image.
///
/// `scale` is augmented font scaling on both the x and y axis (in pixels).
///
/// Note that this function *does not* support newlines, you must do this manually.
#[must_use = "the function does not modify the original image"]
pub fn draw_text<I>(
    image: &I,
    color: I::Pixel,
    x: u32,
    y: u32,
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) -> Image<I::Pixel>
where
    I: GenericImage,
    <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_text_mut(&mut out, color, x, y, scale, font, text);
    out
}
