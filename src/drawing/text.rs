use image::{GenericImage, Pixel};
use std::f32;

use crate::definitions::{Clamp, Image};
use crate::drawing::Canvas;
use crate::pixelops::weighted_sum;

use ab_glyph::{point, Font, GlyphId, OutlinedGlyph, PxScale, Rect, ScaleFont};

fn layout_glyphs(
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
    mut f: impl FnMut(OutlinedGlyph, Rect),
) -> (u32, u32) {
    if text.is_empty() {
        return (0, 0);
    }
    let font = font.as_scaled(scale);

    let mut w = 0.0;
    let mut prev: Option<GlyphId> = None;

    for c in text.chars() {
        let glyph_id = font.glyph_id(c);
        let glyph = glyph_id.with_scale_and_position(scale, point(w, font.ascent()));
        w += font.h_advance(glyph_id);
        if let Some(g) = font.outline_glyph(glyph) {
            if let Some(prev) = prev {
                w += font.kern(glyph_id, prev);
            }
            prev = Some(glyph_id);
            let bb = g.px_bounds();
            f(g, bb);
        }
    }

    let w = w.ceil();
    let h = font.height().ceil();
    assert!(w >= 0.0);
    assert!(h >= 0.0);
    (w as u32, h as u32)
}

/// Get the width and height of the given text, rendered with the given font and scale.
pub fn text_size(scale: impl Into<PxScale> + Copy, font: &impl Font, text: &str) -> (u32, u32) {
    layout_glyphs(scale, font, text, |_, _| {})
}

/// Draws colored text on an image.
///
/// `scale` is augmented font scaling on both the x and y axis (in pixels).
///
/// Note that this function *does not* support newlines, you must do this manually.
#[must_use = "the function does not modify the original image"]
pub fn draw_text<I>(
    image: &I,
    color: I::Pixel,
    x: i32,
    y: i32,
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) -> Image<I::Pixel>
where
    I: GenericImage,
    <I::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    let mut out = Image::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_text_mut(&mut out, color, x, y, scale, font, text);
    out
}

#[doc=generate_mut_doc_comment!("draw_text")]
pub fn draw_text_mut<C>(
    canvas: &mut C,
    color: C::Pixel,
    x: i32,
    y: i32,
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    let image_width = canvas.width() as i32;
    let image_height = canvas.height() as i32;

    layout_glyphs(scale, font, text, |g, bb| {
        let x_shift = x + bb.min.x.round() as i32;
        let y_shift = y + bb.min.y.round() as i32;
        g.draw(|gx, gy, gv| {
            let image_x = gx as i32 + x_shift;
            let image_y = gy as i32 + y_shift;

            if (0..image_width).contains(&image_x) && (0..image_height).contains(&image_y) {
                let image_x = image_x as u32;
                let image_y = image_y as u32;
                let pixel = canvas.get_pixel(image_x, image_y);
                let gv = gv.clamp(0.0, 1.0);
                let weighted_color = weighted_sum(pixel, color, 1.0 - gv, gv);
                canvas.draw_pixel(image_x, image_y, weighted_color);
            }
        })
    });
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{
        proptest_utils::arbitrary_image_with,
        rect::{Rect, Region},
    };
    use ab_glyph::FontRef;
    use image::Luma;
    use proptest::prelude::*;

    const FONT_BYTES: &[u8] = include_bytes!("../../tests/data/fonts/DejaVuSans.ttf");

    proptest! {
        #[test]
        fn proptest_text_size(
            img in arbitrary_image_with::<Luma<u8>>(Just(0), 0..=100, 0..=100),
            x in 0..100,
            y in 0..100,
            scale in 0.0..100f32,
            ref text in "[0-9a-zA-Z]*",
        ) {
            let font = FontRef::try_from_slice(FONT_BYTES).unwrap();
            let background = Luma([0]);
            let text_color = Luma([255u8]);

            let img = draw_text(&img, text_color, x, y, scale, &font, text);

            let (text_w, text_h) = text_size(scale, &font, text);
            // TODO: fix Rect::contains by making Rect a "closed set"
            let (text_w, text_h) = (text_w + 1, text_h + 1);

            let rect = if text.is_empty() {
                Rect::at(x, y).of_size(text_w, text_h)
            } else {
                let first_char = text.chars().next().unwrap();
                let first_x_bearing =
                    font.as_scaled(scale).h_side_bearing(font.glyph_id(first_char));
                if first_x_bearing < 0.0 {
                    let x_shift = first_x_bearing.abs().ceil() as i32;
                    Rect::at(x - x_shift, y).of_size(text_w, text_h)
                } else {
                    Rect::at(x, y).of_size(text_w, text_h)
                }
            };
            for (px, py, &p) in img.enumerate_pixels() {
                if !rect.contains(px as i32, py as i32) {
                    assert_eq!(p, background, "pixel_position: {:?}, rect: {:?}", (px, py), rect);
                }
            }
        }
    }
}
