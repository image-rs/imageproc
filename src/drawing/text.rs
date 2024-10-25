use image::{GenericImage, Pixel};
use std::f32;

use crate::definitions::{Clamp, Image};
use crate::drawing::Canvas;
use crate::pixelops::weighted_sum;
use crate::rect::Rect;

use ab_glyph::{point, Font, GlyphId, OutlinedGlyph, PxScale, ScaleFont};

/// The size a text will take up when rendered with the given font and scale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextSizeInfo {
    /// The bounding box of all the pixels of the text. Might be [`None`] if the text is empty or only contains
    /// whitespace.
    ///
    /// As some fonts have glyphs that extend above or below the line, this is not the same as the `outline_bounds`.
    /// Coordinates might be negative or exceed the [`outline_bounds`].
    pub px_bounds: Option<Rect>,

    /// The logical bounding box of the text. Might be [`None`] if the text is empty.
    ///
    /// Can be used to position other text relative to this text.
    pub outline_bounds: Option<Rect>,
}

fn layout_glyphs(
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
    mut f: impl FnMut(OutlinedGlyph, ab_glyph::Rect),
) -> TextSizeInfo {
    let font = font.as_scaled(scale);

    let mut px_bounds: Option<Rect> = None;
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

            if let Some(current) = px_bounds {
                px_bounds = Some(current.union(bb.into()))
            } else {
                px_bounds = Some(bb.into());
            }

            f(g, bb);
        }
    }

    let outline_width = w.ceil() as u32;
    let outline_height = font.height().ceil() as u32;
    let outline_bounds = if outline_height > 0 && outline_width > 0 {
        Some(Rect::at(0, 0).of_size(outline_width, outline_height))
    } else {
        None
    };

    TextSizeInfo {
        px_bounds,
        outline_bounds,
    }
}

/// Get the sizing info of the given text, rendered with the given font and scale.
pub fn text_size_info(
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) -> TextSizeInfo {
    layout_glyphs(scale, font, text, |_, _| {})
}

/// Get the width and height of the given text, rendered with the given font and scale.
pub fn text_size(scale: impl Into<PxScale> + Copy, font: &impl Font, text: &str) -> (u32, u32) {
    let info = text_size_info(scale, font, text);

    let (width, height) = info
        .outline_bounds
        .map(|b| (b.width(), b.height()))
        .unwrap_or_else(|| (0, 0));

    (height, width)
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
) -> TextSizeInfo
where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    let image_width = canvas.width() as i32;
    let image_height = canvas.height() as i32;

    let info = layout_glyphs(scale, font, text, |g, bb| {
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

    TextSizeInfo {
        px_bounds: info.px_bounds.map(|b| b.translate(x, y)),
        outline_bounds: info.outline_bounds.map(|b| b.translate(x, y)),
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{proptest_utils::arbitrary_image_with, rect::Region};
    use ab_glyph::FontRef;
    use image::Luma;
    use proptest::prelude::*;

    const FONT_BYTES: &[u8] = include_bytes!("../../tests/data/fonts/DejaVuSans.ttf");

    proptest! {
        #[test]
        fn proptest_text_size_info(
            mut img in arbitrary_image_with::<Luma<u8>>(Just(0), 0..=100, 0..=100),
            x in 0..100,
            y in 0..100,
            scale in 0.0..100f32,
            ref text in "[0-9a-zA-Z ]*",
        ) {
            let font = FontRef::try_from_slice(FONT_BYTES).unwrap();
            let background = Luma([0]);
            let text_color = Luma([255u8]);

            let draw_info = draw_text_mut(&mut img, text_color, x, y, scale, &font, text);
            let size_info = text_size_info(scale, &font, text);

            let expected_draw_info = TextSizeInfo {
                px_bounds: size_info.px_bounds.map(|r| r.translate(x, y)),
                outline_bounds: size_info.outline_bounds.map(|r| r.translate(x, y)),
            };
            assert_eq!(draw_info, expected_draw_info);

            if text.is_empty() {
                return Ok(());
            }

            let Some(px_bounds) = draw_info.px_bounds else {
                return Ok(());
            };

            for (px, py, &p) in img.enumerate_pixels() {
                if !px_bounds.contains(px as i32, py as i32) {
                    assert_eq!(p, background, "pixel_position: {:?}, rect: {:?}", (px, py), px_bounds);
                }
            }
        }
    }
}
