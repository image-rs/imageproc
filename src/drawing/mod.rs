//! Helpers for drawing basic shapes on images.
//!
//! Every `draw_` function comes in two variants: one creates a new copy of the input image, one modifies the image in place.
//! The latter is more memory efficient, but you lose the original image.

mod bezier;
pub use self::bezier::{draw_cubic_bezier_curve, draw_cubic_bezier_curve_mut};

mod canvas;
pub use self::canvas::{Blend, Canvas};

mod conics;
pub use self::conics::{
    draw_filled_circle, draw_filled_circle_mut, draw_filled_ellipse, draw_filled_ellipse_mut,
    draw_hollow_circle, draw_hollow_circle_mut, draw_hollow_ellipse, draw_hollow_ellipse_mut,
};

mod cross;
pub use self::cross::{draw_cross, draw_cross_mut};

mod line;
pub use self::line::{
    draw_antialiased_line_segment, draw_antialiased_line_segment_mut, draw_line_segment,
    draw_line_segment_mut, BresenhamLineIter, BresenhamLinePixelIter, BresenhamLinePixelIterMut,
};

mod polygon;
pub use self::polygon::{
    draw_antialiased_polygon, draw_antialiased_polygon_mut, draw_hollow_polygon,
    draw_hollow_polygon_mut, draw_polygon, draw_polygon_mut,
};

mod rect;
pub use self::rect::{
    draw_filled_rect, draw_filled_rect_mut, draw_hollow_rect, draw_hollow_rect_mut,
};

mod text;
pub use self::text::{draw_text, draw_text_mut, text_size};

// Set pixel at (x, y) to color if this point lies within image bounds,
// otherwise do nothing.
fn draw_if_in_bounds<C>(canvas: &mut C, x: i32, y: i32, color: C::Pixel)
where
    C: Canvas,
{
    if x >= 0 && x < canvas.width() as i32 && y >= 0 && y < canvas.height() as i32 {
        canvas.draw_pixel(x as u32, y as u32, color);
    }
}
