//! Helpers for drawing basic shapes on images.

mod bezier;
pub use self::bezier::{
    draw_cubic_bezier_curve,
    draw_cubic_bezier_curve_mut
};

mod canvas;
pub use self::canvas::{
    Blend,
    Canvas
};

mod conics;
pub use self::conics::{
    draw_hollow_ellipse,
    draw_hollow_ellipse_mut,
    draw_filled_ellipse,
    draw_filled_ellipse_mut,
    draw_hollow_circle,
    draw_hollow_circle_mut,
    draw_filled_circle,
    draw_filled_circle_mut
};

mod cross;
pub use self::cross::{
    draw_cross,
    draw_cross_mut
};

mod line;
pub use self::line::{
    BresenhamLineIter,
    BresenhamLinePixelIter,
    BresenhamLinePixelIterMut,
    draw_line_segment,
    draw_line_segment_mut,
    draw_antialiased_line_segment,
    draw_antialiased_line_segment_mut
};

mod polygon;
pub use self::polygon::{
    Point,
    draw_convex_polygon,
    draw_convex_polygon_mut
};

mod rect;
pub use self::rect::{
    draw_hollow_rect,
    draw_hollow_rect_mut,
    draw_filled_rect,
    draw_filled_rect_mut
};

mod text;
pub use self::text::{
    draw_text,
    draw_text_mut
};

// Set pixel at (x, y) to color if this point lies within image bounds,
// otherwise do nothing.
fn draw_if_in_bounds<C>(canvas: &mut C, x: i32, y: i32, color: C::Pixel)
where
    C: Canvas,
    C::Pixel: 'static,
{
    if x >= 0 && x < canvas.width() as i32 && y >= 0 && y < canvas.height() as i32 {
        canvas.draw_pixel(x as u32, y as u32, color);
    }
}
