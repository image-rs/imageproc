//! Helpers for drawing basic shapes on images.

mod bezier;
pub use self::bezier::{draw_cubic_bezier_curve, draw_cubic_bezier_curve_mut};

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
pub use self::polygon::{draw_convex_polygon, draw_convex_polygon_mut, Point};

mod rect;
pub use self::rect::{
    draw_filled_rect, draw_filled_rect_mut, draw_hollow_rect, draw_hollow_rect_mut,
};

mod text;
pub use self::text::{draw_text, draw_text_mut};

use image::GenericImage;

// Set pixel at (x, y) to color if this point lies within image bounds,
// otherwise do nothing.
fn draw_if_in_bounds<I>(image: &mut I, x: i32, y: i32, color: I::Pixel)
where
    I: GenericImage,
    I::Pixel: 'static,
{
    if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}
