use crate::definitions::Image;
use crate::drawing::line::{draw_line_segment_mut, draw_antialiased_line_segment_mut};
use crate::drawing::Canvas;
use crate::point::Point;
use image::{GenericImage, ImageBuffer};
use std::cmp::{max, min};
use std::f32;
use std::i32;

#[must_use = "the function does not modify the original image"]
fn draw_polygon_with<I, L>(image: &I, poly: &[Point<i32>], color: I::Pixel, plotter: L) -> Image<I::Pixel>
where
    I: GenericImage,
    L: Fn(&mut Image<I::Pixel>, (f32, f32), (f32, f32), I::Pixel) -> (),
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_polygon_with_mut(&mut out, poly, color, plotter);
    out
}

fn draw_polygon_with_mut<C, L>(canvas: &mut C, poly: &[Point<i32>], color: C::Pixel, plotter: L)
where
    C: Canvas,
    L: Fn(&mut C, (f32, f32), (f32, f32), C::Pixel) -> (),
{
    if poly.is_empty() {
        return;
    }
    if poly[0] == poly[poly.len() - 1] {
        panic!(
            "First point {:?} == last point {:?}",
            poly[0],
            poly[poly.len() - 1]
        );
    }

    let mut y_min = i32::MAX;
    let mut y_max = i32::MIN;
    for p in poly {
        y_min = min(y_min, p.y);
        y_max = max(y_max, p.y);
    }

    let (width, height) = canvas.dimensions();

    // Intersect polygon vertical range with image bounds
    y_min = max(0, min(y_min, height as i32 - 1));
    y_max = max(0, min(y_max, height as i32 - 1));

    let mut closed: Vec<Point<i32>> = poly.to_vec();
    closed.push(poly[0]);

    let edges: Vec<&[Point<i32>]> = closed.windows(2).collect();
    let mut intersections = Vec::new();

    for y in y_min..y_max + 1 {
        for edge in &edges {
            let p0 = edge[0];
            let p1 = edge[1];

            if p0.y <= y && p1.y >= y || p1.y <= y && p0.y >= y {
                if p0.y == p1.y {
                    // Need to handle horizontal lines specially
                    intersections.push(p0.x);
                    intersections.push(p1.x);
                } else if p0.y == y || p1.y == y {
                    if p1.y > y {
                        intersections.push(p0.x);
                    }
                    if p0.y > y {
                        intersections.push(p1.x);
                    }
                } else {
                    let fraction = (y - p0.y) as f32 / (p1.y - p0.y) as f32;
                    let inter = p0.x as f32 + fraction * (p1.x - p0.x) as f32;
                    intersections.push(inter.round() as i32);
                }
            }
        }

        intersections.sort_unstable();
        intersections.chunks(2).for_each(|range| {
            let mut from = min(range[0], width as i32);
            let mut to = min(range[1], width as i32 - 1);
            if from < width as i32 && to >= 0 {
                // draw only if range appears on the canvas
                from = max(0, from);
                to = max(0, to);

                for x in from..to + 1 {
                    canvas.draw_pixel(x as u32, y as u32, color);
                }
            }
        });

        intersections.clear();

        for edge in &edges {
            let start = (edge[0].x as f32, edge[0].y as f32);
            let end = (edge[1].x as f32, edge[1].y as f32);
            plotter(canvas, start, end, color);
        }
    }
}

/// Draws a polygon and its contents on a new copy of an image.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
pub fn draw_polygon<I>(image: &I, poly: &[Point<i32>], color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
{
    draw_polygon_with(image, poly, color, draw_line_segment_mut)
}

/// Draws a polygon and its contents on an image in place.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
pub fn draw_polygon_mut<C>(canvas: &mut C, poly: &[Point<i32>], color: C::Pixel)
where
    C: Canvas,
{
    draw_polygon_with_mut(canvas, poly, color, draw_line_segment_mut);
}

/// Draws a polygon and its contents on a new copy of an image, without edges.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
pub fn draw_unbordered_polygon<I>(image: &I, poly: &[Point<i32>], color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
{
    // plotter does a no-op
    draw_polygon_with(image, poly, color, |_, _, _, _| ())
}

/// Draws a polygon and its contents on an image in place, without edges.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
pub fn draw_unbordered_polygon_mut<C>(canvas: &mut C, poly: &[Point<i32>], color: C::Pixel)
where
    C: Canvas,
{
    draw_polygon_with_mut(canvas, poly, color, |_, _, _, _| ());
}

/// Draws an anti-aliased polygon polygon and its contents on a new copy of an image.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
///
/// The parameters of blend are (line color, original color, line weight).
/// Consider using [`interpolate`](fn.interpolate.html) for blend.
pub fn draw_antialiased_polygon<I, B>(image: &I, poly: &[Point<i32>], color: I::Pixel, blend: B) -> Image<I::Pixel>
where
    I: GenericImage,
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    draw_polygon_with(image, poly, color, |image, start, end, color|
        draw_antialiased_line_segment_mut(image, (start.0 as i32, start.1 as i32), (end.0 as i32, end.1 as i32), color, &blend)
    )
}

/// Draws an anti-aliased polygon and its contents on an image in place.
///
/// Draws as much of a filled polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
///
/// The parameters of blend are (line color, original color, line weight).
/// Consider using [`interpolate`](fn.interpolate.html) for blend.
pub fn draw_antialiased_polygon_mut<I, B>(image: &mut I, poly: &[Point<i32>], color: I::Pixel, blend: B)
where
    I: GenericImage,
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    draw_polygon_with_mut(image, poly, color, |image, start, end, color|
        draw_antialiased_line_segment_mut(image, (start.0 as i32, start.1 as i32), (end.0 as i32, end.1 as i32), color, &blend)
    );
}
