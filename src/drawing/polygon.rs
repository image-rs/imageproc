use image::{GenericImage, ImageBuffer};
use crate::definitions::Image;
use crate::drawing::Canvas;
use std::cmp::{min, max};
use std::f32;
use std::i32;
use crate::drawing::draw_if_in_bounds;
use crate::drawing::line::draw_line_segment_mut;

/// A 2D point.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point<T: Copy + PartialEq + Eq> {
    x: T,
    y: T,
}

impl<T: Copy + PartialEq + Eq> Point<T> {
    /// Construct a point at (x, y).
    pub fn new(x: T, y: T) -> Point<T> {
        Point::<T> { x, y }
    }
}

/// Draws as much of a filled convex polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
///
/// Does not validate that input is convex.
pub fn draw_convex_polygon<I>(image: &I, poly: &[Point<i32>], color: I::Pixel) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_convex_polygon_mut(&mut out, poly, color);
    out
}

/// Draws as much of a filled convex polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
///
/// Does not validate that input is convex.
pub fn draw_convex_polygon_mut<C>(canvas: &mut C, poly: &[Point<i32>], color: C::Pixel)
where
    C: Canvas,
    C::Pixel: 'static,
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

    let mut closed = Vec::with_capacity(poly.len() + 1);
    for p in poly {
        closed.push(*p);
    }
    closed.push(poly[0]);

    let edges: Vec<&[Point<i32>]> = closed.windows(2).collect();
    let mut intersections: Vec<i32> = Vec::new();

    for y in y_min..y_max + 1 {
        for edge in &edges {
            let p0 = edge[0];
            let p1 = edge[1];

            if p0.y <= y && p1.y >= y || p1.y <= y && p0.y >= y {
                // Need to handle horizontal lines specially
                if p0.y == p1.y {
                    intersections.push(p0.x);
                    intersections.push(p1.x);
                } else {
                    let fraction = (y - p0.y) as f32 / (p1.y - p0.y) as f32;
                    let inter = p0.x as f32 + fraction * (p1.x - p0.x) as f32;
                    intersections.push(inter.round() as i32);
                }
            }
        }

        intersections.sort();
        let mut i = 0;
        loop {
            // Handle points where multiple lines intersect
            while i + 1 < intersections.len() && intersections[i] == intersections[i + 1] {
                i += 1;
            }
            if i >= intersections.len() {
                break;
            }
            if i + 1 == intersections.len() {
                draw_if_in_bounds(canvas, intersections[i], y, color);
                break;
            }
            let from = max(0, min(intersections[i], width as i32 - 1));
            let to = max(0, min(intersections[i + 1], width as i32 - 1));
            for x in from..to + 1 {
                canvas.draw_pixel(x as u32, y as u32, color);
            }
            i += 2;
        }

        intersections.clear();
    }

    for edge in &edges {
        let start = (edge[0].x as f32, edge[0].y as f32);
        let end = (edge[1].x as f32, edge[1].y as f32);
        draw_line_segment_mut(canvas, start, end, color);
    }
}
