use image::{GenericImage, ImageBuffer, Pixel};
use definitions::{VecBuffer, Clamp, Image};
use conv::ValueInto;
use rect::Rect;
use std::mem::swap;
use std::cmp::{min, max};
use std::f32;
use std::i32;

use pixelops::weighted_sum;
use rusttype::{FontCollection, Scale, point, PositionedGlyph};

/// Draws colored text on an image in place. `height` is your desired font height (in pixels), `scale` is augmented font scaling on both the x and y axis (in pixels). Note that this function *does not* support newlines, you must do this manually
pub fn draw_text_mut<I>(image: &mut I, color: I::Pixel, x: u32, y: u32, height: f32, scale: Scale, font_data: &[u8], text: &str) 
    where I: GenericImage, <I::Pixel as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{

    let collection = FontCollection::from_bytes(font_data);
    let font = collection.into_font().unwrap();

    let pixel_height = height.ceil() as usize;
    let v_metrics = font.v_metrics(scale);
    let offset = point(0.0, v_metrics.ascent);

    let glyphs: Vec<PositionedGlyph> = font.layout(text, scale, offset).collect();

    let width = glyphs.iter().rev()
        .filter_map(|g| g.pixel_bounding_box()
                    .map(|b| b.min.x as f32 + g.unpositioned().h_metrics().advance_width))
        .next().unwrap_or(0.0).ceil() as usize;

    for g in glyphs {
        if let Some(bb) = g.pixel_bounding_box() {
            g.draw(|gx, gy, gv| {
                let gx = gx as i32 + bb.min.x;
                let gy = gy as i32 + bb.min.y;
                
                if gx >= 0 && gx < width as i32 && gy >= 0 && gy < pixel_height as i32 {

                  let pixel = image.get_pixel(gx as u32 + x, gy as u32 + y);
                  let weighted_color = weighted_sum(pixel, color, 1.0 - gv, gv);
                  image.put_pixel(gx as u32 + x, gy as u32 + y, weighted_color);
                }
            })
        }
    }
}

/// Draws a colored cross on an image in place. Handles coordinates outside image bounds.
#[cfg_attr(rustfmt, rustfmt_skip)]
pub fn draw_cross_mut<I>(image: &mut I, color: I::Pixel, x: i32, y: i32)
    where I: GenericImage
{
    let (width, height) = image.dimensions();
    let idx = |x, y| (3 * (y + 1) + x + 1) as usize;
    let stencil = [0u8, 1u8, 0u8,
                   1u8, 1u8, 1u8,
                   0u8, 1u8, 0u8];

    for sy in -1..2 {
        let iy = y + sy;
        if iy < 0 || iy >= height as i32 {
            continue;
        }

        for sx in -1..2 {
            let ix = x + sx;
            if ix < 0 || ix >= width as i32 {
                continue;
            }

            if stencil[idx(sx, sy)] == 1u8 {
                // bound checks already done
                unsafe { image.unsafe_put_pixel(ix as u32, iy as u32, color); }
            }
        }
    }
}

/// Draws a colored cross on an image. Handles coordinates outside image bounds.
pub fn draw_cross<I>(image: &I, color: I::Pixel, x: i32, y: i32) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_cross_mut(&mut out, color, x, y);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// Uses [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub fn draw_line_segment<I>(image: &I,
                            start: (f32, f32),
                            end: (f32, f32),
                            color: I::Pixel)
                            -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_line_segment_mut(&mut out, start, end, color);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// Uses [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub fn draw_line_segment_mut<I>(image: &mut I, start: (f32, f32), end: (f32, f32), color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    let (width, height) = image.dimensions();
    let in_bounds = |x, y| x >= 0 && x < width as i32 && y >= 0 && y < height as i32;

    let (mut x0, mut y0) = (start.0, start.1);
    let (mut x1, mut y1) = (end.0, end.1);

    let is_steep = (y1 - y0).abs() > (x1 - x0).abs();

    if is_steep {
        swap(&mut x0, &mut y0);
        swap(&mut x1, &mut y1);
    }

    if x0 > x1 {
        swap(&mut x0, &mut x1);
        swap(&mut y0, &mut y1);
    }

    let dx = x1 - x0;
    let dy = (y1 - y0).abs();
    let mut error = dx / 2f32;

    let y_step = if y0 < y1 { 1f32 } else { -1f32 };
    let mut y = y0 as i32;

    for x in x0 as i32..(x1 + 1f32) as i32 {
        unsafe {
            if is_steep {
                if in_bounds(y, x) {
                    image.unsafe_put_pixel(y as u32, x as u32, color);
                }
            } else if in_bounds(x, y) {
                image.unsafe_put_pixel(x as u32, y as u32, color);
            }
        }
        error -= dy;
        if error < 0f32 {
            y = (y as f32 + y_step) as i32;
            error += dx;
        }
    }
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// The parameters of blend are (line color, original color, line weight).
/// Uses [Xu's line drawing algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm).
pub fn draw_antialiased_line_segment<I, B>(image: &I,
                                           start: (i32, i32),
                                           end: (i32, i32),
                                           color: I::Pixel,
                                           blend: B)
                                           -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static,
          B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_antialiased_line_segment_mut(&mut out, start, end, color, blend);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// The parameters of blend are (line color, original color, line weight).
/// Uses [Xu's line drawing algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm).
pub fn draw_antialiased_line_segment_mut<I, B>(image: &mut I,
                                               start: (i32, i32),
                                               end: (i32, i32),
                                               color: I::Pixel,
                                               blend: B)
    where I: GenericImage,
          I::Pixel: 'static,
          B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel
{
    let (mut x0, mut y0) = (start.0, start.1);
    let (mut x1, mut y1) = (end.0, end.1);

    let is_steep = (y1 - y0).abs() > (x1 - x0).abs();

    if is_steep {
        if y0 > y1 {
            swap(&mut x0, &mut x1);
            swap(&mut y0, &mut y1);
        }
        let plotter = Plotter { image: image, transform: |x, y| (y, x), blend: blend };
        plot_wu_line(plotter, (y0, x0), (y1, x1), color);
    } else {
        if x0 > x1 {
            swap(&mut x0, &mut x1);
            swap(&mut y0, &mut y1);
        }
        let plotter = Plotter { image: image, transform: |x, y| (x, y), blend: blend };
        plot_wu_line(plotter, (x0, y0), (x1, y1), color);
    };
}

fn plot_wu_line<I, T, B>(mut plotter: Plotter<I, T, B>,
                         start: (i32, i32),
                         end: (i32, i32),
                         color: I::Pixel)
    where I: GenericImage, I::Pixel: 'static,
          T: Fn(i32, i32) -> (i32, i32),
          B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel
{
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let gradient = dy as f32 / dx as f32;
    let mut fy = start.1 as f32;

    for x in start.0..(end.0 + 1) {
        plotter.plot(x, fy as i32, color, 1.0 - fy.fract());
        plotter.plot(x, fy as i32 + 1, color, fy.fract());
        fy += gradient;
    }
}

struct Plotter<'a, I: 'a, T, B>
    where I: GenericImage, I::Pixel: 'static,
          T: Fn(i32, i32) -> (i32, i32),
          B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel
{
    image: &'a mut I,
    transform: T,
    blend: B
}

impl<'a, I, T, B> Plotter<'a, I, T, B>
    where I: GenericImage, I::Pixel: 'static,
          T: Fn(i32, i32) -> (i32, i32),
          B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel
{
    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.image.width() as i32 && y >= 0 && y < self.image.height() as i32
    }

    pub fn plot(&mut self, x: i32, y: i32, line_color: I::Pixel, line_weight: f32) {
        let (x_trans, y_trans) = (self.transform)(x, y);
        if self.in_bounds(x_trans, y_trans) {
            let original = self.image.get_pixel(x_trans as u32, y_trans as u32);
            let blended = (self.blend)(line_color, original, line_weight);
            self.image.put_pixel(x_trans as u32, y_trans as u32, blended);
        }
    }
}

/// Draws as much of the boundary of a rectangle as lies inside the image bounds.
pub fn draw_hollow_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_hollow_rect_mut(&mut out, rect, color);
    out
}

/// Draws as much of the boundary of a rectangle as lies inside the image bounds.
pub fn draw_hollow_rect_mut<I>(image: &mut I, rect: Rect, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    let left = rect.left() as f32;
    let right = rect.right() as f32;
    let top = rect.top() as f32;
    let bottom = rect.bottom() as f32;

    draw_line_segment_mut(image, (left, top), (right, top), color);
    draw_line_segment_mut(image, (left, bottom), (right, bottom), color);
    draw_line_segment_mut(image, (left, top), (left, bottom), color);
    draw_line_segment_mut(image, (right, top), (right, bottom), color);
}

/// Draw as much of a rectangle, including its boundary, as lies inside the image bounds.
pub fn draw_filled_rect<I>(image: &I, rect: Rect, color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_filled_rect_mut(&mut out, rect, color);
    out
}

/// Draw as much of a rectangle, including its boundary, as lies inside the image bounds.
pub fn draw_filled_rect_mut<I>(image: &mut I, rect: Rect, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    let image_bounds = Rect::at(0, 0).of_size(image.width(), image.height());
    if let Some(intersection) = image_bounds.intersect(rect) {
        for dy in 0..intersection.height() {
            for dx in 0..intersection.width() {
                let x = intersection.left() as u32 + dx;
                let y = intersection.top() as u32 + dy;
                unsafe { image.unsafe_put_pixel(x, y, color); }
            }
        }
    }
}

/// Draw as much of an ellipse as lies inside the image bounds.
/// Uses Midpoint Ellipse Drawing Algorithm. (Modified from Bresenham's algorithm) (http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// (`x^2 / width_radius^2) + (y^2 / height_radius^2) = 1`
pub fn draw_hollow_ellipse<I>(image: &I,
                              center: (i32, i32),
                              width_radius: i32,
                              height_radius: i32,
                              color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_hollow_ellipse_mut(&mut out, center, width_radius, height_radius, color);
    out
}

/// Draw as much of an ellipse as lies inside the image bounds.
/// Uses Midpoint Ellipse Drawing Algorithm. (Modified from Bresenham's algorithm) (http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) = 1`
pub fn draw_hollow_ellipse_mut<I>(image: &mut I, center: (i32, i32), width_radius: i32, height_radius: i32, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    // Circle drawing algorithm is faster, so use it if the given ellipse is actually a circle.
    if width_radius == height_radius {
        draw_hollow_circle_mut(image, center, width_radius, color);
        return;
    }

    let draw_quad_pixels = |x0: i32, y0: i32, x: i32, y: i32| {
        draw_if_in_bounds(image, x0 + x, y0 + y, color);
        draw_if_in_bounds(image, x0 - x, y0 + y, color);
        draw_if_in_bounds(image, x0 + x, y0 - y, color);
        draw_if_in_bounds(image, x0 - x, y0 - y, color);
    };

    draw_ellipse(draw_quad_pixels, center, width_radius, height_radius);
}

/// Draw as much of an ellipse, including its contents, as lies inside the image bounds.
/// Uses Midpoint Ellipse Drawing Algorithm. (Modified from Bresenham's algorithm) (http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) <= 1`
pub fn draw_filled_ellipse<I>(image: &I,
                              center: (i32, i32),
                              width_radius: i32,
                              height_radius: i32,
                              color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_filled_ellipse_mut(&mut out, center, width_radius, height_radius, color);
    out
}

/// Draw as much of an ellipse, including its contents, as lies inside the image bounds.
/// Uses Midpoint Ellipse Drawing Algorithm. (Modified from Bresenham's algorithm) (http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) <= 1`
pub fn draw_filled_ellipse_mut<I>(image: &mut I, center: (i32, i32), width_radius: i32, height_radius: i32, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    // Circle drawing algorithm is faster, so use it if the given ellipse is actually a circle.
    if width_radius == height_radius {
        draw_filled_circle_mut(image, center, width_radius, color);
        return;
    }

    let draw_line_pairs = |x0: i32, y0: i32, x: i32, y: i32| {
        draw_line_segment_mut(image, ((x0 - x) as f32, (y0 + y) as f32), ((x0 + x) as f32, (y0 + y) as f32), color);
        draw_line_segment_mut(image, ((x0 - x) as f32, (y0 - y) as f32), ((x0 + x) as f32, (y0 - y) as f32), color);
    };

    draw_ellipse(draw_line_pairs, center, width_radius, height_radius);
}

// Implements the Midpoint Ellipse Drawing Algorithm. (Modified from Bresenham's algorithm) (http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/)
// Takes a function that determines how to render the points on the ellipse.
fn draw_ellipse<F>(mut render_func: F, center: (i32, i32), width_radius: i32, height_radius: i32)
    where F: FnMut(i32, i32, i32, i32)
{
    let (x0, y0) = center;
    let w2 = width_radius * width_radius;
    let h2 = height_radius * height_radius;
    let mut x = 0;
    let mut y = height_radius;
    let mut px = 0;
    let mut py = 2 * w2 * y;

    render_func(x0, y0, x, y);

    // Top and bottom regions.
    let mut p = (h2 - (w2 * height_radius)) as f32 + (0.25 * w2 as f32);
    while px < py {
        x += 1;
        px += 2 * h2;
        if p < 0.0 {
            p += (h2 + px) as f32;
        } else {
            y -= 1;
            py += -2 * w2;
            p += (h2 + px - py) as f32;
        }

        render_func(x0, y0, x, y);
    }

    // Left and right regions.
    p = (h2 as f32) * (x as f32 + 0.5).powi(2) + (w2 * (y - 1).pow(2)) as f32 - (w2 * h2) as f32;
    while y > 0 {
        y -= 1;
        py += -2 * w2;
        if p > 0.0 {
            p += (w2 - py) as f32;
        } else {
            x += 1;
            px += 2 * h2;
            p += (w2 - py + px) as f32;
        }

        render_func(x0, y0, x, y);
    }
}

/// Draw as much of a circle as lies inside the image bounds.
pub fn draw_hollow_circle<I>(image: &I,
                             center: (i32, i32),
                             radius: i32,
                             color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_hollow_circle_mut(&mut out, center, radius, color);
    out
}

/// Draw as much of a circle as lies inside the image bounds.
pub fn draw_hollow_circle_mut<I>(image: &mut I, center: (i32, i32), radius: i32, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut x = radius;
    let mut y = 0i32;
    let mut err = 0i32;
    let x0 = center.0;
    let y0 = center.1;

    while x >= y {
       draw_if_in_bounds(image, x0 + x, y0 + y, color);
       draw_if_in_bounds(image, x0 + y, y0 + x, color);
       draw_if_in_bounds(image, x0 - y, y0 + x, color);
       draw_if_in_bounds(image, x0 - x, y0 + y, color);
       draw_if_in_bounds(image, x0 - x, y0 - y, color);
       draw_if_in_bounds(image, x0 - y, y0 - x, color);
       draw_if_in_bounds(image, x0 + y, y0 - x, color);
       draw_if_in_bounds(image, x0 + x, y0 - y, color);

       y += 1;
       err += 1 + 2 * y;
       if 2 * (err - x) + 1 > 0 {
           x -= 1;
           err += 1 - 2 * x;
       }
    }
}

/// Draw as much of a circle, including its contents, as lies inside the image bounds.
pub fn draw_filled_circle_mut<I>(image: &mut I, center: (i32, i32), radius: i32, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut x = radius;
    let mut y = 0i32;
    let mut err = 0i32;
    let x0 = center.0;
    let y0 = center.1;

    while x >= y {
        draw_line_segment_mut(image, ((x0 - x) as f32, (y0 + y) as f32), ((x0 + x) as f32, (y0 + y) as f32), color);
        draw_line_segment_mut(image, ((x0 - y) as f32, (y0 + x) as f32), ((x0 + y) as f32, (y0 + x) as f32), color);
        draw_line_segment_mut(image, ((x0 - x) as f32, (y0 - y) as f32), ((x0 + x) as f32, (y0 - y) as f32), color);
        draw_line_segment_mut(image, ((x0 - y) as f32, (y0 - x) as f32), ((x0 + y) as f32, (y0 - x) as f32), color);

        y += 1;
        err += 1 + 2 * y;
        if 2 * (err - x) + 1 > 0 {
            x -= 1;
            err += 1 - 2 * x;
        }
    }
}

/// Draw as much of a circle and its contents as lies inside the image bounds.
pub fn draw_filled_circle<I>(image: &I,
                             center: (i32, i32),
                             radius: i32,
                             color: I::Pixel) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_filled_circle_mut(&mut out, center, radius, color);
    out
}

// Set pixel at (x, y) to color if this point lies within image bounds,
// otherwise do nothing.
fn draw_if_in_bounds<I>(image: &mut I, x: i32, y: i32, color: I::Pixel)
    where I: GenericImage,
          I::Pixel: 'static
{
    if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32{
        image.put_pixel(x as u32, y as u32, color);
    }
}

/// A 2D point.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point<T: Copy + PartialEq + Eq> {
    x: T,
    y: T,
}

impl<T: Copy + PartialEq + Eq> Point<T> {
    /// Construct a point at (x, y).
    pub fn new (x: T, y: T) -> Point<T> {
        Point::<T> { x: x, y: y}
    }
}

/// Draws as much of a filled convex polygon as lies within image bounds. The provided
/// list of points should be an open path, i.e. the first and last points must not be equal.
/// An implicit edge is added from the last to the first point in the slice.
///
/// Does not validate that input is convex.
pub fn draw_convex_polygon<I>(image: &I, poly: &[Point<i32>], color: I::Pixel) -> Image<I::Pixel>
    where I : GenericImage, I::Pixel: 'static
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
pub fn draw_convex_polygon_mut<I>(image: &mut I, poly: &[Point<i32>], color: I::Pixel)
    where I : GenericImage, I::Pixel: 'static
{
    if poly.len() == 0 {
        return;
    }
    if poly[0] == poly[poly.len() - 1] {
        panic!("First point {:?} == last point {:?}", poly[0], poly[poly.len() - 1]);
    }

    let mut y_min = i32::MAX;
    let mut y_max = i32::MIN;
    for p in poly {
        y_min = min(y_min, p.y);
        y_max = max(y_max, p.y);
    }

    let (width, height) = image.dimensions();

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
                }
                else {
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
                draw_if_in_bounds(image, intersections[i], y, color);
                break;
            }
            let from = max(0, min(intersections[i], width as i32 - 1));
            let to = max(0, min(intersections[i + 1], width as i32 - 1));
            for x in from..to + 1 {
                image.put_pixel(x as u32, y as u32, color);
            }
            i += 2;
        }

        intersections.clear();
    }

    for edge in &edges {
        let start = (edge[0].x as f32, edge[0].y as f32);
        let end = (edge[1].x as f32, edge[1].y as f32);
        draw_line_segment_mut(image, start, end, color);
    }
}

/// Draws as much of a cubic bezier curve as lies within image bounds.
pub fn draw_cubic_bezier_curve<I>(image: &I, start: (f32, f32), end: (f32, f32), control_a: (f32, f32), control_b: (f32, f32), color: I::Pixel) -> Image<I::Pixel>
    where I : GenericImage, I::Pixel: 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_cubic_bezier_curve_mut(&mut out, start, end, control_a, control_b, color);
    out
}

/// Draws as much of a cubic bezier curve as lies within image bounds.
pub fn draw_cubic_bezier_curve_mut<I>(image: &mut I, start: (f32, f32), end: (f32, f32), control_a: (f32, f32), control_b: (f32, f32), color: I::Pixel)
    where I : GenericImage, I::Pixel: 'static
{
    // Bezier Curve function from: https://pomax.github.io/bezierinfo/#control
    let cubic_bezier_curve = |t: f32| {
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let x = (start.0 * mt3) + (3.0 * control_a.0 * mt2 * t) + (3.0 * control_b.0 * mt * t2) + (end.0 * t3);
        let y = (start.1 * mt3) + (3.0 * control_a.1 * mt2 * t) + (3.0 * control_b.1 * mt * t2) + (end.1 * t3);
        (x.round(), y.round()) // round to nearest pixel, to avoid ugly line artifacts
    };

    let distance = |point_a: (f32, f32), point_b: (f32, f32)| {
        ((point_a.0 - point_b.0).powi(2) + (point_a.1 - point_b.1).powi(2)).sqrt()
    };

    // Approximate curve's length by adding distance between control points.
    let curve_length_bound: f32 = distance(start, control_a) + distance(control_a, control_b) + distance(control_b, end);

    // Use hyperbola function to give shorter curves a bias in number of line segments.
    let num_segments: i32 = ((curve_length_bound.powi(2) + 800.0).sqrt() / 8.0) as i32;

    // Sample points along the curve and connect them with line segments.
    let t_interval = 1f32 / (num_segments as f32);
    let mut t1 = 0f32;
    for i in 0..num_segments {
        let t2 = (i as f32 + 1.0) * t_interval;
        draw_line_segment_mut(image, cubic_bezier_curve(t1), cubic_bezier_curve(t2), color);
        t1 = t2;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rect::Rect;
    use image::{GrayImage, ImageBuffer, Luma, RgbImage, Rgb};
    use test::{Bencher, black_box};

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_inside_bounds() {
      let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

      let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
          1, 1, 1, 1, 1,
          1, 1, 2, 1, 1,
          1, 2, 2, 2, 1,
          1, 1, 2, 1, 1,
          1, 1, 1, 1, 1]).unwrap();

      assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 2, 2), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_partially_outside_left() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            2, 1, 1, 1, 1,
            2, 2, 1, 1, 1,
            2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 0, 2), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_partially_outside_right() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 2,
            1, 1, 1, 2, 2,
            1, 1, 1, 1, 2,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([2u8]), 4, 2), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_partially_outside_bottom() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 3, 1, 1,
            1, 3, 3, 3, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([3u8]), 2, 4), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_partially_outside_top() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 9, 9, 9, 1,
            1, 1, 9, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 2, 0), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_outside_bottom() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            9, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 0, 5), expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_corner_outside_right() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 9,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        assert_pixels_eq!(draw_cross(&image, Luma([9u8]), 5, 0), expected);
    }


// Octants for line directions:
//
//   \ 5 | 6 /
//   4 \ | / 7
//   ---   ---
//   3 / | \ 0
//   / 2 | 1 \

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_horizontal() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            4, 4, 4, 4, 4,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let right = draw_line_segment(&image, (-3f32, 1f32), (6f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(right, expected);

        let left = draw_line_segment(&image, (6f32, 1f32), (-3f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(left, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_oct0_and_oct4() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 9, 9, 1, 1,
            1, 1, 1, 9, 9,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let oct0 = draw_line_segment(&image, (1f32, 1f32), (4f32, 2f32), Luma([9u8]));
        assert_pixels_eq!(oct0, expected);

        let oct4 = draw_line_segment(&image, (4f32, 2f32), (1f32, 1f32), Luma([9u8]));
        assert_pixels_eq!(oct4, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_diagonal() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 6, 1, 1, 1,
            1, 1, 6, 1, 1,
            1, 1, 1, 6, 1,
            1, 1, 1, 1, 1]).unwrap();

        let down_right = draw_line_segment(&image, (1f32, 1f32), (3f32, 3f32), Luma([6u8]));
        assert_pixels_eq!(down_right, expected);

        let up_left = draw_line_segment(&image, (3f32, 3f32), (1f32, 1f32), Luma([6u8]));
        assert_pixels_eq!(up_left, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_oct1_and_oct5() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            5, 1, 1, 1, 1,
            5, 1, 1, 1, 1,
            5, 1, 1, 1, 1,
            1, 5, 1, 1, 1,
            1, 5, 1, 1, 1]).unwrap();

        let oct1 = draw_line_segment(&image, (0f32, 0f32), (1f32, 4f32), Luma([5u8]));
        assert_pixels_eq!(oct1, expected);

        let oct5 = draw_line_segment(&image, (1f32, 4f32), (0f32, 0f32), Luma([5u8]));
        assert_pixels_eq!(oct5, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_vertical() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 8, 1,
            1, 1, 1, 1, 1]).unwrap();

        let down = draw_line_segment(&image, (3f32, 1f32), (3f32, 3f32), Luma([8u8]));
        assert_pixels_eq!(down, expected);

        let up = draw_line_segment(&image, (3f32, 3f32), (3f32, 1f32), Luma([8u8]));
        assert_pixels_eq!(up, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_oct2_and_oct6() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 4, 1, 1,
            1, 1, 4, 1, 1,
            1, 4, 1, 1, 1,
            1, 4, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let oct2 = draw_line_segment(&image, (2f32, 0f32), (1f32, 3f32), Luma([4u8]));
        assert_pixels_eq!(oct2, expected);

        let oct6 = draw_line_segment(&image, (1f32, 3f32), (2f32, 0f32), Luma([4u8]));
        assert_pixels_eq!(oct6, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_line_segment_oct3_and_oct7() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 2, 2,
            2, 2, 2, 1, 1]).unwrap();

        let oct3 = draw_line_segment(&image, (0f32, 4f32), (5f32, 3f32), Luma([2u8]));
        assert_pixels_eq!(oct3, expected);

        let oct7 = draw_line_segment(&image, (5f32, 3f32), (0f32, 4f32), Luma([2u8]));
        assert_pixels_eq!(oct7, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_antialiased_line_segment_horizontal_and_vertical() {
        use image::imageops::rotate270;
        use pixelops::interpolate;

        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 2, 2, 2, 2,
            1, 1, 1, 1, 1]).unwrap();

        let color = Luma([2u8]);
        // Deliberately ends one pixel out of bounds
        let right = draw_antialiased_line_segment(&image, (1, 3), (5, 3), color, interpolate);
        assert_pixels_eq!(right, expected);

        // Deliberately starts one pixel out of bounds
        let left = draw_antialiased_line_segment(&image, (5, 3), (1, 3), color, interpolate);
        assert_pixels_eq!(left, expected);

        // Deliberately starts one pixel out of bounds
        let down = draw_antialiased_line_segment(&image, (3, -1), (3, 3), color, interpolate);
        assert_pixels_eq!(down, rotate270(&expected));

        // Deliberately end one pixel out of bounds
        let up = draw_antialiased_line_segment(&image, (3, 3), (3, -1), color, interpolate);
        assert_pixels_eq!(up, rotate270(&expected));
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_antialiased_line_segment_diagonal() {
        use image::imageops::rotate90;
        use pixelops::interpolate;

        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 2, 1, 1,
            1, 2, 1, 1, 1,
            2, 1, 1, 1, 1,
            1, 1, 1, 1, 1]).unwrap();

        let color = Luma([2u8]);
        let up_right = draw_antialiased_line_segment(&image, (0, 3), (2, 1), color, interpolate);
        assert_pixels_eq!(up_right, expected);

        let down_left = draw_antialiased_line_segment(&image, (2, 1), (0, 3), color, interpolate);
        assert_pixels_eq!(down_left, expected);

        let up_left = draw_antialiased_line_segment(&image, (1, 0), (3, 2), color, interpolate);
        assert_pixels_eq!(up_left, rotate90(&expected));

        let down_right = draw_antialiased_line_segment(&image, (3, 2), (1, 0), color, interpolate);
        assert_pixels_eq!(down_right, rotate90(&expected));
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_antialiased_line_segment_oct7_and_oct3() {
        use pixelops::interpolate;

        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        // Gradient is 3/4
        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1,  1,  1,  13, 50,
            1,  1,  25, 37,  1,
            1,  37, 25,  1,  1,
            50, 13, 1,   1,  1,
            1,  1,  1,   1,  1]).unwrap();

        let color = Luma([50u8]);
        let oct7 = draw_antialiased_line_segment(&image, (0, 3), (4, 0), color, interpolate);
        assert_pixels_eq!(oct7, expected);

        let oct3 = draw_antialiased_line_segment(&image, (4, 0), (0, 3), color, interpolate);
        assert_pixels_eq!(oct3, expected);
    }

    macro_rules! bench_antialiased_lines {
        ($name:ident, $start:expr, $end:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use pixelops::interpolate;
                use super::draw_antialiased_line_segment_mut;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_antialiased_line_segment_mut(&mut image, $start, $end, color, interpolate);
                    test::black_box(&image);
                    });
            }
        }
    }

    bench_antialiased_lines!(bench_draw_antialiased_line_segment_horizontal, (10, 10), (450, 10));
    bench_antialiased_lines!(bench_draw_antialiased_line_segment_vertical, (10, 10), (10, 450));
    bench_antialiased_lines!(bench_draw_antialiased_line_segment_diagonal, (10, 10), (450, 450));
    bench_antialiased_lines!(bench_draw_antialiased_line_segment_shallow, (10, 10), (450, 80));

    macro_rules! bench_cubic_bezier_curve {
        ($name:ident, $start:expr, $end:expr, $control_a:expr, $control_b:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use super::draw_cubic_bezier_curve_mut;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_cubic_bezier_curve_mut(&mut image, $start, $end, $control_a, $control_b, color);
                    test::black_box(&image);
                    });
            }
        }
    }

    bench_cubic_bezier_curve!(bench_draw_cubic_bezier_curve_short, (100.0, 100.0), (130.0, 130.0), (110.0, 100.0), (120.0, 130.0));
    bench_cubic_bezier_curve!(bench_draw_cubic_bezier_curve_long, (100.0, 100.0), (400.0, 400.0), (500.0, 0.0), (0.0, 500.0));

    macro_rules! bench_hollow_ellipse {
        ($name:ident, $center:expr, $width_radius:expr, $height_radius:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use super::draw_hollow_ellipse_mut;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_hollow_ellipse_mut(&mut image, $center, $width_radius, $height_radius, color);
                    test::black_box(&image);
                    });
            }
        }
    }

    bench_hollow_ellipse!(bench_bench_hollow_ellipse_circle, (200, 200), 80, 80);
    bench_hollow_ellipse!(bench_bench_hollow_ellipse_vertical, (200, 200), 40, 100);
    bench_hollow_ellipse!(bench_bench_hollow_ellipse_horizontal, (200, 200), 100, 40);

    macro_rules! bench_filled_ellipse {
        ($name:ident, $center:expr, $width_radius:expr, $height_radius:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use super::draw_filled_ellipse_mut;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_filled_ellipse_mut(&mut image, $center, $width_radius, $height_radius, color);
                    test::black_box(&image);
                    });
            }
        }
    }

    bench_filled_ellipse!(bench_bench_filled_ellipse_circle, (200, 200), 80, 80);
    bench_filled_ellipse!(bench_bench_filled_ellipse_vertical, (200, 200), 40, 100);
    bench_filled_ellipse!(bench_bench_filled_ellipse_horizontal, (200, 200), 100, 40);

    #[bench]
    fn bench_draw_filled_rect_mut_rgb(b: &mut Bencher) {
        let mut image = RgbImage::new(200, 200);
        let color = Rgb([120u8, 60u8, 47u8]);
        let rect = Rect::at(50, 50).of_size(80, 90);
        b.iter(|| {
            draw_filled_rect_mut(&mut image, rect, color);
            black_box(&image);
        });
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_hollow_rect() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 4, 4, 4,
            1, 1, 4, 1, 4,
            1, 1, 4, 4, 4]).unwrap();

        let actual = draw_hollow_rect(&image, Rect::at(2, 2).of_size(3, 3), Luma([4u8]));
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn test_draw_filled_rect() {
        let image: GrayImage = ImageBuffer::from_pixel(5, 5, Luma([1u8]));

        let expected: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1, 1, 1, 1, 1,
            1, 4, 4, 4, 1,
            1, 4, 4, 4, 1,
            1, 4, 4, 4, 1,
            1, 1, 1, 1, 1]).unwrap();

        let actual = draw_filled_rect(&image, Rect::at(1, 1).of_size(3, 3), Luma([4u8]));
        assert_pixels_eq!(actual, expected);
    }
}
