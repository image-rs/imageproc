use crate::definitions::Image;
use crate::drawing::draw_if_in_bounds;
use crate::drawing::line::draw_line_segment_mut;
use crate::drawing::Canvas;
use image::{GenericImage, ImageBuffer};

/// Draws the outline of an ellipse on a new copy of an image.
///
/// Draws as much of an ellipse as lies inside the image bounds.
///
/// Uses the [Midpoint Ellipse Drawing Algorithm](https://web.archive.org/web/20160128020853/http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/).
/// (Modified from Bresenham's algorithm)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// (`x^2 / width_radius^2) + (y^2 / height_radius^2) = 1`
#[must_use = "the function does not modify the original image"]
pub fn draw_hollow_ellipse<I>(
    image: &I,
    center: (i32, i32),
    width_radius: i32,
    height_radius: i32,
    color: I::Pixel,
) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_hollow_ellipse_mut(&mut out, center, width_radius, height_radius, color);
    out
}

/// Draws the outline of an ellipse on an image in place.
///
/// Draws as much of an ellipse as lies inside the image bounds.
///
/// Uses the [Midpoint Ellipse Drawing Algorithm](https://web.archive.org/web/20160128020853/http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/).
/// (Modified from Bresenham's algorithm)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) = 1`
pub fn draw_hollow_ellipse_mut<C>(
    canvas: &mut C,
    center: (i32, i32),
    width_radius: i32,
    height_radius: i32,
    color: C::Pixel,
) where
    C: Canvas,
{
    // Circle drawing algorithm is faster, so use it if the given ellipse is actually a circle.
    if width_radius == height_radius {
        draw_hollow_circle_mut(canvas, center, width_radius, color);
        return;
    }

    let draw_quad_pixels = |x0: i32, y0: i32, x: i32, y: i32| {
        draw_if_in_bounds(canvas, x0 + x, y0 + y, color);
        draw_if_in_bounds(canvas, x0 - x, y0 + y, color);
        draw_if_in_bounds(canvas, x0 + x, y0 - y, color);
        draw_if_in_bounds(canvas, x0 - x, y0 - y, color);
    };

    draw_ellipse(draw_quad_pixels, center, width_radius, height_radius);
}

/// Draws an ellipse and its contents on a new copy of the image.
///
/// Draw as much of the ellipse and its contents as lies inside the image bounds.
///
/// Uses the [Midpoint Ellipse Drawing Algorithm](https://web.archive.org/web/20160128020853/http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/).
/// (Modified from Bresenham's algorithm)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) <= 1`
#[must_use = "the function does not modify the original image"]
pub fn draw_filled_ellipse<I>(
    image: &I,
    center: (i32, i32),
    width_radius: i32,
    height_radius: i32,
    color: I::Pixel,
) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_filled_ellipse_mut(&mut out, center, width_radius, height_radius, color);
    out
}

/// Draws an ellipse and its contents on an image in place.
///
/// Draw as much of the ellipse and its contents as lies inside the image bounds.
///
/// Uses the [Midpoint Ellipse Drawing Algorithm](https://web.archive.org/web/20160128020853/http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/).
/// (Modified from Bresenham's algorithm)
///
/// The ellipse is axis-aligned and satisfies the following equation:
///
/// `(x^2 / width_radius^2) + (y^2 / height_radius^2) <= 1`
pub fn draw_filled_ellipse_mut<C>(
    canvas: &mut C,
    center: (i32, i32),
    width_radius: i32,
    height_radius: i32,
    color: C::Pixel,
) where
    C: Canvas,
{
    // Circle drawing algorithm is faster, so use it if the given ellipse is actually a circle.
    if width_radius == height_radius {
        draw_filled_circle_mut(canvas, center, width_radius, color);
        return;
    }

    let draw_line_pairs = |x0: i32, y0: i32, x: i32, y: i32| {
        draw_line_segment_mut(
            canvas,
            ((x0 - x) as f32, (y0 + y) as f32),
            ((x0 + x) as f32, (y0 + y) as f32),
            color,
        );
        draw_line_segment_mut(
            canvas,
            ((x0 - x) as f32, (y0 - y) as f32),
            ((x0 + x) as f32, (y0 - y) as f32),
            color,
        );
    };

    draw_ellipse(draw_line_pairs, center, width_radius, height_radius);
}

// Implements the Midpoint Ellipse Drawing Algorithm https://web.archive.org/web/20160128020853/http://tutsheap.com/c/mid-point-ellipse-drawing-algorithm/). (Modified from Bresenham's algorithm)
//
// Takes a function that determines how to render the points on the ellipse.
fn draw_ellipse<F>(mut render_func: F, center: (i32, i32), width_radius: i32, height_radius: i32)
where
    F: FnMut(i32, i32, i32, i32),
{
    let (x0, y0) = center;
    let w2 = (width_radius * width_radius) as f32;
    let h2 = (height_radius * height_radius) as f32;
    let mut x = 0;
    let mut y = height_radius;
    let mut px = 0.0;
    let mut py = 2.0 * w2 * y as f32;

    render_func(x0, y0, x, y);

    // Top and bottom regions.
    let mut p = h2 - (w2 * height_radius as f32) + (0.25 * w2);
    while px < py {
        x += 1;
        px += 2.0 * h2;
        if p < 0.0 {
            p += h2 + px;
        } else {
            y -= 1;
            py += -2.0 * w2;
            p += h2 + px - py;
        }

        render_func(x0, y0, x, y);
    }

    // Left and right regions.
    p = h2 * (x as f32 + 0.5).powi(2) + (w2 * (y - 1).pow(2) as f32) - w2 * h2;
    while y > 0 {
        y -= 1;
        py += -2.0 * w2;
        if p > 0.0 {
            p += w2 - py;
        } else {
            x += 1;
            px += 2.0 * h2;
            p += w2 - py + px;
        }

        render_func(x0, y0, x, y);
    }
}

/// Draws the outline of a circle on a new copy of an image.
///
/// Draw as much of the circle as lies inside the image bounds.
#[must_use = "the function does not modify the original image"]
pub fn draw_hollow_circle<I>(
    image: &I,
    center: (i32, i32),
    radius: i32,
    color: I::Pixel,
) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_hollow_circle_mut(&mut out, center, radius, color);
    out
}

/// Draws the outline of a circle on an image in place.
///
/// Draw as much of the circle as lies inside the image bounds.
pub fn draw_hollow_circle_mut<C>(canvas: &mut C, center: (i32, i32), radius: i32, color: C::Pixel)
where
    C: Canvas,
{
    let mut x = 0i32;
    let mut y = radius;
    let mut p = 1 - radius;
    let x0 = center.0;
    let y0 = center.1;

    while x <= y {
        draw_if_in_bounds(canvas, x0 + x, y0 + y, color);
        draw_if_in_bounds(canvas, x0 + y, y0 + x, color);
        draw_if_in_bounds(canvas, x0 - y, y0 + x, color);
        draw_if_in_bounds(canvas, x0 - x, y0 + y, color);
        draw_if_in_bounds(canvas, x0 - x, y0 - y, color);
        draw_if_in_bounds(canvas, x0 - y, y0 - x, color);
        draw_if_in_bounds(canvas, x0 + y, y0 - x, color);
        draw_if_in_bounds(canvas, x0 + x, y0 - y, color);

        x += 1;
        if p < 0 {
            p += 2 * x + 1;
        } else {
            y -= 1;
            p += 2 * (x - y) + 1;
        }
    }
}

/// Draws a circle and its contents on an image in place.
///
/// Draws as much of a circle and its contents as lies inside the image bounds.
pub fn draw_filled_circle_mut<C>(canvas: &mut C, center: (i32, i32), radius: i32, color: C::Pixel)
where
    C: Canvas,
{
    let mut x = 0i32;
    let mut y = radius;
    let mut p = 1 - radius;
    let x0 = center.0;
    let y0 = center.1;

    while x <= y {
        draw_line_segment_mut(
            canvas,
            ((x0 - x) as f32, (y0 + y) as f32),
            ((x0 + x) as f32, (y0 + y) as f32),
            color,
        );
        draw_line_segment_mut(
            canvas,
            ((x0 - y) as f32, (y0 + x) as f32),
            ((x0 + y) as f32, (y0 + x) as f32),
            color,
        );
        draw_line_segment_mut(
            canvas,
            ((x0 - x) as f32, (y0 - y) as f32),
            ((x0 + x) as f32, (y0 - y) as f32),
            color,
        );
        draw_line_segment_mut(
            canvas,
            ((x0 - y) as f32, (y0 - x) as f32),
            ((x0 + y) as f32, (y0 - x) as f32),
            color,
        );

        x += 1;
        if p < 0 {
            p += 2 * x + 1;
        } else {
            y -= 1;
            p += 2 * (x - y) + 1;
        }
    }
}

/// Draws a circle and its contents on a new copy of the image.
///
/// Draws as much of a circle and its contents as lies inside the image bounds.
#[must_use = "the function does not modify the original image"]
pub fn draw_filled_circle<I>(
    image: &I,
    center: (i32, i32),
    radius: i32,
    color: I::Pixel,
) -> Image<I::Pixel>
where
    I: GenericImage,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_filled_circle_mut(&mut out, center, radius, color);
    out
}

#[cfg(test)]
mod tests {
    use super::draw_filled_ellipse_mut;
    use image::GenericImage;

    struct Ellipse {
        center: (i32, i32),
        width_radius: i32,
        height_radius: i32,
    }

    impl Ellipse {
        fn normalized_distance_from_center(&self, (x, y): (i32, i32)) -> f32 {
            let (cx, cy) = self.center;
            let (w, h) = (self.width_radius as f32, self.height_radius as f32);
            ((cx - x) as f32 / w).powi(2) + ((cy - y) as f32 / h).powi(2)
        }
        fn is_boundary_point(&self, (x, y): (i32, i32), boundary_eps: f32) -> bool {
            assert!(boundary_eps >= 0.0);
            (self.normalized_distance_from_center((x, y)) - 1.0).abs() < boundary_eps
        }
        fn is_inner_point(&self, (x, y): (i32, i32)) -> bool {
            self.normalized_distance_from_center((x, y)) < 1.0
        }
    }

    fn check_filled_ellipse<I: GenericImage>(
        img: &I,
        ellipse: Ellipse,
        inner_color: I::Pixel,
        outer_color: I::Pixel,
        boundary_eps: f32,
    ) where
        I::Pixel: core::fmt::Debug + PartialEq,
    {
        for x in 0..img.width() as i32 {
            for y in 0..img.height() as i32 {
                if ellipse.is_boundary_point((x, y), boundary_eps) {
                    continue;
                }
                let pixel = img.get_pixel(x as u32, y as u32);
                if ellipse.is_inner_point((x, y)) {
                    assert_eq!(pixel, inner_color);
                } else {
                    assert_eq!(pixel, outer_color);
                }
            }
        }
    }

    #[cfg_attr(miri, ignore = "slow [>1480s]")]
    #[test]
    fn test_draw_filled_ellipse() {
        let ellipse = Ellipse {
            center: (960, 540),
            width_radius: 960,
            height_radius: 540,
        };
        let inner_color = image::Rgb([255, 0, 0]);
        let outer_color = image::Rgb([0, 0, 0]);
        let mut img = image::RgbImage::new(1920, 1080);
        draw_filled_ellipse_mut(
            &mut img,
            ellipse.center,
            ellipse.width_radius,
            ellipse.height_radius,
            inner_color,
        );
        const EPS: f32 = 0.0019;
        check_filled_ellipse(&img, ellipse, inner_color, outer_color, EPS);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use image::{GrayImage, Luma};

    macro_rules! bench_hollow_ellipse {
        ($name:ident, $center:expr, $width_radius:expr, $height_radius:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use super::draw_hollow_ellipse_mut;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_hollow_ellipse_mut(
                        &mut image,
                        $center,
                        $width_radius,
                        $height_radius,
                        color,
                    );
                    test::black_box(&image);
                });
            }
        };
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
                    draw_filled_ellipse_mut(
                        &mut image,
                        $center,
                        $width_radius,
                        $height_radius,
                        color,
                    );
                    test::black_box(&image);
                });
            }
        };
    }

    bench_filled_ellipse!(bench_bench_filled_ellipse_circle, (200, 200), 80, 80);
    bench_filled_ellipse!(bench_bench_filled_ellipse_vertical, (200, 200), 40, 100);
    bench_filled_ellipse!(bench_bench_filled_ellipse_horizontal, (200, 200), 100, 40);
}
