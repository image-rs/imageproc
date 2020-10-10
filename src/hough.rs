//! Line detection via the [Hough transform].
//!
//! [Hough transform]: https://en.wikipedia.org/wiki/Hough_transform

use crate::definitions::Image;
use crate::drawing::draw_line_segment_mut;
use crate::suppress::suppress_non_maximum;
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel};
use std::f32;

/// A detected line, in polar coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PolarLine {
    /// Signed distance of the line from the origin (top-left of the image), in pixels.
    pub r: f32,
    /// Clockwise angle in degrees between the x-axis and the line.
    /// Always between 0 and 180.
    pub angle_in_degrees: u32,
}

/// Options for Hough line detection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineDetectionOptions {
    /// Number of votes required to be detected as a line.
    pub vote_threshold: u32,
    /// Non-maxima suppression is applied to accumulator buckets before
    /// returning lines. Only lines which have the greatest vote in the
    /// block centred on them of side length `2 * suppression_radius + 1`
    /// are returned. Set to `0` if you don't want to apply non-maxima suppression.
    pub suppression_radius: u32,
}

/// Detects lines in a binary input image using the Hough transform.
///
/// Points are considered to be in the foreground (and thus vote for lines)
/// if their intensity is non-zero.
///
/// See ./examples/hough.rs for example usage.
pub fn detect_lines(image: &GrayImage, options: LineDetectionOptions) -> Vec<PolarLine> {
    let (width, height) = image.dimensions();

    // The maximum possible radius is the diagonal of the image.
    let rmax = ((width * width + height * height) as f64).sqrt() as i32;

    // Measure angles in degrees, and use bins of width 1 pixel and height 1 degree.
    // We use the convention that distances are positive for angles in (0, 180] and
    // negative for angles in [180, 360).
    let mut acc: ImageBuffer<Luma<u32>, Vec<u32>> = ImageBuffer::new(2 * rmax as u32 + 1, 180u32);

    // Precalculate values of (cos(m), sin(m))
    let lut: Vec<(f32, f32)> = (0..180u32)
        .map(|deg| (deg as f32).to_radians())
        .map(f32::sin_cos)
        .collect();

    for y in 0..height {
        for x in 0..width {
            let p = unsafe { image.unsafe_get_pixel(x, y)[0] };

            if p > 0 {
                for (m, (s, c)) in lut.iter().enumerate() {
                    let r = (x as f32) * c + (y as f32) * s;
                    let d = r as i32 + rmax;

                    if d <= 2 * rmax && d >= 0 {
                        unsafe {
                            let vote_incr = acc.unsafe_get_pixel(d as u32, m as u32)[0] + 1;
                            acc.unsafe_put_pixel(d as u32, m as u32, Luma([vote_incr]));
                        }
                    }
                }
            }
        }
    }

    let acc_sup = suppress_non_maximum(&acc, options.suppression_radius);

    let mut lines = Vec::new();

    for m in 0..acc_sup.height() {
        for r in 0..acc_sup.width() {
            let votes = unsafe { acc_sup.unsafe_get_pixel(r, m)[0] };
            if votes >= options.vote_threshold {
                let line = PolarLine {
                    r: (r as i32 - rmax) as f32,
                    angle_in_degrees: m,
                };
                lines.push(line);
            }
        }
    }

    lines
}

/// Draws each element of `lines` on `image` in the provided `color`.
///
/// See ./examples/hough.rs for example usage.
pub fn draw_polar_lines<P>(image: &Image<P>, lines: &[PolarLine], color: P) -> Image<P>
where
    P: Pixel + 'static,
{
    let mut out = image.clone();
    draw_polar_lines_mut(&mut out, lines, color);
    out
}

/// Draws each element of `lines` on `image` in the provided `color`.
///
/// See ./examples/hough.rs for example usage.
pub fn draw_polar_lines_mut<P>(image: &mut Image<P>, lines: &[PolarLine], color: P)
where
    P: Pixel + 'static,
{
    for line in lines {
        draw_polar_line(image, *line, color);
    }
}

fn draw_polar_line<P>(image: &mut Image<P>, line: PolarLine, color: P)
where
    P: Pixel + 'static,
{
    if let Some((s, e)) = intersection_points(line, image.width(), image.height()) {
        draw_line_segment_mut(image, s, e, color);
    }
}

/// Returns the intersection points of a `PolarLine` with an image of given width and height,
/// or `None` if the line and image bounding box are disjoint. The x value of an intersection
/// point lies within the closed interval [0, image_width] and the y value within the closed
/// interval [0, image_height].
fn intersection_points(
    line: PolarLine,
    image_width: u32,
    image_height: u32,
) -> Option<((f32, f32), (f32, f32))> {
    let r = line.r;
    let m = line.angle_in_degrees;
    let w = image_width as f32;
    let h = image_height as f32;

    // Vertical line
    if m == 0 {
        return if r >= 0.0 && r <= w {
            Some(((r, 0.0), (r, h)))
        } else {
            None
        };
    }

    // Horizontal line
    if m == 90 {
        return if r >= 0.0 && r <= h {
            Some(((0.0, r), (w, r)))
        } else {
            None
        };
    }

    let theta = (m as f32).to_radians();
    let (sin, cos) = theta.sin_cos();

    let right_y = cos.mul_add(-w, r) / sin;
    let left_y = r / sin;
    let bottom_x = sin.mul_add(-h, r) / cos;
    let top_x = r / cos;

    let mut start = None;

    if right_y >= 0.0 && right_y <= h {
        let right_intersect = (w, right_y);
        if let Some(s) = start {
            return Some((s, right_intersect));
        }
        start = Some(right_intersect);
    }

    if left_y >= 0.0 && left_y <= h {
        let left_intersect = (0.0, left_y);
        if let Some(s) = start {
            return Some((s, left_intersect));
        }
        start = Some(left_intersect);
    }

    if bottom_x >= 0.0 && bottom_x <= w {
        let bottom_intersect = (bottom_x, h);
        if let Some(s) = start {
            return Some((s, bottom_intersect));
        }
        start = Some(bottom_intersect);
    }

    if top_x >= 0.0 && top_x <= w {
        let top_intersect = (top_x, 0.0);
        if let Some(s) = start {
            return Some((s, top_intersect));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, ImageBuffer, Luma};
    use test::{black_box, Bencher};

    fn assert_points_eq(
        actual: Option<((f32, f32), (f32, f32))>,
        expected: Option<((f32, f32), (f32, f32))>,
    ) {
        match (actual, expected) {
            (None, None) => {}
            (Some(ps), Some(qs)) => {
                let points_eq = |p: (f32, f32), q: (f32, f32)| {
                    (p.0 - q.0).abs() < 1.0e-6 && (p.1 - q.1).abs() < 1.0e-6
                };

                match (points_eq(ps.0, qs.0), points_eq(ps.1, qs.1)) {
                    (true, true) => {}
                    _ => {
                        panic!("Expected {:?}, got {:?}", expected, actual);
                    }
                };
            }
            (Some(_), None) => {
                panic!("Expected None, got {:?}", actual);
            }
            (None, Some(_)) => {
                panic!("Expected {:?}, got None", expected);
            }
        }
    }

    #[test]
    fn intersection_points_zero_signed_distance() {
        // Vertical
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 0.0,
                    angle_in_degrees: 0,
                },
                10,
                5,
            ),
            Some(((0.0, 0.0), (0.0, 5.0))),
        );
        // Horizontal
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 0.0,
                    angle_in_degrees: 90,
                },
                10,
                5,
            ),
            Some(((0.0, 0.0), (10.0, 0.0))),
        );
        // Bottom left to top right
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 0.0,
                    angle_in_degrees: 45,
                },
                10,
                5,
            ),
            Some(((0.0, 0.0), (0.0, 0.0))),
        );
        // Top left to bottom right
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 0.0,
                    angle_in_degrees: 135,
                },
                10,
                5,
            ),
            Some(((0.0, 0.0), (5.0, 5.0))),
        );
        // Top left to bottom right, square image (because a previous version of the code
        // got this case wrong)
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 0.0,
                    angle_in_degrees: 135,
                },
                10,
                10,
            ),
            Some(((10.0, 10.0), (0.0, 0.0))),
        );
    }

    #[test]
    fn intersection_points_positive_signed_distance() {
        // Vertical intersecting image
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 9.0,
                    angle_in_degrees: 0,
                },
                10,
                5,
            ),
            Some(((9.0, 0.0), (9.0, 5.0))),
        );
        // Vertical outside image
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 8.0,
                    angle_in_degrees: 0,
                },
                5,
                10,
            ),
            None,
        );
        // Horizontal intersecting image
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 9.0,
                    angle_in_degrees: 90,
                },
                5,
                10,
            ),
            Some(((0.0, 9.0), (5.0, 9.0))),
        );
        // Horizontal outside image
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 8.0,
                    angle_in_degrees: 90,
                },
                10,
                5,
            ),
            None,
        );
        // Positive gradient
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: 5.0,
                    angle_in_degrees: 45,
                },
                10,
                5,
            ),
            Some(((50f32.sqrt() - 5.0, 5.0), (50f32.sqrt(), 0.0))),
        );
    }

    #[test]
    fn intersection_points_negative_signed_distance() {
        // Vertical
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: -1.0,
                    angle_in_degrees: 0,
                },
                10,
                5,
            ),
            None,
        );
        // Horizontal
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: -1.0,
                    angle_in_degrees: 90,
                },
                5,
                10,
            ),
            None,
        );
        // Negative gradient
        assert_points_eq(
            intersection_points(
                PolarLine {
                    r: -5.0,
                    angle_in_degrees: 135,
                },
                10,
                5,
            ),
            Some(((10.0, 10.0 - 50f32.sqrt()), (50f32.sqrt(), 0.0))),
        );
    }

    //  --------------------
    // |                    |
    // |                    |
    // |    *****  *****    |
    // |                    |
    // |                    |
    //  --------------------
    fn separated_horizontal_line_segment() -> GrayImage {
        let white = Luma([255u8]);
        let mut image = GrayImage::new(20, 5);
        for i in 5..10 {
            image.put_pixel(i, 2, white);
        }
        for i in 12..17 {
            image.put_pixel(i, 2, white);
        }
        image
    }

    #[test]
    fn detect_lines_horizontal_below_threshold() {
        let image = separated_horizontal_line_segment();
        let options = LineDetectionOptions {
            vote_threshold: 11,
            suppression_radius: 0,
        };
        let detected = detect_lines(&image, options);
        assert_eq!(detected.len(), 0);
    }

    #[test]
    fn detect_lines_horizontal_above_threshold() {
        let image = separated_horizontal_line_segment();
        let options = LineDetectionOptions {
            vote_threshold: 10,
            suppression_radius: 8,
        };
        let detected = detect_lines(&image, options);
        assert_eq!(detected.len(), 1);
        let line = detected[0];
        assert_eq!(line.r, 1f32);
        assert_eq!(line.angle_in_degrees, 90);
    }

    fn image_with_polar_line(
        width: u32,
        height: u32,
        r: f32,
        angle_in_degrees: u32,
        color: Luma<u8>,
    ) -> GrayImage {
        let mut image = GrayImage::new(width, height);
        draw_polar_line(
            &mut image,
            PolarLine {
                r,
                angle_in_degrees,
            },
            color,
        );
        image
    }

    #[test]
    fn draw_polar_line_horizontal() {
        let actual = image_with_polar_line(5, 5, 2.0, 90, Luma([1]));
        let expected = gray_image!(
            0, 0, 0, 0, 0;
            0, 0, 0, 0, 0;
            1, 1, 1, 1, 1;
            0, 0, 0, 0, 0;
            0, 0, 0, 0, 0);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn draw_polar_line_vertical() {
        let actual = image_with_polar_line(5, 5, 2.0, 0, Luma([1]));
        let expected = gray_image!(
            0, 0, 1, 0, 0;
            0, 0, 1, 0, 0;
            0, 0, 1, 0, 0;
            0, 0, 1, 0, 0;
            0, 0, 1, 0, 0);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn draw_polar_line_bottom_left_to_top_right() {
        let actual = image_with_polar_line(5, 5, 3.0, 45, Luma([1]));
        let expected = gray_image!(
            0, 0, 0, 0, 1;
            0, 0, 0, 1, 0;
            0, 0, 1, 0, 0;
            0, 1, 0, 0, 0;
            1, 0, 0, 0, 0);
        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn draw_polar_line_top_left_to_bottom_right() {
        let actual = image_with_polar_line(5, 5, 0.0, 135, Luma([1]));
        let expected = gray_image!(
            1, 0, 0, 0, 0;
            0, 1, 0, 0, 0;
            0, 0, 1, 0, 0;
            0, 0, 0, 1, 0;
            0, 0, 0, 0, 1);
        assert_pixels_eq!(actual, expected);
    }

    macro_rules! test_detect_line {
        ($name:ident, $r:expr, $angle:expr) => {
            #[test]
            fn $name() {
                let options = LineDetectionOptions {
                    vote_threshold: 10,
                    suppression_radius: 8,
                };
                let image = image_with_polar_line(100, 100, $r, $angle, Luma([255]));
                let detected = detect_lines(&image, options);
                assert_eq!(detected.len(), 1);

                let line = detected[0];
                assert_approx_eq!(line.r, $r, 1.1);
                assert_approx_eq!(line.angle_in_degrees as f32, $angle as f32, 5.0);
            }
        };
    }

    test_detect_line!(detect_line_50_45, 50.0, 45);
    test_detect_line!(detect_line_eps_135, 0.001, 135);
    // https://github.com/image-rs/imageproc/issues/280
    test_detect_line!(detect_line_neg10_120, -10.0, 120);

    macro_rules! bench_detect_lines {
        ($name:ident, $r:expr, $angle:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let options = LineDetectionOptions {
                    vote_threshold: 10,
                    suppression_radius: 8,
                };
                let mut image = GrayImage::new(100, 100);
                draw_polar_line(
                    &mut image,
                    PolarLine {
                        r: $r,
                        angle_in_degrees: $angle,
                    },
                    Luma([255u8]),
                );

                b.iter(|| {
                    let lines = detect_lines(&image, options);
                    black_box(lines);
                });
            }
        };
    }

    bench_detect_lines!(bench_detect_line_50_45, 50.0, 45);
    bench_detect_lines!(bench_detect_line_eps_135, 0.001, 135);
    bench_detect_lines!(bench_detect_line_neg10_120, -10.0, 120);

    fn chessboard(width: u32, height: u32) -> GrayImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                Luma([255u8])
            } else {
                Luma([0u8])
            }
        })
    }

    #[bench]
    fn bench_detect_lines(b: &mut Bencher) {
        let image = chessboard(100, 100);

        let options = LineDetectionOptions {
            vote_threshold: 10,
            suppression_radius: 3,
        };

        b.iter(|| {
            let lines = detect_lines(&image, options);
            black_box(lines);
        });
    }
}
