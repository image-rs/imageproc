//! Line detection via the [Hough transform].
//!
//! [Hough transform]: https://en.wikipedia.org/wiki/Hough_transform

use image::{GenericImage, GrayImage, Luma, ImageBuffer, Pixel};
use drawing::draw_line_segment_mut;
use definitions::Image;
use suppress::suppress_non_maximum;
use std::f32;

/// A detected line, in polar coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PolarLine {
    /// Distance of the line from the origin (top-left of the image), in pixels.
    pub r: f32,
    /// Clockwise angle in degrees between the x-axis and the line.
    pub theta: u32
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
    pub suppression_radius: u32
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
    let mut acc: ImageBuffer<Luma<u32>, Vec<u32>> = ImageBuffer::new(rmax as u32, 180u32);

    let cos_lut: Vec<f32> = (0..180u32).map(|m| degrees_to_radians(m).cos()).collect();
    let sin_lut: Vec<f32> = (0..180u32).map(|m| degrees_to_radians(m).sin()).collect();

    for y in 0..height {
        for x in 0..width {
            let p = unsafe { image.unsafe_get_pixel(x, y)[0] };

            if p > 0 {
                for m in 0..180u32 {
                    let fy = y as f32;
                    let fx = x as f32;

                    let r = unsafe {
                        (fx * *cos_lut.get_unchecked(m as usize) + fy * *sin_lut.get_unchecked(m as usize)) as i32
                    };

                    if r < rmax && r >= 0 {
                        unsafe {
                            let vote_incr = acc.unsafe_get_pixel(r as u32, m)[0] + 1;
                            acc.unsafe_put_pixel(r as u32, m, Luma([vote_incr]));
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
                    r: r as f32,
                    theta: m
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
    where P: Pixel + 'static
{
    let mut out = image.clone();
    draw_polar_lines_mut(&mut out, lines, color);
    out
}

/// Draws each element of `lines` on `image` in the provided `color`.
///
/// See ./examples/hough.rs for example usage.
pub fn draw_polar_lines_mut<P>(image: &mut Image<P>, lines: &[PolarLine], color: P)
    where P: Pixel + 'static
{
    for line in lines {
        draw_polar_line(image, *line, color);
    }
}

fn draw_polar_line<P>(image: &mut Image<P>, line: PolarLine, color: P)
    where P: Pixel + 'static
{
    // TODO: make this less of a mess and add tests
    let r = line.r;
    let m = line.theta;
    let w = image.width() as f32;
    let h = image.height() as f32;

    // Vertical line
    if m == 0 {
        draw_line_segment_mut(image, (r, 0.0), (r, h), color);
        return;
    }

    // Horizontal line
    if m == 90 {
        draw_line_segment_mut(image, (0.0, r), (w, r), color);
    }

    let theta = degrees_to_radians(m);
    let sin = theta.sin();
    let cos = theta.cos();

    let right_y = (r - w * cos) / sin;
    let left_y = r / sin;
    let bottom_x = (r - h * sin) / cos;
    let top_x = r / cos;

    let mut start = None;
    let mut end = None;

    if right_y >= 0.0 && right_y < h {
        let right_intersect = (w, right_y);
        if start == None {
            start = Some(right_intersect);
        } else if end == None {
            end = Some(right_intersect);
        }
    }

    if left_y >= 0.0 && left_y < h {
        let left_intersect = (0.0, left_y);
        if start == None {
            start = Some(left_intersect);
        } else if end == None {
            end = Some(left_intersect);
        }
    }

    if bottom_x >= 0.0 && bottom_x < w {
        let bottom_intersect = (bottom_x, h);
        if start == None {
            start = Some(bottom_intersect);
        } else if end == None {
            end = Some(bottom_intersect);
        }
    }

    if top_x >= 0.0 && top_x < w {
        let top_intersect = (top_x, 0.0);
        if start == None {
            start = Some(top_intersect);
        } else if end == None {
            end = Some(top_intersect);
        }
    }

    match (start, end) {
        (Some(s), Some(e)) => draw_line_segment_mut(image, s, e, color),
        _ => {}
    }
}

fn degrees_to_radians(degrees: u32) -> f32 {
    degrees as f32 * f32::consts::PI / 180.0
}

#[cfg(test)]
mod test {
    use super::*;
    use image::{GrayImage, ImageBuffer, Luma};
    use test::{Bencher, black_box};

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
            suppression_radius: 0
        };
        let detected = detect_lines(&image, options);
        assert_eq!(detected.len(), 0);
    }

    #[test]
    fn detect_lines_horizontal_above_threshold() {
        let image = separated_horizontal_line_segment();
        let options = LineDetectionOptions {
            vote_threshold: 10,
            suppression_radius: 8
        };
        let detected = detect_lines(&image, options);
        assert_eq!(detected.len(), 1);
        let line = detected[0];
        assert_eq!(line.r, 1f32);
        assert_eq!(line.theta, 90);
    }

    // TODO: This is an exact duplicate of a function in tbe regionlabelling tests.
    // TODO: Add some unit tests and benchmarks of more interesting cases.
    fn chessboard(width: u32, height: u32) -> GrayImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 { return Luma([255u8]); }
            else { return Luma([0u8]); }
        })
    }

    #[bench]
    fn bench_detect_lines(b: &mut Bencher) {
        let image = chessboard(200, 200);

        let options = LineDetectionOptions {
            vote_threshold: 10,
            suppression_radius: 3
        };

        b.iter(|| {
            let lines = detect_lines(&image, options);
            black_box(lines);
        });
    }
}