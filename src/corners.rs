//! Functions for detecting corners, also known as interest points.

use crate::definitions::{Position, Score};
use image::{GenericImageView, GrayImage};

/// A location and score for a detected corner.
/// The scores need not be comparable between different
/// corner detectors.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Corner {
    /// x-coordinate of the corner.
    pub x: u32,
    /// y-coordinate of the corner.
    pub y: u32,
    /// Score of the detected corner.
    pub score: f32,
}

impl Corner {
    /// A corner at location (x, y) with score `score`.
    pub fn new(x: u32, y: u32, score: f32) -> Corner {
        Corner { x, y, score }
    }
}

impl Position for Corner {
    /// x-coordinate of the corner.
    fn x(&self) -> u32 {
        self.x
    }

    /// y-coordinate of the corner.
    fn y(&self) -> u32 {
        self.y
    }
}

impl Score for Corner {
    fn score(&self) -> f32 {
        self.score
    }
}

/// Variants of the [FAST](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test)
/// corner detector. These classify a point based on its intensity relative to the 16 pixels
/// in the Bresenham circle of radius 3 around it. A point P with intensity I is detected as a
/// corner if all pixels in a sufficiently long contiguous section of this circle either
/// all have intensity greater than I + t or all have intensity less than
/// I - t, for some user-provided threshold t. The score of a corner is
/// the greatest threshold for which the given pixel still qualifies as
/// a corner.
pub enum Fast {
    /// Corners require a section of length as least nine.
    Nine,
    /// Corners require a section of length as least twelve.
    Twelve,
}

/// Finds corners using FAST-12 features. See comment on `Fast`.
pub fn corners_fast12(image: &GrayImage, threshold: u8) -> Vec<Corner> {
    let (width, height) = image.dimensions();
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast12(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Twelve);
                corners.push(Corner::new(x, y, score as f32));
            }
        }
    }

    corners
}

/// Finds corners using FAST-9 features. See comment on Fast enum.
pub fn corners_fast9(image: &GrayImage, threshold: u8) -> Vec<Corner> {
    let (width, height) = image.dimensions();
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast9(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Nine);
                corners.push(Corner::new(x, y, score as f32));
            }
        }
    }

    corners
}

/// The score of a corner detected using the FAST
/// detector is the largest threshold for which this
/// pixel is still a corner. We input the threshold at which
/// the corner was detected as a lower bound on the search.
/// Note that the corner check uses a strict inequality, so if
/// the smallest intensity difference between the center pixel
/// and a corner pixel is n then the corner will have a score of n - 1.
pub fn fast_corner_score(image: &GrayImage, threshold: u8, x: u32, y: u32, variant: Fast) -> u8 {
    let mut max = 255u8;
    let mut min = threshold;

    loop {
        if max == min {
            return max;
        }

        let mean = ((max as u16 + min as u16) / 2u16) as u8;
        let probe = if max == min + 1 { max } else { mean };

        let is_corner = match variant {
            Fast::Nine => is_corner_fast9(image, probe, x, y),
            Fast::Twelve => is_corner_fast12(image, probe, x, y),
        };

        if is_corner {
            min = probe;
        } else {
            max = probe - 1;
        }
    }
}

// Note [FAST circle labels]
//
//          15 00 01
//       14          02
//     13              03
//     12       p      04
//     11              05
//       10          06
//          09 08 07

/// Checks if the given pixel is a corner according to the FAST9 detector.
/// The current implementation is extremely inefficient.
// TODO: Make this much faster!
fn is_corner_fast9(image: &GrayImage, threshold: u8, x: u32, y: u32) -> bool {
    // UNSAFETY JUSTIFICATION
    //  Benefit
    //      Removing all unsafe pixel accesses in this file makes
    //      bench_is_corner_fast9_9_contiguous_lighter_pixels 60% slower, and
    //      bench_is_corner_fast12_12_noncontiguous 40% slower
    //  Correctness
    //      All pixel accesses in this function, and in the called get_circle,
    //      access pixels with x-coordinate in the range [x - 3, x + 3] and
    //      y-coordinate in the range [y - 3, y + 3]. The precondition below
    //      guarantees that these are within image bounds.
    let (width, height) = image.dimensions();
    if x >= u32::max_value() - 3
        || y >= u32::max_value() - 3
        || x < 3
        || y < 3
        || width <= x + 3
        || height <= y + 3
    {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let c = unsafe { image.unsafe_get_pixel(x, y)[0] };
    let low_thresh: i16 = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    // JUSTIFICATION - see comment at the start of this function
    let (p0, p4, p8, p12) = unsafe {
        (
            image.unsafe_get_pixel(x, y - 3)[0] as i16,
            image.unsafe_get_pixel(x, y + 3)[0] as i16,
            image.unsafe_get_pixel(x + 3, y)[0] as i16,
            image.unsafe_get_pixel(x - 3, y)[0] as i16,
        )
    };

    let above = (p0 > high_thresh && p4 > high_thresh)
        || (p4 > high_thresh && p8 > high_thresh)
        || (p8 > high_thresh && p12 > high_thresh)
        || (p12 > high_thresh && p0 > high_thresh);

    let below = (p0 < low_thresh && p4 < low_thresh)
        || (p4 < low_thresh && p8 < low_thresh)
        || (p8 < low_thresh && p12 < low_thresh)
        || (p12 < low_thresh && p0 < low_thresh);

    if !above && !below {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let pixels = unsafe { get_circle(image, x, y, p0, p4, p8, p12) };

    // above and below could both be true
    (above && has_bright_span(&pixels, 9, high_thresh))
        || (below && has_dark_span(&pixels, 9, low_thresh))
}

/// Checks if the given pixel is a corner according to the FAST12 detector.
fn is_corner_fast12(image: &GrayImage, threshold: u8, x: u32, y: u32) -> bool {
    // UNSAFETY JUSTIFICATION
    //  Benefit
    //      Removing all unsafe pixel accesses in this file makes
    //      bench_is_corner_fast9_9_contiguous_lighter_pixels 60% slower, and
    //      bench_is_corner_fast12_12_noncontiguous 40% slower
    //  Correctness
    //      All pixel accesses in this function, and in the called get_circle,
    //      access pixels with x-coordinate in the range [x - 3, x + 3] and
    //      y-coordinate in the range [y - 3, y + 3]. The precondition below
    //      guarantees that these are within image bounds.
    let (width, height) = image.dimensions();
    if x >= u32::max_value() - 3
        || y >= u32::max_value() - 3
        || x < 3
        || y < 3
        || width <= x + 3
        || height <= y + 3
    {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let c = unsafe { image.unsafe_get_pixel(x, y)[0] };
    let low_thresh: i16 = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    // JUSTIFICATION - see comment at the start of this function
    let (p0, p8) = unsafe {
        (
            image.unsafe_get_pixel(x, y - 3)[0] as i16,
            image.unsafe_get_pixel(x, y + 3)[0] as i16,
        )
    };

    let mut above = p0 > high_thresh && p8 > high_thresh;
    let mut below = p0 < low_thresh && p8 < low_thresh;

    if !above && !below {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let (p4, p12) = unsafe {
        (
            image.unsafe_get_pixel(x + 3, y)[0] as i16,
            image.unsafe_get_pixel(x - 3, y)[0] as i16,
        )
    };

    above = above && ((p4 > high_thresh) || (p12 > high_thresh));
    below = below && ((p4 < low_thresh) || (p12 < low_thresh));

    if !above && !below {
        return false;
    }

    // TODO: Generate a list of pixel offsets once per image,
    // TODO: and use those offsets directly when reading pixels.
    // TODO: This is a little tricky as we can't always do it - we'd
    // TODO: need to distinguish between GenericImages and ImageBuffers.
    // TODO: We can also reduce the number of checks we do below.

    // JUSTIFICATION - see comment at the start of this function
    let pixels = unsafe { get_circle(image, x, y, p0, p4, p8, p12) };

    // Exactly one of above or below is true
    if above {
        has_bright_span(&pixels, 12, high_thresh)
    } else {
        has_dark_span(&pixels, 12, low_thresh)
    }
}

/// # Safety
///
/// The caller must ensure that:
///
///   x + 3 < image.width() &&
///   x >= 3 &&
///   y + 3 < image.height() &&
///   y >= 3
///
#[inline]
unsafe fn get_circle(
    image: &GrayImage,
    x: u32,
    y: u32,
    p0: i16,
    p4: i16,
    p8: i16,
    p12: i16,
) -> [i16; 16] {
    [
        p0,
        image.unsafe_get_pixel(x + 1, y - 3)[0] as i16,
        image.unsafe_get_pixel(x + 2, y - 2)[0] as i16,
        image.unsafe_get_pixel(x + 3, y - 1)[0] as i16,
        p4,
        image.unsafe_get_pixel(x + 3, y + 1)[0] as i16,
        image.unsafe_get_pixel(x + 2, y + 2)[0] as i16,
        image.unsafe_get_pixel(x + 1, y + 3)[0] as i16,
        p8,
        image.unsafe_get_pixel(x - 1, y + 3)[0] as i16,
        image.unsafe_get_pixel(x - 2, y + 2)[0] as i16,
        image.unsafe_get_pixel(x - 3, y + 1)[0] as i16,
        p12,
        image.unsafe_get_pixel(x - 3, y - 1)[0] as i16,
        image.unsafe_get_pixel(x - 2, y - 2)[0] as i16,
        image.unsafe_get_pixel(x - 1, y - 3)[0] as i16,
    ]
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly greater than the threshold.
fn has_bright_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c > threshold)
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly less than the threshold.
fn has_dark_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c < threshold)
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels match f condition.
fn search_span<F>(circle: &[i16; 16], length: u8, f: F) -> bool
where
    F: Fn(&i16) -> bool,
{
    if length > 16 {
        return false;
    }

    let mut nb_ok = 0u8;
    let mut nb_ok_start = None;

    for c in circle.iter() {
        if f(c) {
            nb_ok += 1;
            if nb_ok == length {
                return true;
            }
        } else {
            if nb_ok_start.is_none() {
                nb_ok_start = Some(nb_ok);
            }
            nb_ok = 0;
        }
    }

    nb_ok + nb_ok_start.unwrap() >= length
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::{black_box, Bencher};

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels_large_threshold() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        assert_eq!(is_corner_fast12(&image, 15, 3, 3), false);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_lighter_pixels() {
        let image = gray_image!(
            00, 00, 10, 10, 10, 00, 00;
            00, 10, 00, 00, 00, 10, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            00, 10, 00, 00, 00, 00, 00;
            00, 00, 10, 10, 10, 00, 00);

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_noncontiguous() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 00;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), false);
    }

    #[bench]
    fn bench_is_corner_fast12_12_noncontiguous(b: &mut Bencher) {
        let image = black_box(gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 00;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10));

        b.iter(|| black_box(is_corner_fast12(&image, 8, 3, 3)));
    }

    #[test]
    fn test_is_corner_fast12_near_image_boundary() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        assert_eq!(is_corner_fast12(&image, 8, 1, 1), false);
    }

    #[test]
    fn test_fast_corner_score_12() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        let score = fast_corner_score(&image, 5, 3, 3, Fast::Twelve);
        assert_eq!(score, 9);

        let score = fast_corner_score(&image, 9, 3, 3, Fast::Twelve);
        assert_eq!(score, 9);
    }

    #[test]
    fn test_is_corner_fast9_9_contiguous_darker_pixels() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 10);

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast9_9_contiguous_lighter_pixels() {
        let image = gray_image!(
            00, 00, 10, 10, 10, 00, 00;
            00, 10, 00, 00, 00, 10, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            00, 10, 00, 00, 00, 00, 00;
            00, 00, 00, 00, 00, 00, 00);

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), true);
    }

    #[bench]
    fn bench_is_corner_fast9_9_contiguous_lighter_pixels(b: &mut Bencher) {
        let image = black_box(gray_image!(
            00, 00, 10, 10, 10, 00, 00;
            00, 10, 00, 00, 00, 10, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            10, 00, 00, 00, 00, 00, 00;
            00, 10, 00, 00, 00, 00, 00;
            00, 00, 00, 00, 00, 00, 00));

        b.iter(|| black_box(is_corner_fast9(&image, 8, 3, 3)));
    }

    #[test]
    fn test_is_corner_fast9_12_noncontiguous() {
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 00;
            10, 00, 10, 10, 10, 10, 10;
            10, 10, 00, 00, 00, 10, 10);

        assert_eq!(is_corner_fast9(&image, 8, 3, 3), false);
    }

    #[test]
    fn test_corner_score_fast9() {
        // 8 pixels with an intensity diff of 20, then 1 with a diff of 10
        let image = gray_image!(
            10, 10, 00, 00, 00, 10, 10;
            10, 00, 10, 10, 10, 00, 10;
            00, 10, 10, 10, 10, 10, 10;
            00, 10, 10, 20, 10, 10, 10;
            00, 10, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 10;
            10, 10, 10, 10, 10, 10, 10);

        let score = fast_corner_score(&image, 5, 3, 3, Fast::Nine);
        assert_eq!(score, 9);

        let score = fast_corner_score(&image, 9, 3, 3, Fast::Nine);
        assert_eq!(score, 9);
    }
}
