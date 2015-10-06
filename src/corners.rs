//! Functions for corner point detection, also known as interest points.

use image::{
    GenericImage,
    Luma
};

use std::cmp;

/// A location and score for a detected corner.
/// The scores need not be comparable between different
/// corner detectors.
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Corner {
    pub x: u32,
    pub y: u32,
    pub score: f32
}

impl Corner {
    pub fn new(x: u32, y: u32, score: f32) -> Corner {
        Corner {x: x, y: y, score: score}
    }
}

/// Finds corners using FAST-12 features. Given a point P, define
/// C_P to be the 16-pixel Bresenham circle of radius 3 centred at P.
/// We define a point P of intensity I to be a corner if at least
/// 12 contiguous pixels in C_P have an intensity strictly
/// greater than I + threshold or strictly less than I - threshold.
/// The score of a corner is the greatest threshold for which the given
/// pixel still qualifies as a corner.
/// https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test
pub fn corners_fast12<I>(image: &I, threshold: u8) -> Vec<Corner>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast12(image, threshold, x, y) {
                let score = corner_score_fast12(image, threshold, x, y);
                corners.push(Corner::new(x,y, score as f32));
            }
        }
    }

    corners
}

// /// Returns all corners which have the highest score in the
// /// (2 * radius + 1) square block centred on them.
pub fn suppress_non_maximum(corners: &[Corner], radius: u32)
        -> Vec<Corner> {

    let mut ordered_corners = corners.to_vec();
    ordered_corners.sort_by(|c, d| {(c.y, c.x).cmp(&(d.y, d.x))});
    let height = match ordered_corners.last() {
        Some(corner) => corner.y,
        None => 0
    };

    let mut corners_by_row = vec![vec![]; (height + 1) as usize];
    for corner in ordered_corners.iter() {
        corners_by_row[corner.y as usize].push(corner);
    }

    let mut max_corners = vec![];
    for corner in ordered_corners.iter() {
        let cx = corner.x;
        let cy = corner.y;
        let cs = corner.score;

        let mut is_max = true;
        let min_row = if radius > cy {0} else {cy - radius};
        let max_row = if cy + radius > height {height} else {cy + radius};
        for y in min_row..max_row {
            for c in corners_by_row[y as usize].iter() {
                if c.x + radius < cx {
                    continue;
                }
                if c.x > cx + radius {
                    break;
                }
                if c.score > cs {
                    is_max = false;
                    break;
                }
                if c.score < cs {
                    continue;
                }
                // Break tiebreaks lexicographically
                if (c.y, c.x) < (cy, cx) {
                    is_max = false;
                    break;
                }
            }
            if !is_max {
                break;
            }
        }

        if is_max {
            max_corners.push(corner.clone());
        }
    }

    max_corners
}

/// The score of a corner detected using the FAST12
/// detector is the largest threshold for which this
/// pixel is still a corner. We input the threshold at which
/// the corner was detected as a lower bound on the search.
/// Note that the corner check uses a strict inequality, so if
/// the smallest intensity difference between the center pixel
/// and a corner pixel is n then the corner will have a score of n - 1.
fn corner_score_fast12<I>(image: &I, threshold: u8, x: u32, y: u32)
        -> u8
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let mut max = 255u8;
    let mut min = threshold;

    loop {
        if max == min {
            return max;
        }

        let mean = ((max as u16 + min as u16) / 2u16) as u8;
        let probe = if max == min + 1 { max } else { mean };

        if is_corner_fast12(image, probe, x, y) {
            min = probe;
        }
        else {
            max = probe - 1;
        }
    }
}

/// Checks if the given pixel is a corner.
fn is_corner_fast12<I>(image: &I, threshold: u8, x: u32, y: u32)
        -> bool
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();

    if x < 3 || y < 3 || x >= width - 3 || y >= height - 3 {
        return false;
    }

    let c = image.get_pixel(x, y)[0];
    let low_thresh: i16  = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // Circle pixels are labelled clockwise, starting at 0 directly above
    // the center, and finishing at 15, three pixels above and one to
    // the left of the center

    let p0 = image.get_pixel(x, y - 3)[0] as i16;
    let p8 = image.get_pixel(x, y + 3)[0] as i16;

    let mut above = p0 > high_thresh && p8 > high_thresh;
    let mut below = p0 < low_thresh  && p8 < low_thresh;

    if !above && !below {
        return false;
    }

    let p4  = image.get_pixel(x + 3, y)[0] as i16;
    let p12 = image.get_pixel(x - 3, y)[0] as i16;

    above = above && ((p4 > high_thresh) || (p12 > high_thresh));
    below = below && ((p4 < low_thresh)  || (p12 < low_thresh));

    if !above && !below {
        return false;
    }

    // TODO: Generate a list of pixel offsets once per image,
    // TODO: and use those offsets directly when reading pixels.
    // TODO: This is a little tricky as we can't always do it - we'd
    // TODO: need to distinguish between GenericImages and ImageBuffers.
    // TODO: We can also reduce the number of checks we do below.

    let mut pixels = [0i16; 16];

    pixels[0]  = p0;
    pixels[1]  = image.get_pixel(x + 1, y - 3)[0] as i16;
    pixels[2]  = image.get_pixel(x + 2, y - 2)[0] as i16;
    pixels[3]  = image.get_pixel(x + 3, y - 1)[0] as i16;
    pixels[4]  = p4;
    pixels[5]  = image.get_pixel(x + 3, y + 1)[0] as i16;
    pixels[6]  = image.get_pixel(x + 2, y + 2)[0] as i16;
    pixels[7]  = image.get_pixel(x + 1, y + 3)[0] as i16;
    pixels[8]  = p8;
    pixels[9]  = image.get_pixel(x - 1, y + 3)[0] as i16;
    pixels[10] = image.get_pixel(x - 2, y + 2)[0] as i16;
    pixels[11] = image.get_pixel(x - 3, y + 1)[0] as i16;
    pixels[12] = p12;
    pixels[13] = image.get_pixel(x - 3, y - 1)[0] as i16;
    pixels[14] = image.get_pixel(x - 2, y - 2)[0] as i16;
    pixels[15] = image.get_pixel(x - 1, y - 3)[0] as i16;


    if above {
        for i in 0..15 {
            let mut is_corner = true;
            let end   = i + 12;
            let upper = cmp::min(end, 16);
            let wrap  = if end > 16 { end - 16 } else { 0 };

            for j in i..upper {
                if pixels[j] <= high_thresh {
                    is_corner = false;
                    break;
                }
            }
            if is_corner {
                for j in 0..wrap {
                    if pixels[j] <= high_thresh {
                        is_corner = false;
                        break;
                    }
                }
            }
            if is_corner {
                return true;
            }
        }
    }
    else {
        for i in 0..15 {
            let mut is_corner = true;
            let end   = i + 12;
            let upper = cmp::min(end, 16);
            let wrap  = if end > 16 { end - 16 } else { 0 };

            for j in i..upper {
                if pixels[j] >= low_thresh {
                    is_corner = false;
                    break;
                }
            }
            if is_corner {
                for j in 0..wrap {
                    if pixels[j] >= low_thresh {
                        is_corner = false;
                        break;
                    }
                }
            }
            if is_corner {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod test {

    use super::{
        Corner,
        corner_score_fast12,
        is_corner_fast12,
        suppress_non_maximum
    };
    use image::{
        GrayImage,
        ImageBuffer
    };

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_darker_pixels_large_threshold() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 15, 3, 3), false);
    }

    #[test]
    fn test_is_corner_fast12_12_contiguous_lighter_pixels() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            00, 00, 10, 10, 10, 00, 00,
            00, 10, 00, 00, 00, 10, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            10, 00, 00, 00, 00, 00, 00,
            00, 10, 00, 00, 00, 00, 00,
            00, 00, 10, 10, 10, 00, 00]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), true);
    }

    #[test]
    fn test_is_corner_fast12_12_noncontiguous() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 00,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 3, 3), false);
    }

    #[test]
    fn test_is_corner_fast12_near_image_boundary() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        assert_eq!(is_corner_fast12(&image, 8, 1, 1), false);
    }

    #[test]
    fn test_corner_score_fast12() {
        let image: GrayImage = ImageBuffer::from_raw(7, 7, vec![
            10, 10, 00, 00, 00, 10, 10,
            10, 00, 10, 10, 10, 00, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            00, 10, 10, 10, 10, 10, 10,
            10, 00, 10, 10, 10, 10, 10,
            10, 10, 00, 00, 00, 10, 10]).unwrap();

        let score = corner_score_fast12(&image, 5, 3, 3);
        assert_eq!(score, 9);

        let score = corner_score_fast12(&image, 9, 3, 3);
        assert_eq!(score, 9);
    }

    #[test]
    fn test_suppress_non_maximum() {
        let corners = vec![
            // Suppress vertically
            Corner::new(0, 0, 10f32),
            Corner::new(0, 2, 8f32),
            // Suppress horizontally
            Corner::new(5, 5, 10f32),
            Corner::new(7, 5, 15f32),
            // Tiebreak
            Corner::new(12, 20, 10f32),
            Corner::new(13, 20, 10f32),
            Corner::new(13, 21, 10f32)
        ];

        let expected = vec![
            Corner::new(0, 0, 10f32),
            Corner::new(7, 5, 15f32),
            Corner::new(12, 20, 10f32)
        ];

        let max = suppress_non_maximum(&corners, 3);
        assert_eq!(max, expected);
    }
}
