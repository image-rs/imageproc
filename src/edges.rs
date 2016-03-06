//! Functions for detecting edges in images.

use std::f32;
use image::{GenericImage, ImageBuffer, Luma};
use gradients::{vertical_sobel, horizontal_sobel};
use definitions::{HasWhite, HasBlack};
use filter::gaussian_blur_f32;

/// Runs the canny edge detection algorithm on the provided `ImageBuffer`.
///
/// # Params
///
/// - low_threshold: Low threshold for the hysteresis procedure.
/// Edges with a strength higher than the low threshold will appear
/// in the output image, if there are strong edges nearby.
/// - high_threshold: High threshold for the hysteresis procedure.
/// Edges with a strength higher than the high threshold will always
/// appear as edges in the output image.
///
/// Returns a binary image, where edge pixels have a value of 255 and non-edge pixels a value of 0.
pub fn canny<I>(image: &I,
                low_threshold: f32,
                high_threshold: f32)
                -> ImageBuffer<Luma<u8>, Vec<u8>>
    where I: GenericImage<Pixel = Luma<u8>> + 'static
{
    assert!(high_threshold >= low_threshold);
    // Heavily based on the implementation proposed by wikipedia.
    // 1. Gaussian blur.
    const SIGMA: f32 = 1.4;
    let blurred = gaussian_blur_f32(image, SIGMA);

    // 2. Intensity of gradients.
    let gx = horizontal_sobel(&blurred);
    let gy = vertical_sobel(&blurred);
    let g = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let g = (gx[(x, y)][0] as f32).hypot(gy[(x, y)][0] as f32);
        Luma { data: [g] }
    });

    // 3. Non-maximum-suppression (Make edges thinner)
    let thinned = non_maximum_suppression(&g, &gx, &gy);

    // 4. Hysteresis to filter out edges based on thresholds.
    hysteresis(&thinned, low_threshold, high_threshold)
}

/// Finds local maxima to make the edges thinner.
fn non_maximum_suppression(g: &ImageBuffer<Luma<f32>, Vec<f32>>,
                           gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
                           gy: &ImageBuffer<Luma<i16>, Vec<i16>>)
                           -> ImageBuffer<Luma<f32>, Vec<f32>> {
    const RADIANS_TO_DEGREES: f32 = 180f32 / f32::consts::PI;
    let mut out = ImageBuffer::from_pixel(g.width(), g.height(), Luma { data: [0.0] });
    for y in 1..g.height() - 1 {
        for x in 1..g.width() - 1 {
            let mut angle = (((gy[(x, y)][0] as f32).atan2(gx[(x, y)][0] as f32) *
                              RADIANS_TO_DEGREES) as i16) as f32;
            if angle < 0.0 {
                angle += 180.0
            }
            // Clamp angle.
            let clamped_angle = if angle >= 157.5 || angle < 22.5 {
                0
            } else if angle >= 22.5 && angle < 67.5 {
                45
            } else if angle >= 67.5 && angle < 112.5 {
                90
            } else if angle >= 112.5 && angle < 157.5 {
                135
            } else {
                unreachable!()
            };

            // Get the two perpendicular neighbors.
            let (cmp1, cmp2) = unsafe {
                match clamped_angle {
                    0 => (g.unsafe_get_pixel(x, y - 1), g.unsafe_get_pixel(x, y + 1)),
                    45 => {
                        (g.unsafe_get_pixel(x + 1, y + 1),
                         g.unsafe_get_pixel(x - 1, y - 1))
                    }
                    90 => (g.unsafe_get_pixel(x + 1, y), g.unsafe_get_pixel(x - 1, y)),
                    135 => {
                        (g.unsafe_get_pixel(x - 1, y + 1),
                         g.unsafe_get_pixel(x + 1, y - 1))
                    }
                    _ => unreachable!(),
                }
            };
            let pixel = *g.get_pixel(x, y);
            // If the pixel is not a local maximum, suppress it.
            if pixel[0] < cmp1[0] || pixel[0] < cmp2[0] {
                out.put_pixel(x, y, Luma { data: [0.0] });
            } else {
                out.put_pixel(x, y, pixel);
            }
        }
    }
    out
}

/// Filter out edges with the thresholds.
/// Non-recursive breadth-first search.
fn hysteresis(input: &ImageBuffer<Luma<f32>, Vec<f32>>,
              low_thresh: f32,
              high_thresh: f32)
              -> ImageBuffer<Luma<u8>, Vec<u8>> {

    let max_brightness = Luma::white();
    let min_brightness = Luma::black();
    // Init output image as all black.
    let mut out = ImageBuffer::from_pixel(input.width(), input.height(), min_brightness);
    // Stack. Possible optimization: Use previously allocated memory, i.e. gx.
    let mut edges = Vec::with_capacity(((input.width() * input.height()) / 2) as usize);
    for y in 1..input.height() - 1 {
        for x in 1..input.width() - 1 {
            let inp_pix = *input.get_pixel(x, y);
            let out_pix = *out.get_pixel(x, y);
            // If the edge strength is higher than high_thresh, mark it as an edge.
            if inp_pix[0] >= high_thresh && out_pix[0] == 0 {
                out.put_pixel(x, y, max_brightness);
                edges.push((x, y));
                // Track neighbors until no neighbor is >= low_thresh.
                while edges.len() > 0 {
                    let (nx, ny) = edges.pop().unwrap();
                    let neighbor_indices = [(nx + 1, ny),
                                            (nx + 1, ny + 1),
                                            (nx, ny + 1),
                                            (nx - 1, ny - 1),
                                            (nx - 1, ny),
                                            (nx - 1, ny + 1)];

                    for neighbor_idx in &neighbor_indices {
                        let in_neighbor = *input.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        let out_neighbor = *out.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        if in_neighbor[0] >= low_thresh && out_neighbor[0] == 0 {
                            out.put_pixel(neighbor_idx.0, neighbor_idx.1, max_brightness);
                            edges.push((neighbor_idx.0, neighbor_idx.1));
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::canny;
    use utils::edge_detect_bench_image;
    use test;

    #[bench]
    fn bench_canny(b: &mut test::Bencher) {
        let image = edge_detect_bench_image(250, 250);
        b.iter(|| {
            let output = canny(&image, 250.0, 300.0);
            test::black_box(output);
        });
    }
}
