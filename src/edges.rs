use image::{Pixel, GenericImage, ImageBuffer, Luma};
use image::imageops::blur;
use gradients::{vertical_sobel, horizontal_sobel};

/// Runs the canny edge detection algorithm on the provided `ImageBuffer`.
///
/// Returns a binary image, where edge pixels have a value of 255 and non-edge pixels a value of 0.
pub fn canny<I>(image: &I,
                low_threshold: u8,
                high_threshold: u8)
                -> ImageBuffer<Luma<u8>, Vec<<Luma<u8> as Pixel>::Subpixel>>
    where I: GenericImage<Pixel = Luma<u8>> + 'static
{
    if high_threshold < low_threshold {
        panic!("Low threshold is greater than high threshold: {}; {}",
               low_threshold,
               high_threshold)
    }
    // Heavily based on the implementation proposed by wikipedia.
    // 1. Gaussian blur.
    const SIGMA: f32 = 1.4;
    let blurred = blur(image, SIGMA);

    // 2. Intensity of gradients.
    let gx = horizontal_sobel(&blurred);
    let gy = vertical_sobel(&blurred);
    let mut g = ImageBuffer::from_fn(image.width(), image.height(), |x, y| {
        let g = (gx[(x, y)][0] as f32).hypot(gy[(x, y)][0] as f32);
        Luma { data: [g as u8] }
    });

    // 3. Non-maximum-suppression (Make edges thinner)
    non_maximum_suppression(&mut g, &gx, &gy);

    // 4. Hysteresis to filter out edges based on thresholds.
    hysteresis(&g, low_threshold, high_threshold)
}

/// Finds local maxima to make the edges thinner.
fn non_maximum_suppression(g: &mut ImageBuffer<Luma<u8>, Vec<<Luma<u8> as Pixel>::Subpixel>>,
                           gx: &ImageBuffer<Luma<i16>, Vec<<Luma<i16> as Pixel>::Subpixel>>,
                           gy: &ImageBuffer<Luma<i16>, Vec<<Luma<i16> as Pixel>::Subpixel>>) {
    const RADIANS_TO_DEGREES: f32 = 57.2958;
    for x in 1..g.width() - 1 {
        for y in 1..g.height() - 1 {
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
            let (cmp1, cmp2) = match clamped_angle {
                0 => (*g.get_pixel(x, y - 1), *g.get_pixel(x, y + 1)),
                45 => (*g.get_pixel(x + 1, y + 1), *g.get_pixel(x - 1, y - 1)),
                90 => (*g.get_pixel(x + 1, y), *g.get_pixel(x - 1, y)),
                135 => (*g.get_pixel(x - 1, y + 1), *g.get_pixel(x + 1, y - 1)),
                _ => unreachable!(),
            };
            let pixel = *g.get_pixel(x, y);
            // If the pixel is not a local maximum, suppress it.
            if pixel[0] < cmp1[0] || pixel[0] < cmp2[0] {
                g.put_pixel(x, y, Luma { data: [0] });
            }
        }
    }
}

// Filter out edges with the thresholds.
fn hysteresis(g: &ImageBuffer<Luma<u8>, Vec<<Luma<u8> as Pixel>::Subpixel>>,
              low_thresh: u8,
              high_thresh: u8)
              -> ImageBuffer<Luma<u8>, Vec<<Luma<u8> as Pixel>::Subpixel>> {
    const MAX_BRIGHTNESS: Luma<u8> = Luma { data: [255u8] };
    const MIN_BRIGHTNESS: Luma<u8> = Luma { data: [0u8] };
    // Init output image as all black.
    let mut out = ImageBuffer::from_pixel(g.width(), g.height(), MIN_BRIGHTNESS);
    for x in 1..g.width() - 1 {
        for y in 1..g.height() - 1 {
            let inp_pix = *g.get_pixel(x, y);
            // Higher than high_thresh -> its definitely an edge.
            if inp_pix[0] >= high_thresh {
                out.put_pixel(x, y, MAX_BRIGHTNESS);
            } else if inp_pix[0] >= low_thresh && inp_pix[0] < high_thresh {
                // First, check if any neighbors from the input image are edges.
                let in_neighbors = [*g.get_pixel(x + 1, y),
                                    *g.get_pixel(x + 1, y + 1),
                                    *g.get_pixel(x, y + 1),
                                    *g.get_pixel(x - 1, y - 1),
                                    *g.get_pixel(x - 1, y),
                                    *g.get_pixel(x - 1, y + 1)];
                let mut edge_detected = false;
                for neighbor in &in_neighbors {
                    if neighbor[0] >= high_thresh {
                        out.put_pixel(x, y, MAX_BRIGHTNESS);
                        edge_detected = true;
                        break;
                    }
                }
                // Still not sure? check if any neighbors from the output image are edges.
                if !edge_detected {
                    let out_neighbors = [*out.get_pixel(x + 1, y),
                                         *out.get_pixel(x + 1, y + 1),
                                         *out.get_pixel(x, y + 1),
                                         *out.get_pixel(x - 1, y - 1),
                                         *out.get_pixel(x - 1, y),
                                         *out.get_pixel(x - 1, y + 1)];
                    for neighbor in &out_neighbors {
                        if neighbor[0] == MAX_BRIGHTNESS[0] {
                            out.put_pixel(x, y, MAX_BRIGHTNESS);
                            break;
                        }
                    }
                }
            }
        }
    }
    out
}
