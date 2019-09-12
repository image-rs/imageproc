//! Functions for finding and labelling connected components of an image.

use image::{GenericImage, GenericImageView, ImageBuffer, Luma};

use crate::definitions::Image;
use crate::union_find::DisjointSetForest;
use std::cmp;

/// Determines which neighbors of a pixel we consider
/// to be connected to it.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Connectivity {
    /// A pixel is connected to its N, S, E and W neighbors.
    Four,
    /// A pixel is connected to all of its neighbors.
    Eight,
}

/// Returns an image of the same size as the input, where each pixel
/// is labelled by the connected foreground component it belongs to,
/// or 0 if it's in the background. Input pixels are treated as belonging
/// to the background if and only if they are equal to the provided background pixel.
///
/// # Panics
/// Panics if the image contains 2<sup>32</sup> or more pixels. If this limitation causes you
/// problems then open an issue and we can rewrite this function to support larger images.
///
/// # Examples
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::region_labelling::{connected_components, Connectivity};
///
/// let background_color = Luma([0u8]);
///
/// let image = gray_image!(
///     1, 0, 1, 1;
///     0, 1, 1, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 1);
///
/// // With four-way connectivity the foreground regions which
/// // are only connected across diagonals belong to different
/// // connected components.
/// let components_four = gray_image!(type: u32,
///     1, 0, 2, 2;
///     0, 2, 2, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 3);
///
/// assert_pixels_eq!(
///     connected_components(&image, Connectivity::Four, background_color),
///     components_four);
///
/// // With eight-way connectivity all foreground pixels in the top two rows
/// // belong to the same connected component.
/// let components_eight = gray_image!(type: u32,
///     1, 0, 1, 1;
///     0, 1, 1, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 2);
///
/// assert_pixels_eq!(
///     connected_components(&image, Connectivity::Eight, background_color),
///     components_eight);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// // This example is like the first, except that not all of the input foreground
/// // pixels are the same color. Pixels of different color are never counted
/// // as belonging to the same connected component.
///
/// use image::Luma;
/// use imageproc::region_labelling::{connected_components, Connectivity};
///
/// let background_color = Luma([0u8]);
///
/// let image = gray_image!(
///     1, 0, 1, 1;
///     0, 1, 2, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 1);
///
/// let components_four = gray_image!(type: u32,
///     1, 0, 2, 2;
///     0, 3, 4, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 5);
///
/// assert_pixels_eq!(
///     connected_components(&image, Connectivity::Four, background_color),
///     components_four);
///
/// // If this behaviour is not what you want then you can first
/// // threshold the input image.
/// use imageproc::contrast::threshold;
///
/// // Pixels equal to the threshold are treated as background.
/// let thresholded = threshold(&image, 0);
///
/// let thresholded_components_four = gray_image!(type: u32,
///     1, 0, 2, 2;
///     0, 2, 2, 0;
///     0, 0, 0, 0;
///     0, 0, 0, 3);
///
/// assert_pixels_eq!(
///     connected_components(&thresholded, Connectivity::Four, background_color),
///     thresholded_components_four);
/// # }
/// ```
pub fn connected_components<I>(
    image: &I,
    conn: Connectivity,
    background: I::Pixel,
) -> Image<Luma<u32>>
where
    I: GenericImage,
    I::Pixel: Eq,
{
    let (width, height) = image.dimensions();
    let image_size = width as usize * height as usize;
    if image_size >= 2usize.saturating_pow(32) {
        panic!("Images with 2^32 or more pixels are not supported");
    }

    let mut out = ImageBuffer::new(width, height);

    // TODO: add macro to abandon early if either dimension is zero
    if width == 0 || height == 0 {
        return out;
    }

    let mut forest = DisjointSetForest::new(image_size);
    let mut adj_labels = [0u32; 4];
    let mut next_label = 1;

    for y in 0..height {
        for x in 0..width {
            let current = unsafe { image.unsafe_get_pixel(x, y) };
            if current == background {
                continue;
            }

            let mut num_adj = 0;

            if x > 0 {
                // West
                let pixel = unsafe { image.unsafe_get_pixel(x - 1, y) };
                if pixel == current {
                    let label = unsafe { out.unsafe_get_pixel(x - 1, y)[0] };
                    adj_labels[num_adj] = label;
                    num_adj += 1;
                }
            }

            if y > 0 {
                // North
                let pixel = unsafe { image.unsafe_get_pixel(x, y - 1) };
                if pixel == current {
                    let label = unsafe { out.unsafe_get_pixel(x, y - 1)[0] };
                    adj_labels[num_adj] = label;
                    num_adj += 1;
                }

                if conn == Connectivity::Eight {
                    if x > 0 {
                        // North West
                        let pixel = unsafe { image.unsafe_get_pixel(x - 1, y - 1) };
                        if pixel == current {
                            let label = unsafe { out.unsafe_get_pixel(x - 1, y - 1)[0] };
                            adj_labels[num_adj] = label;
                            num_adj += 1;
                        }
                    }
                    if x < width - 1 {
                        // North East
                        let pixel = unsafe { image.unsafe_get_pixel(x + 1, y - 1) };
                        if pixel == current {
                            let label = unsafe { out.unsafe_get_pixel(x + 1, y - 1)[0] };
                            adj_labels[num_adj] = label;
                            num_adj += 1;
                        }
                    }
                }
            }

            if num_adj == 0 {
                unsafe {
                    out.unsafe_put_pixel(x, y, Luma([next_label]));
                }
                next_label += 1;
            } else {
                let mut min_label = u32::max_value();
                for n in 0..num_adj {
                    min_label = cmp::min(min_label, adj_labels[n]);
                }
                unsafe {
                    out.unsafe_put_pixel(x, y, Luma([min_label]));
                }
                for n in 0..num_adj {
                    forest.union(min_label as usize, adj_labels[n] as usize);
                }
            }
        }
    }

    // Make components start at 1
    let mut output_labels = vec![0u32; image_size];
    let mut count = 1;

    unsafe {
        for y in 0..height {
            for x in 0..width {
                let label = {
                    if image.unsafe_get_pixel(x, y) == background {
                        continue;
                    }
                    out.unsafe_get_pixel(x, y)[0]
                };
                let root = forest.root(label as usize);
                let mut output_label = *output_labels.get_unchecked(root);
                if output_label < 1 {
                    output_label = count;
                    count += 1;
                }
                *output_labels.get_unchecked_mut(root) = output_label;
                out.unsafe_put_pixel(x, y, Luma([output_label]));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    extern crate wasm_bindgen_test;

    use super::connected_components;
    use super::Connectivity::{Eight, Four};
    use crate::definitions::{HasBlack, HasWhite};
    use ::test;
    use image::{GrayImage, ImageBuffer, Luma};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_connected_components_eight_white_background() {
        let image = gray_image!(
              1, 255,   2,   1;
            255,   1,   1, 255;
            255, 255, 255, 255;
            255, 255, 255,   1);

        let expected = gray_image!(type: u32,
            1, 0, 2, 1;
            0, 1, 1, 0;
            0, 0, 0, 0;
            0, 0, 0, 3);

        let labelled = connected_components(&image, Eight, Luma::white());
        assert_pixels_eq!(labelled, expected);
    }

    // One huge component with eight-way connectivity, loads of
    // isolated components with four-way conectivity.
    fn chessboard(width: u32, height: u32) -> GrayImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            if (x + y) % 2 == 0 {
                return Luma([255u8]);
            } else {
                return Luma([0u8]);
            }
        })
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_connected_components_eight_chessboard() {
        let image = chessboard(30, 30);
        let components = connected_components(&image, Eight, Luma::black());
        let max_component = components.pixels().map(|p| p[0]).max();
        assert_eq!(max_component, Some(1u32));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    fn test_connected_components_four_chessboard() {
        let image = chessboard(30, 30);
        let components = connected_components(&image, Four, Luma::black());
        let max_component = components.pixels().map(|p| p[0]).max();
        assert_eq!(max_component, Some(450u32));
    }

    #[bench]
    fn bench_connected_components_eight_chessboard(b: &mut test::Bencher) {
        let image = chessboard(300, 300);
        b.iter(|| {
            let components = connected_components(&image, Eight, Luma::black());
            test::black_box(components);
        });
    }

    #[bench]
    fn bench_connected_components_four_chessboard(b: &mut test::Bencher) {
        let image = chessboard(300, 300);
        b.iter(|| {
            let components = connected_components(&image, Four, Luma::black());
            test::black_box(components);
        });
    }
}
