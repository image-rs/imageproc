//! Functions for finding and labelling connected components of an image.

use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel};

use crate::definitions::{Image,Position};
use crate::union_find::DisjointSetForest;
use crate::point::Point;

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

/// Returns an [Iterator]<Item=[Point]<[u32]>> containing the indices of the nearest four/eight neighbors (based on the [Connectivity] value passed to the function) of the given point, performing boundary
/// checks. Note that (u32,u32) implements [From]<[Point]<[u32]>> so conversion is trivial.
/// 
/// ## Example
///
/// ```
/// /*
/// This simple image is used for the test below, where P is the point whose neighbors we want to
/// obtain. The 4s indicate the indices which will be yielded by [Connectivity::Four]; if
/// [Connectivity::Eight] is used instead, the indices of pixels labeled 8 will be included as
/// well. Note that this test covers an instance where the indices on the left side are out of
/// bounds -- they will not be included.
/// 
///   0 0 0 0 0
///   4 8 0 0 0
///   P 4 0 0 0
///   4 8 0 0 0
///   0 0 0 0 0
///
/// */
///
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::definitions::Position;
/// use imageproc::point::Point;
/// use imageproc::region_labelling::{Connectivity, neighbor_indices};
/// let pos = (0,2); // X and Y indices of the point P in the image above
///
/// // Nearest 4 neighbors
/// let mut neighbors: Vec<(u32,u32)> = neighbor_indices(&pos, 5, 5, &Connectivity::Four)
///     .map(|p| <(u32,u32)>::from(p))
///     .collect();
/// assert_eq!(neighbors.sort(), [(0,1), (1,2), (0,3)].to_vec().sort());
///
/// // Nearest 8 neighbors
/// neighbors = neighbor_indices(&pos, 5, 5, &Connectivity::Eight)
///     .map(|p| <(u32,u32)>::from(p))
///     .collect();
/// assert_eq!(neighbors.sort(), [(0,1), (1,1), (1,2), (0,3), (1,3)].to_vec().sort());
///
/// # }
/// ```
///
///
pub fn neighbor_indices<T: Position>(point: &T, height: u32, width: u32, connectivity: &Connectivity) -> impl Iterator<Item=Point<u32>> {
    let (x,y) = (point.x(), point.y());
    if y >= height || x >= width { // no need to bounds check for negative inputs as the Position trait guarantees u32 indices
        Vec::new().into_iter() // no allocation needed
    }
    else {
        let mut out = match connectivity {
            Connectivity::Four => Vec::with_capacity(4),
            Connectivity::Eight => Vec::with_capacity(8),
        };
        if y >= 1 {
            out.push(Point::new(x, y-1)); // North
        }
        if x >= 1 {
            out.push(Point::new(x-1, y)); // West
        }
        if x+1 < width {
            out.push(Point::new(x+1, y)); // East
        }
        if y+1 < height {
            out.push(Point::new(x, y+1)); // South
        }
        if connectivity == &Connectivity::Eight {
            if x >= 1 && y >= 1 {
                out.push(Point::new(x-1, y-1)); // Northwest
            }
            if x+1 < width && y >= 1 {
                out.push(Point::new(x+1, y-1)); // Northeast
            }
            if x+1 < width && y+1 < height {
                out.push(Point::new(x+1, y+1)); // Southeast
            }
            if x >= 1 && y+1 < height {
                out.push(Point::new(x-1, y+1)); // Southwest
            }
        }
        out.into_iter()
    }
}

/// Given a reference to an image, the index of a certain pixel, and a [Connectivity] value,
/// returns an iterator containing references to the neighboring pixels
/// 
/// ## Example
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
///
/// use image::Luma;
/// use imageproc::region_labelling::{neighbors,Connectivity};
/// use imageproc::point::Point;
///
///
/// let image = gray_image!(
///     4, 0, 1, 2, 1;
///     0, 3, 255, 4, 0;
///     0, 5, 6, 7, 1;
///     1, 0, 0, 1, 4);
///
/// let point: Point<u32> = Point::new(2, 1);
/// let n4: Vec<&Luma<u8>> = neighbors(&image, &point, &Connectivity::Four)
///     .collect();
/// assert_eq!(n4, [&image[(2,0)], &image[(1,1)], &image[(3,1)], &image[(2,2)]].to_vec());
/// 
///
/// # }
///
/// ```
pub fn neighbors<'a, T,P>(image: &'a Image<P>, point: &T, connectivity: &Connectivity) -> impl Iterator<Item= &'a P>
where
P: Pixel,
T: Position,
{
    let (width,height) = image.dimensions();
    neighbor_indices(point, height, width, connectivity)
        .map(move |n| {
            let tuple = <(u32,u32)>::from(n);
            &image[tuple]
        }
    )
}

/// Given a seed point to start at and a new color to fill the region with,
/// fill all connected pixels matching the original color of the seed point with the
/// new color
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
///
/// use image::Luma;
/// use imageproc::region_labelling::floodfill_mut;
///
/// let fill_color = Luma([5u8]);
///
/// let mut image = gray_image!(
///     4, 0, 1, 1, 1;
///     0, 1, 1, 2, 0;
///     0, 0, 1, 0, 1;
///     1, 0, 0, 1, 4);
/// let seed_point = (2, 1);
/// floodfill_mut(&mut image, &seed_point, fill_color);
///
/// let expected_output = gray_image!(
///     4, 0, 5, 5, 5;
///     0, 5, 5, 2, 0;
///     0, 0, 5, 0, 1;
///     1, 0, 0, 1, 4);
///
/// assert_pixels_eq!(image, expected_output);
///
/// # }
///
/// ```
pub fn floodfill_mut<P,T>(image: &mut Image<P>, seed_point: &T, fill_color: P)
where
    P: Pixel+PartialEq,
    T: Position
{
    let seed_point = (seed_point.x(), seed_point.y());
    let (width, height) = image.dimensions();
    let old_color = image[seed_point]; // connected pixels matching this color will be filled in with the fill_color
    if old_color == fill_color || seed_point.0 >= width || seed_point.1 >= height {
        return;
    }
    else {
        let mut frontier = Vec::with_capacity(50); // preallocation size chosen arbitrarily as the required size varies
        frontier.push(seed_point);
        image[seed_point] = fill_color;
        let connectivity = Connectivity::Four;
        while !frontier.is_empty() {
            let q = frontier.pop().unwrap(); // unwrap will not panic here as we have checked that frontier is not empty
            let neighbors = neighbor_indices(&q, height, width, &connectivity);
            for neighbor in neighbors {
                if image[neighbor.into()] == old_color {
                    let neighbor: (u32,u32) = neighbor.into();
                    frontier.push(neighbor);
                    image[neighbor] = fill_color;
                }
            }
        }

    }
}

/// Creates a clone of the input image and performs an in-place floodfill operation on 
/// the clone using [floodfill_mut]
///
pub fn floodfill<P,T>(image: &Image<P>, seed_point: &T, fill_color: P) -> Image<P>
where
    P: Pixel+PartialEq,
    T: Position
{
    let mut out = image.clone();
    floodfill_mut(&mut out, seed_point, fill_color);
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
