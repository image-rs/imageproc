//! Functions for creating and evaluating [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features).

use definitions::{HasBlack, HasWhite, Image};
use integralimage;
use image::{GenericImage, ImageBuffer, Luma};
use itertools::Itertools;
use std::marker::PhantomData;
use std::ops::Range;

/// A Haar-like filter.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HaarFilter {
    sign: Sign,
    filter_type: FilterType,
    block_size: Size<Pixels>,
    left: u8,
    top: u8,
}

/// Whether the top left region in a Haar filter is counted
/// with positive or negative sign.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Sign {
    /// Top left region is counted with a positive sign.
    Positive,
    /// Top left region is counted with a negative sign.
    Negative
}

/// The type of a Haar-like filter determines the number of regions and their orientation.
/// The diagrams in the comments for each variant use the symbols (*, &) to represent either
/// (+, -) or (-, +), depending on which `Sign` the filter type is used with.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum FilterType {
    /// Two horizontally-adjacent regions of equal width.
    /// <pre>
    ///      -----------
    ///     |  *  |  &  |
    ///      -----------
    /// </pre>
    TwoRegionHorizontal,
    /// Three horizontally-adjacent regions of equal width.
    /// <pre>
    ///      -----------------
    ///     |  *  |  &  |  *  |
    ///      -----------------
    /// </pre>
    ThreeRegionHorizontal,
    /// Two vertically-adjacent regions of equal height.
    /// <pre>
    ///      -----
    ///     |  *  |
    ///      -----
    ///     |  &  |
    ///      -----
    /// </pre>
    TwoRegionVertical,
    /// Three vertically-adjacent regions of equal height.
    /// <pre>
    ///      -----
    ///     |  *  |
    ///      -----
    ///     |  &  |
    ///      -----
    ///     |  *  |
    ///      -----
    /// </pre>
    ThreeRegionVertical,
    /// Four regions arranged in a two-by-two grid. The two columns
    /// have equal width and the two rows have equal height.
    /// <pre>
    ///      -----------
    ///     |  *  |  &  | 
    ///      -----------
    ///     |  &  |  *  |
    ///      -----------
    /// </pre>
    FourRegion
}

impl FilterType {
    // The width and height of a filter shape, in blocks.
    fn shape(&self) -> Size<Blocks> {
        match *self {
            FilterType::TwoRegionHorizontal => Size::new(2, 1),
            FilterType::ThreeRegionHorizontal => Size::new(3, 1),
            FilterType::TwoRegionVertical => Size::new(1, 2),
            FilterType::ThreeRegionVertical => Size::new(1, 3),
            FilterType::FourRegion => Size::new(2, 2)
        }
    }
}

impl HaarFilter {
    /// Evaluates the Haar filter on an integral image.
    pub fn evaluate(&self, integral: &Image<Luma<u32>>) -> i32 {
        // The corners of each block are lettered. Not all letters are evaluated for each filter type.
        // A   B   C   D
        //
        // E   F   G   H
        //
        // I   J   K   L
        //
        // M   N   O

        let a = self.block_boundary(0, 0);
        let b = self.block_boundary(1, 0);
        let e = self.block_boundary(0, 1);
        let f = self.block_boundary(1, 1);

        let sum = match self.filter_type {
            FilterType::TwoRegionHorizontal => {
                let c = self.block_boundary(2, 0);
                let g = self.block_boundary(2, 1);

                unsafe {
                    read(integral, a)
                        - 2 * read(integral, b)
                        + read(integral, c)
                        - read(integral, e)
                        + 2 * read(integral, f)
                        - read(integral, g)
                }
            },

            FilterType::ThreeRegionHorizontal => {
                let c = self.block_boundary(2, 0);
                let g = self.block_boundary(2, 1);
                let d = self.block_boundary(3, 0);
                let h = self.block_boundary(3, 1);

                unsafe {
                    read(integral, a)
                        - 2 * read(integral, b)
                        + 2 * read(integral, c)
                        - read(integral, d)
                        - read(integral, e)
                        + 2 * read(integral, f)
                        - 2 * read(integral, g)
                        + read(integral, h)
                }
            },

            FilterType::TwoRegionVertical => {
                let i = self.block_boundary(0, 2);
                let j = self.block_boundary(1, 2);

                unsafe {
                    read(integral, a)
                        - read(integral, b)
                        - 2 * read(integral, e)
                        + 2 * read(integral, f)
                        + read(integral, i)
                        - read(integral, j)
                }
            },

            FilterType::ThreeRegionVertical => {
                let i = self.block_boundary(0, 2);
                let j = self.block_boundary(1, 2);
                let m = self.block_boundary(0, 3);
                let n = self.block_boundary(1, 3);

                unsafe {
                    read(integral, a)
                        - read(integral, b)
                        - 2 * read(integral, e)
                        + 2 * read(integral, f)
                        + 2 * read(integral, i)
                        - 2 * read(integral, j)
                        - read(integral, m)
                        + read(integral, n)
                }
            },

            FilterType::FourRegion => {
                let c = self.block_boundary(2, 0);
                let g = self.block_boundary(2, 1);
                let i = self.block_boundary(0, 2);
                let j = self.block_boundary(1, 2);
                let k = self.block_boundary(2, 2);

                unsafe {
                    read(integral, a)
                        - 2 * read(integral, b)
                        + read(integral, c)
                        - 2 * read(integral, e)
                        + 4 * read(integral, f)
                        - 2 * read(integral, g)
                        + read(integral, i)
                        - 2 * read(integral, j)
                        + read(integral, k)
                }
            }
        };

        let mul = if self.sign == Sign::Positive { 1i32 } else { -1i32 };
        sum * mul
    }

    fn block_boundary(&self, x: u8, y: u8) -> (u8, u8) {
        (self.left + x * self.block_width(), self.top + y * self.block_height())
    }

    /// Width of this filter in blocks.
    fn blocks_wide(&self) -> u8 {
        self.filter_type.shape().width
    }

    /// Height of this filter in blocks.
    fn blocks_high(&self) -> u8 {
        self.filter_type.shape().height
    }

    /// Width of each block in pixels.
    fn block_width(&self) -> u8 {
        self.block_size.width
    }

    /// Height of each block in pixels.
    fn block_height(&self) -> u8 {
        self.block_size.height
    }
}

unsafe fn read(integral: &Image<Luma<u32>>, location: (u8, u8)) -> i32 {
    integral.unsafe_get_pixel(location.0 as u32, location.1 as u32)[0] as i32
}

// The total width and height of a filter with the given type and block size.
fn filter_size(filter_type: FilterType, block_size: Size<Pixels>) -> Size<Pixels> {
    let shape = filter_type.shape();
    Size::new(shape.width * block_size.width, shape.height * block_size.height)
}

/// Returns a vector of all valid Haar filters for an image with given width and height.
pub fn enumerate_haar_filters(frame_width: u8, frame_height: u8) -> Vec<HaarFilter> {
    let frame_size = Size::new(frame_width, frame_height);

    let filter_types = vec![
        FilterType::TwoRegionHorizontal,
        FilterType::ThreeRegionHorizontal,
        FilterType::TwoRegionVertical,
        FilterType::ThreeRegionVertical,
        FilterType::FourRegion
    ];

    filter_types
        .into_iter()
        .flat_map(|filter_type| haar_filters_of_type(filter_type, frame_size))
        .collect()
}

fn haar_filters_of_type(filter_type: FilterType, frame_size: Size<Pixels>) -> Vec<HaarFilter> {
    let mut filters = Vec::new();

    for block_size in block_sizes(filter_type.shape(), frame_size) {       
        for (left, top) in filter_positions(filter_size(filter_type, block_size), frame_size) {
            for &sign in [Sign::Positive, Sign::Negative].iter() {
                filters.push(HaarFilter { sign, filter_type, block_size, left, top });
            }
        }
    }

    filters
}

// Indicates that a size size is measured in pixels, e.g. the width of an individual block within a Haar-like filter.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct Pixels(u8);

// Indicates that a size is measured in blocks, e.g. the width of a Haar-like filter in blocks.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct Blocks(u8);

// A Size, measured either in pixels (T = Pixels) or in blocks (T = Blocks)
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
struct Size<T> {
    width: u8,
    height: u8,
    units: PhantomData<T>
}

impl<T> Size<T> {
    fn new(width: u8, height: u8) -> Size<T> {
        Size { 
            width: width,
            height: height,
            units: PhantomData
        }
    }
}

// Returns the valid block sizes for a feature of shape `feature_shape` in a frame of size `frame_size`.
fn block_sizes(feature_shape: Size<Blocks>, frame_size: Size<Pixels>) -> Vec<Size<Pixels>> {
    (1..frame_size.width / feature_shape.width + 1)
        .cartesian_product(1..frame_size.height / feature_shape.height + 1)
        .map(|(w, h)| Size::new(w, h))
        .collect()
}

// Returns the start positions for an interval of length `inner` for which the
// interval is wholly contained within an interval of length `outer`.
fn start_positions(inner: u8, outer: u8) -> Range<u8> {
    let upper = if inner > outer { 0 } else { outer - inner + 1};
    0..upper
}

// Returns all valid (left, top) coordinates for a filter of the given total size
fn filter_positions(feature_size: Size<Pixels>, frame_size: Size<Pixels>) -> Vec<(u8, u8)> {
    start_positions(feature_size.width, frame_size.width)
        .cartesian_product(start_positions(feature_size.height, frame_size.height))
        .collect()
}

/// Returns the number of distinct Haar filters for an image of the given dimensions.
/// 
/// Includes positive and negative, two and three region, vertical and horizontal filters,
/// as well as positive and negative four region filters.
///
/// Consider a k-region horizontal filter in an image of height 1 and width w. The largest valid block size
/// for such a filter is M = floor(w / k), and for a block size s there are `(w + 1) - 2 * s`
/// valid locations for the leftmost column of this filter.
/// Summing over s gives `M * (w + 1) - k * [(M * (M + 1)) / 2].
/// 
/// An equivalent argument applies vertically.
pub fn number_of_haar_filters(width: u32, height: u32) -> u32 {
    let num_positive_filters = 
        // Two-region horizontal
        num_filters(width, 2) * num_filters(height, 1)
        // Three-region horizontal
        + num_filters(width, 3) * num_filters(height, 1)
        // Two-region vertical
        + num_filters(width, 1) * num_filters(height, 2)
        // Three-region vertical
        + num_filters(width, 1) * num_filters(height, 3)
        // Four-region
        + num_filters(width, 2) * num_filters(height, 2);

    num_positive_filters * 2
}

fn num_filters(image_side: u32, num_blocks: u32) -> u32 {
    let m = image_side / num_blocks;
    m * (image_side + 1) - num_blocks * ((m * (m + 1)) / 2)
}

/// Draws the given Haar filter on an image, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_filter<I>(image: &I, filter: HaarFilter) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite + 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_haar_filter_mut(&mut out, filter);
    out
}

/// Draws the given Haar filter on an image in place, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_filter_mut<I>(image: &mut I, filter: HaarFilter)
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite
{
    let parity_shift = if filter.sign == Sign::Positive { 0 } else { 1 };

    for w in 0..filter.blocks_wide() {
        for h in 0..filter.blocks_high() {
            let parity = (w + h + parity_shift) % 2;
            let color = if parity == 0 { I::Pixel::white() } else { I::Pixel::black() };
            for x in 0..filter.block_width() {
                for y in 0..filter.block_height() {
                    let px = filter.left + w * filter.block_width() + x;
                    let py = filter.top + h * filter.block_height() + y;
                    image.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use image::{GrayImage, ImageBuffer};
    use integralimage::{integral_image};
    use utils::gray_bench_image;
    use test;

    #[test]
    fn test_block_sizes() {
        assert_eq!(
            block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(1, 1)), 
            vec![]);

        assert_eq!(
            block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(2, 1)), 
            vec![Size::new(1, 1)]);

        assert_eq!(
            block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(5, 1)),
            vec![Size::new(1, 1), Size::new(2, 1)]);

        assert_eq!(
            block_sizes(FilterType::TwoRegionVertical.shape(), Size::new(1, 2)),
            vec![Size::new(1, 1)]);
    }

    #[test]
    fn test_filter_positions() {
        assert_eq!(filter_positions(Size::new(2, 3), Size::new(2, 2)), 
            vec![]);

        assert_eq!(filter_positions(Size::new(2, 2), Size::new(2, 2)), 
            vec![(0, 0)]);

        assert_eq!(filter_positions(Size::new(2, 2), Size::new(3, 2)), 
            vec![(0, 0), (1, 0)]);

        assert_eq!(filter_positions(Size::new(2, 2), Size::new(3, 3)), 
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn test_number_of_haar_filters() {
        for h in 0..6 {
            for w in 0..6 {
                let filters = enumerate_haar_filters(w, h);
                let actual = filters.len() as u32;
                let expected = number_of_haar_filters(w as u32, h as u32);
                assert_eq!(actual, expected, "w = {}, h = {}", w, h);
            }
        }
    }

    #[test]
    fn test_two_region_horizontal() {
        let image = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8, 3u8,     4u8,     5u8,
                 /***+++++++++*****---------***/
            6u8, /**/7u8, 8u8,/**/ 9u8, 0u8,/**/
            9u8, /**/8u8, 7u8,/**/ 6u8, 5u8,/**/
            4u8, /**/3u8, 2u8,/**/ 1u8, 0u8,/**/
                 /***+++++++++*****---------***/
            6u8,     5u8, 4u8,     2u8, 1u8     ]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter { 
            sign: Sign::Positive,
            filter_type: FilterType::TwoRegionHorizontal,
            block_size: Size::new(2, 3),
            left: 1,
            top: 1
        };
        assert_eq!(filter.evaluate(&integral), 14i32);
    }

    #[test]
    fn test_three_region_vertical() {
        let image = ImageBuffer::from_raw(5, 5, vec![
        /*****************/
        /*-*/1u8, 2u8,/*-*/ 3u8, 4u8, 5u8,
        /*****************/
        /*+*/6u8, 7u8,/*+*/ 8u8, 9u8, 0u8,
        /*****************/
        /*-*/9u8, 8u8,/*-*/ 7u8, 6u8, 5u8,
        /*****************/
             4u8, 3u8,      2u8, 1u8, 0u8,
             6u8, 5u8,      4u8, 2u8, 1u8]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter { 
            sign: Sign::Negative,
            filter_type: FilterType::ThreeRegionVertical,
            block_size: Size::new(2, 1),
            left: 0,
            top: 0
        };
        assert_eq!(filter.evaluate(&integral), -7i32);
    }

    #[test]
    fn test_four_region() {
        let image = ImageBuffer::from_raw(5, 5, vec![
            /*****************************/
        1u8,/**/2u8, 3u8,/**/ 4u8, 5u8,/**/
        6u8,/**/7u8, 8u8,/**/ 9u8, 0u8,/**/ 
            /*****************************/
        9u8,/**/8u8, 7u8,/**/ 6u8, 5u8,/**/
        4u8,/**/3u8, 2u8,/**/ 1u8, 0u8,/**/
            /*****************************/
        6u8,    5u8, 4u8,     2u8, 1u8]).unwrap();

        let integral = integral_image(&image);
        let filter = HaarFilter { 
            sign: Sign::Positive,
            filter_type: FilterType::FourRegion,
            block_size: Size::new(2, 2),
            left: 1,
            top: 0
        };

        assert_eq!(filter.evaluate(&integral), -6i32);
    }

    // Reference implementation of Haar filter evaluation, to validate faster implementations against.
    fn reference_evaluate(filter: HaarFilter, integral: &Image<Luma<u32>>) -> i32 {
        let parity_shift = if filter.sign == Sign::Positive { 0 } else { 1 };

        let mut sum = 0i32;

        for w in 0..filter.blocks_wide() {
            let left = filter.left + filter.block_width() * w;
            let right = left + filter.block_width() - 1;

            for h in 0..filter.blocks_high() {
                let top = filter.top + filter.block_height() * h;
                let bottom = top + filter.block_height() - 1;
                let parity = (w + h + parity_shift) & 1;
                let multiplier = 1 - 2 * (parity as i32);

                let block_sum = integralimage::sum_image_pixels(integral, left as u32, top as u32, right as u32, bottom as u32) as i32;
                sum += multiplier * block_sum;
            }
        }

        sum
    }

    #[test]
    fn test_haar_evaluate_against_reference_implementation() {
        for w in 0..6 {
            for h in 0..6 {
                let filters = enumerate_haar_filters(w, h);
                let image = gray_bench_image(w as u32, h as u32);
                let integral = integral_image(&image);

                for filter in filters {
                    let actual = filter.evaluate(&integral);
                    let expected = reference_evaluate(filter, &integral);
                    assert_eq!(actual, expected, "w = {}, h = {}", w, h);
                }
            }
        }
    }

    #[test]
    fn test_draw_haar_filter_two_region_horizontal() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8, 3u8,     4u8, 5u8,
                 /***+++++++++*****---------***/
            6u8, /**/7u8, 8u8,/**/ 9u8, 0u8,/**/
            9u8, /**/8u8, 7u8,/**/ 6u8, 5u8,/**/
            4u8, /**/3u8, 2u8,/**/ 1u8, 0u8,/**/
                 /***+++++++++*****---------***/
            6u8,     5u8, 4u8,     2u8, 1u8     ]).unwrap();

        let filter = HaarFilter { 
            sign: Sign::Positive,
            filter_type: FilterType::TwoRegionHorizontal,
            block_size: Size::new(2, 3),
            left: 1,
            top: 1
        };
        let actual = draw_haar_filter(&image, filter);

        let expected = ImageBuffer::from_raw(5, 5, vec![
            1u8,     2u8,  3u8,        4u8,     5u8,
                 /***+++++++++++++*****---------***/
            6u8, /**/255u8, 255u8,/**/ 0u8, 0u8,/**/
            9u8, /**/255u8, 255u8,/**/ 0u8, 0u8,/**/
            4u8, /**/255u8, 255u8,/**/ 0u8, 0u8,/**/
                 /***+++++++++++++*****---------***/
            6u8,     5u8,   4u8,       2u8,     1u8]).unwrap();

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_draw_haar_filter_four_region() {
        let image: GrayImage = ImageBuffer::from_raw(5, 5, vec![
            /*****************************/
        1u8,/**/2u8, 3u8,/**/ 4u8, 5u8,/**/
        6u8,/**/7u8, 8u8,/**/ 9u8, 0u8,/**/ 
            /*****************************/
        9u8,/**/8u8, 7u8,/**/ 6u8, 5u8,/**/
        4u8,/**/3u8, 2u8,/**/ 1u8, 0u8,/**/
            /*****************************/
        6u8,    5u8, 4u8,     2u8, 1u8]).unwrap();

        let filter = HaarFilter { 
            sign: Sign::Positive,
            filter_type: FilterType::FourRegion,
            block_size: Size::new(2, 2),
            left: 1,
            top: 0
        };

        let actual = draw_haar_filter(&image, filter);

        let expected = ImageBuffer::from_raw(5, 5, vec![
            /*************************************/
        1u8,/**/255u8, 255u8,/**/ 0u8,   0u8,  /**/
        6u8,/**/255u8, 255u8,/**/ 0u8,   0u8,  /**/
            /*************************************/
        9u8,/**/0u8,   0u8,  /**/ 255u8, 255u8,/**/
        4u8,/**/0u8,   0u8,  /**/ 255u8, 255u8,/**/
            /*************************************/
        6u8,    5u8,   4u8,       2u8,   1u8      ]).unwrap();

        assert_pixels_eq!(actual, expected);
    }

    #[bench]
    fn bench_evaluate_all_filters_10x10(b: &mut test::Bencher) {
        // 10050 filters in total
        let filters = enumerate_haar_filters(10, 10);
        let image = gray_bench_image(10, 10);
        let integral = integral_image(&image);

        b.iter(|| {
            for filter in &filters {
                let x = filter.evaluate(&integral);
                test::black_box(x);
            }
        });
    }
}
