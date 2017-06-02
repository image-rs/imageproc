//! Functions for creating and evaluating [Haar-like features](https://en.wikipedia.org/wiki/Haar-like_features).

use definitions::{HasBlack, HasWhite, Image};
use integralimage;
use image::{GenericImage, ImageBuffer, Luma};
use itertools::Itertools;
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
pub enum Sign {
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
            FilterType::TwoRegionHorizontal => Size::new(Blocks(2), Blocks(1)),
            FilterType::ThreeRegionHorizontal => Size::new(Blocks(3), Blocks(1)),
            FilterType::TwoRegionVertical => Size::new(Blocks(1), Blocks(2)),
            FilterType::ThreeRegionVertical => Size::new(Blocks(1), Blocks(3)),
            FilterType::FourRegion => Size::new(Blocks(2), Blocks(2))
        }
    }
}

impl HaarFilter {
    /// Evaluates the Haar filter on an integral image.
    pub fn evaluate(&self, integral: &Image<Luma<u32>>) -> i32 {
        let blocks_wide = self.filter_type.shape().width.0;
        let blocks_high = self.filter_type.shape().height.0;
        let block_width = self.block_size.width.0;
        let block_height = self.block_size.height.0;

        let parity_shift = if self.sign == Sign::Positive { 0 } else { 1 };

        let mut sum = 0i32;

        // TODO: this does too much work - we don't need to evaluate all of the regions individually
        for w in 0..blocks_wide {
            let left = self.left + block_width * w;
            let right = left + block_width - 1;

            for h in 0..blocks_high {
                let top = self.top + block_height * h;
                let bottom = top + block_height - 1;

                let parity = (w + h + parity_shift) % 2;
                let multiplier = if parity == 0 { 1i32 } else { -1i32 };

                let block_sum = integralimage::sum_image_pixels(integral, left as u32, top as u32, right as u32, bottom as u32) as i32;
                sum += multiplier * block_sum;
            }
        }

        sum
    }
}

// The total width and height of a filter with the given type and block size.
fn filter_size(filter_type: FilterType, block_size: Size<Pixels>) -> Size<Pixels> {
    let shape = filter_type.shape();
    Size {
        width: Pixels(shape.width.0 * block_size.width.0),
        height: Pixels(shape.height.0 * block_size.height.0)
    }
}

/// Returns a vector of all valid Haar filters for an image with given width and height.
pub fn enumerate_haar_filters(frame_width: u8, frame_height: u8) -> Vec<HaarFilter> {
    let frame_size = Size::new(Pixels(frame_width), Pixels(frame_height));

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

// A marker trait to indicate that a type defines the units that some size is measured in.
trait Unit {}

// A size measured in pixels, e.g. the width of an individual block
// within a Haar-like filter.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct Pixels(u8);

impl Unit for Pixels {}

// A size measured in blocks, e.g. the width of a Haar-like filter in blocks.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct Blocks(u8);

impl Unit for Blocks {}

// A Size, measured either in pixels (T = Pixels) or in blocks (T = Blocks)
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
struct Size<T> {
    width: T,
    height: T,
}

impl<T: Unit> Size<T> {
    fn new(width: T, height: T) -> Size<T> {
        Size { 
            width: width, 
            height: height 
        }
    }
}

// Returns the valid block sizes for a feature of shape `feature_shape` in a frame of size `frame_size`.
fn block_sizes(feature_shape: Size<Blocks>, frame_size: Size<Pixels>) -> Vec<Size<Pixels>> {
    (1..frame_size.width.0 / feature_shape.width.0 + 1)
        .cartesian_product(1..frame_size.height.0 / feature_shape.height.0 + 1)
        .map(|(w, h)| Size::new(Pixels(w), Pixels(h)))
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
    start_positions(feature_size.width.0, frame_size.width.0)
        .cartesian_product(start_positions(feature_size.height.0, frame_size.height.0))
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
    let shape = filter.filter_type.shape();
    let blocks_wide = shape.width.0;
    let blocks_high = shape.height.0;
    let block_width = filter.block_size.width.0;
    let block_height = filter.block_size.height.0;

    let parity_shift = if filter.sign == Sign::Positive { 0 } else { 1 };

    for w in 0..blocks_wide {
        for h in 0..blocks_high {
            let parity = (w + h + parity_shift) % 2;
            let color = if parity == 0 { I::Pixel::white() } else { I::Pixel::black() };
            for x in 0..block_width {
                for y in 0..block_height {
                    let px = filter.left + w * block_width + x;
                    let py = filter.top + h * block_height + y;
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
        assert_eq!(block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(Pixels(1), Pixels(1))),
            vec![]);

        assert_eq!(block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(Pixels(2), Pixels(1))),
            vec![Size::new(Pixels(1), Pixels(1))]);

        assert_eq!(block_sizes(FilterType::TwoRegionHorizontal.shape(), Size::new(Pixels(5), Pixels(1))),
            vec![Size::new(Pixels(1), Pixels(1)), Size::new(Pixels(2), Pixels(1))]);

        assert_eq!(block_sizes(FilterType::TwoRegionVertical.shape(), Size::new(Pixels(1), Pixels(2))),
            vec![Size::new(Pixels(1), Pixels(1))]);
    }

    #[test]
    fn test_filter_positions() {
        assert_eq!(filter_positions(Size::new(Pixels(2), Pixels(3)), Size::new(Pixels(2), Pixels(2))), 
            vec![]);

        assert_eq!(filter_positions(Size::new(Pixels(2), Pixels(2)), Size::new(Pixels(2), Pixels(2))), 
            vec![(0, 0)]);

        assert_eq!(filter_positions(Size::new(Pixels(2), Pixels(2)), Size::new(Pixels(3), Pixels(2))), 
            vec![(0, 0), (1, 0)]);

        assert_eq!(filter_positions(Size::new(Pixels(2), Pixels(2)), Size::new(Pixels(3), Pixels(3))), 
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
            block_size: Size::new(Pixels(2), Pixels(3)),
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
            block_size: Size::new(Pixels(2), Pixels(1)),
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
            block_size: Size::new(Pixels(2), Pixels(2)),
            left: 1,
            top: 0
        };

        assert_eq!(filter.evaluate(&integral), -6i32);
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
            block_size: Size::new(Pixels(2), Pixels(3)),
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
            block_size: Size::new(Pixels(2), Pixels(2)),
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
