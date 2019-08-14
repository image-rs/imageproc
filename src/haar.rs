//! Functions for creating and evaluating [Haar-like features].
//!
//! [Haar-like features]: https://en.wikipedia.org/wiki/Haar-like_features

use crate::definitions::{HasBlack, HasWhite, Image};
use image::{GenericImage, GenericImageView, ImageBuffer, Luma};
use itertools::Itertools;
use std::marker::PhantomData;
use std::ops::Range;

/// A [Haar-like feature].
///
/// [Haar-like feature]: https://en.wikipedia.org/wiki/Haar-like_features
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HaarFeature {
    sign: Sign,
    feature_type: HaarFeatureType,
    block_size: Size<Pixels>,
    left: u8,
    top: u8,
}

/// Whether the top left region in a Haar-like feature is counted
/// with positive or negative sign.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Sign {
    /// Top left region is counted with a positive sign.
    Positive,
    /// Top left region is counted with a negative sign.
    Negative
}

/// The type of a Haar-like feature determines the number of regions it contains and their orientation.
/// The diagrams in the comments for each variant use the symbols (*, &) to represent either
/// (+, -) or (-, +), depending on which `Sign` the feature type is used with.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum HaarFeatureType {
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

impl HaarFeatureType {
    // The width and height of Haar-like feature, in blocks.
    fn shape(&self) -> Size<Blocks> {
        match *self {
            HaarFeatureType::TwoRegionHorizontal => Size::new(2, 1),
            HaarFeatureType::ThreeRegionHorizontal => Size::new(3, 1),
            HaarFeatureType::TwoRegionVertical => Size::new(1, 2),
            HaarFeatureType::ThreeRegionVertical => Size::new(1, 3),
            HaarFeatureType::FourRegion => Size::new(2, 2)
        }
    }
}

impl HaarFeature {
    /// Evaluates the Haar-like feature on an integral image.
    pub fn evaluate(&self, integral: &Image<Luma<u32>>) -> i32 {
        // This check increases the run time of bench_evaluate_all_features_10x10
        // by approximately 16%. Without it this function is unsafe as insufficiently
        // large input images result in out of bounds accesses.
        //
        // We could alternatively create a new API where an image and a set of filters
        // are validated to be compatible up front, or just mark the function
        // as unsafe and document the requirement on image size.
        let size = feature_size(self.feature_type, self.block_size);
        assert!(integral.width() > size.width as u32 + self.left as u32);
        assert!(integral.height() > size.height as u32 + self.top as u32);

        // The corners of each block are lettered. Not all letters are evaluated for each feature type.
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

        let sum = match self.feature_type {
            HaarFeatureType::TwoRegionHorizontal => {
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

            HaarFeatureType::ThreeRegionHorizontal => {
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

            HaarFeatureType::TwoRegionVertical => {
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

            HaarFeatureType::ThreeRegionVertical => {
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

            HaarFeatureType::FourRegion => {
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

    /// Width of this feature in blocks.
    fn blocks_wide(&self) -> u8 {
        self.feature_type.shape().width
    }

    /// Height of this feature in blocks.
    fn blocks_high(&self) -> u8 {
        self.feature_type.shape().height
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

// The total width and height of a feature with the given type and block size.
fn feature_size(feature_type: HaarFeatureType, block_size: Size<Pixels>) -> Size<Pixels> {
    let shape = feature_type.shape();
    Size::new(shape.width * block_size.width, shape.height * block_size.height)
}

/// Returns a vector of all valid Haar-like features for an image with given width and height.
pub fn enumerate_haar_features(frame_width: u8, frame_height: u8) -> Vec<HaarFeature> {
    let frame_size = Size::new(frame_width, frame_height);

    let feature_types = vec![
        HaarFeatureType::TwoRegionHorizontal,
        HaarFeatureType::ThreeRegionHorizontal,
        HaarFeatureType::TwoRegionVertical,
        HaarFeatureType::ThreeRegionVertical,
        HaarFeatureType::FourRegion
    ];

    feature_types
        .into_iter()
        .flat_map(|feature_type| haar_features_of_type(feature_type, frame_size))
        .collect()
}

fn haar_features_of_type(feature_type: HaarFeatureType, frame_size: Size<Pixels>) -> Vec<HaarFeature> {
    let mut features = Vec::new();

    for block_size in block_sizes(feature_type.shape(), frame_size) {
        for (left, top) in feature_positions(feature_size(feature_type, block_size), frame_size) {
            for &sign in [Sign::Positive, Sign::Negative].iter() {
                features.push(HaarFeature { sign, feature_type, block_size, left, top });
            }
        }
    }

    features
}

// Indicates that a size size is measured in pixels, e.g. the width of an individual block within a Haar-like feature.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
struct Pixels(u8);

// Indicates that a size is measured in blocks, e.g. the width of a Haar-like feature in blocks.
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
            width,
            height,
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

// Returns all valid (left, top) coordinates for a feature of the given total size
fn feature_positions(feature_size: Size<Pixels>, frame_size: Size<Pixels>) -> Vec<(u8, u8)> {
    start_positions(feature_size.width, frame_size.width)
        .cartesian_product(start_positions(feature_size.height, frame_size.height))
        .collect()
}

/// Returns the number of distinct Haar-like features for an image of the given dimensions.
///
/// Includes positive and negative, two and three region, vertical and horizontal features,
/// as well as positive and negative four region features.
///
/// Consider a `k`-region horizontal feature in an image of height `1` and width `w`. The largest valid block size
/// for such a feature is `M = floor(w / k)`, and for a block size `s` there are `(w + 1) - 2 * s`
/// valid locations for the leftmost column of this feature.
/// Summing over `s` gives `M * (w + 1) - k * [(M * (M + 1)) / 2]`.
///
/// An equivalent argument applies vertically.
pub fn number_of_haar_features(width: u32, height: u32) -> u32 {
    let num_positive_features =
        // Two-region horizontal
        num_features(width, 2) * num_features(height, 1)
        // Three-region horizontal
        + num_features(width, 3) * num_features(height, 1)
        // Two-region vertical
        + num_features(width, 1) * num_features(height, 2)
        // Three-region vertical
        + num_features(width, 1) * num_features(height, 3)
        // Four-region
        + num_features(width, 2) * num_features(height, 2);

    num_positive_features * 2
}

fn num_features(image_side: u32, num_blocks: u32) -> u32 {
    let m = image_side / num_blocks;
    m * (image_side + 1) - num_blocks * ((m * (m + 1)) / 2)
}

/// Draws the given Haar-like feature on an image, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_feature<I>(image: &I, feature: HaarFeature) -> Image<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite + 'static
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_haar_feature_mut(&mut out, feature);
    out
}

/// Draws the given Haar-like feature on an image in place, drawing pixels
/// with a positive sign white and those with a negative sign black.
pub fn draw_haar_feature_mut<I>(image: &mut I, feature: HaarFeature)
    where I: GenericImage,
          I::Pixel: HasBlack + HasWhite
{
    let parity_shift = if feature.sign == Sign::Positive { 0 } else { 1 };

    for w in 0..feature.blocks_wide() {
        for h in 0..feature.blocks_high() {
            let parity = (w + h + parity_shift) % 2;
            let color = if parity == 0 { I::Pixel::white() } else { I::Pixel::black() };
            for x in 0..feature.block_width() {
                for y in 0..feature.block_height() {
                    let px = feature.left + w * feature.block_width() + x;
                    let py = feature.top + h * feature.block_height() + y;
                    image.put_pixel(px as u32, py as u32, color);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integral_image::{integral_image, sum_image_pixels};
    use crate::utils::gray_bench_image;
    use ::test;

    #[test]
    fn test_block_sizes() {
        assert_eq!(
            block_sizes(HaarFeatureType::TwoRegionHorizontal.shape(), Size::new(1, 1)),
            vec![]);

        assert_eq!(
            block_sizes(HaarFeatureType::TwoRegionHorizontal.shape(), Size::new(2, 1)),
            vec![Size::new(1, 1)]);

        assert_eq!(
            block_sizes(HaarFeatureType::TwoRegionHorizontal.shape(), Size::new(5, 1)),
            vec![Size::new(1, 1), Size::new(2, 1)]);

        assert_eq!(
            block_sizes(HaarFeatureType::TwoRegionVertical.shape(), Size::new(1, 2)),
            vec![Size::new(1, 1)]);
    }

    #[test]
    fn test_feature_positions() {
        assert_eq!(feature_positions(Size::new(2, 3), Size::new(2, 2)),
            vec![]);

        assert_eq!(feature_positions(Size::new(2, 2), Size::new(2, 2)),
            vec![(0, 0)]);

        assert_eq!(feature_positions(Size::new(2, 2), Size::new(3, 2)),
            vec![(0, 0), (1, 0)]);

        assert_eq!(feature_positions(Size::new(2, 2), Size::new(3, 3)),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn test_number_of_haar_features() {
        for h in 0..6 {
            for w in 0..6 {
                let features = enumerate_haar_features(w, h);
                let actual = features.len() as u32;
                let expected = number_of_haar_features(w as u32, h as u32);
                assert_eq!(actual, expected, "w = {}, h = {}", w, h);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_haar_invalid_image_size_top_left() {
        let image = gray_image!(type: u32, 0, 0; 0, 1);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(1, 1),
            left: 0,
            top: 0
        };
        // For a haar feature of width 2 the input image needs to have width
        // at least 2, and so its integral image needs to have width at least 3.
        let _ = feature.evaluate(&image);
    }

    #[test]
    fn test_haar_valid_image_size_top_left() {
        let image = gray_image!(type: u32, 0, 0, 0; 0, 1, 1);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(1, 1),
            left: 0,
            top: 0
        };
        let x = feature.evaluate(&image);
        assert_eq!(x, 1);
    }

    #[test]
    #[should_panic]
    fn test_haar_invalid_image_size_with_offset_feature() {
        let image = gray_image!(type: u32, 0, 0, 0; 0, 1, 1);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(1, 1),
            left: 1,
            top: 0
        };
        // The feature's left offset would result in attempting to
        // read outside the image boundaries
        let _ = feature.evaluate(&image);
    }

    #[test]
    fn test_haar_valid_image_size_with_offset_feature() {
        let image = gray_image!(type: u32, 0, 0, 0, 0; 0, 1, 1, 1);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(1, 1),
            left: 1,
            top: 0
        };
        let x = feature.evaluate(&image);
        assert_eq!(x, 0);
    }

    #[test]
    fn test_two_region_horizontal() {
        let image = gray_image!(
            1u8,     2u8, 3u8,     4u8,     5u8;
                 /***+++++++++*****---------***/
            6u8, /**/7u8, 8u8,/**/ 9u8, 0u8;/**/
            9u8, /**/8u8, 7u8,/**/ 6u8, 5u8;/**/
            4u8, /**/3u8, 2u8,/**/ 1u8, 0u8;/**/
                 /***+++++++++*****---------***/
            6u8,     5u8, 4u8,     2u8, 1u8     );

        let integral = integral_image(&image);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(2, 3),
            left: 1,
            top: 1
        };
        assert_eq!(feature.evaluate(&integral), 14i32);
    }

    #[test]
    fn test_three_region_vertical() {
        let image = gray_image!(
        /*****************/
        /*-*/1u8, 2u8,/*-*/ 3u8, 4u8, 5u8;
        /*****************/
        /*+*/6u8, 7u8,/*+*/ 8u8, 9u8, 0u8;
        /*****************/
        /*-*/9u8, 8u8,/*-*/ 7u8, 6u8, 5u8;
        /*****************/
             4u8, 3u8,      2u8, 1u8, 0u8;
             6u8, 5u8,      4u8, 2u8, 1u8);

        let integral = integral_image(&image);
        let feature = HaarFeature {
            sign: Sign::Negative,
            feature_type: HaarFeatureType::ThreeRegionVertical,
            block_size: Size::new(2, 1),
            left: 0,
            top: 0
        };
        assert_eq!(feature.evaluate(&integral), -7i32);
    }

    #[test]
    fn test_four_region() {
        let image = gray_image!(
            /*****************************/
        1u8,/**/2u8, 3u8,/**/ 4u8, 5u8;/**/
        6u8,/**/7u8, 8u8,/**/ 9u8, 0u8;/**/
            /*****************************/
        9u8,/**/8u8, 7u8,/**/ 6u8, 5u8;/**/
        4u8,/**/3u8, 2u8,/**/ 1u8, 0u8;/**/
            /*****************************/
        6u8,    5u8, 4u8,     2u8, 1u8);

        let integral = integral_image(&image);
        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::FourRegion,
            block_size: Size::new(2, 2),
            left: 1,
            top: 0
        };

        assert_eq!(feature.evaluate(&integral), -6i32);
    }

    // Reference implementation of Haar-like feature evaluation, to validate faster implementations against.
    fn reference_evaluate(feature: HaarFeature, integral: &Image<Luma<u32>>) -> i32 {
        let parity_shift = if feature.sign == Sign::Positive { 0 } else { 1 };

        let mut sum = 0i32;

        for w in 0..feature.blocks_wide() {
            let left = feature.left + feature.block_width() * w;
            let right = left + feature.block_width() - 1;

            for h in 0..feature.blocks_high() {
                let top = feature.top + feature.block_height() * h;
                let bottom = top + feature.block_height() - 1;
                let parity = (w + h + parity_shift) & 1;
                let multiplier = 1 - 2 * (parity as i32);

                let block_sum = sum_image_pixels(integral, left as u32, top as u32, right as u32, bottom as u32)[0] as i32;
                sum += multiplier * block_sum;
            }
        }

        sum
    }

    #[test]
    fn test_haar_evaluate_against_reference_implementation() {
        for w in 0..6 {
            for h in 0..6 {
                let features = enumerate_haar_features(w, h);
                let image = gray_bench_image(w as u32, h as u32);
                let integral = integral_image(&image);

                for feature in features {
                    let actual = feature.evaluate(&integral);
                    let expected = reference_evaluate(feature, &integral);
                    assert_eq!(actual, expected, "w = {}, h = {}", w, h);
                }
            }
        }
    }

    #[test]
    fn test_draw_haar_feature_two_region_horizontal() {
        let image = gray_image!(
            1u8,     2u8, 3u8,     4u8, 5u8;
                 /***+++++++++*****---------***/
            6u8, /**/7u8, 8u8,/**/ 9u8, 0u8;/**/
            9u8, /**/8u8, 7u8,/**/ 6u8, 5u8;/**/
            4u8, /**/3u8, 2u8,/**/ 1u8, 0u8;/**/
                 /***+++++++++*****---------***/
            6u8,     5u8, 4u8,     2u8, 1u8);

        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::TwoRegionHorizontal,
            block_size: Size::new(2, 3),
            left: 1,
            top: 1
        };
        let actual = draw_haar_feature(&image, feature);

        let expected = gray_image!(
            1u8,     2u8,  3u8,        4u8,     5u8;
                 /***+++++++++++++*****---------***/
            6u8, /**/255u8, 255u8,/**/ 0u8, 0u8;/**/
            9u8, /**/255u8, 255u8,/**/ 0u8, 0u8;/**/
            4u8, /**/255u8, 255u8,/**/ 0u8, 0u8;/**/
                 /***+++++++++++++*****---------***/
            6u8,     5u8,   4u8,       2u8,     1u8);

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_draw_haar_feature_four_region() {
        let image = gray_image!(
            /*****************************/
        1u8,/**/2u8, 3u8,/**/ 4u8, 5u8;/**/
        6u8,/**/7u8, 8u8,/**/ 9u8, 0u8;/**/
            /*****************************/
        9u8,/**/8u8, 7u8,/**/ 6u8, 5u8;/**/
        4u8,/**/3u8, 2u8,/**/ 1u8, 0u8;/**/
            /*****************************/
        6u8,    5u8, 4u8,     2u8, 1u8);

        let feature = HaarFeature {
            sign: Sign::Positive,
            feature_type: HaarFeatureType::FourRegion,
            block_size: Size::new(2, 2),
            left: 1,
            top: 0
        };

        let actual = draw_haar_feature(&image, feature);

        let expected = gray_image!(
            /*************************************/
        1u8,/**/255u8, 255u8,/**/ 0u8,   0u8;  /**/
        6u8,/**/255u8, 255u8,/**/ 0u8,   0u8;  /**/
            /*************************************/
        9u8,/**/0u8,   0u8,  /**/ 255u8, 255u8;/**/
        4u8,/**/0u8,   0u8,  /**/ 255u8, 255u8;/**/
            /*************************************/
        6u8,    5u8,   4u8,       2u8,   1u8);

        assert_pixels_eq!(actual, expected);
    }

    #[bench]
    fn bench_evaluate_all_features_10x10(b: &mut test::Bencher) {
        // 10050 features in total
        let features = enumerate_haar_features(10, 10);
        let image = gray_bench_image(10, 10);
        let integral = integral_image(&image);

        b.iter(|| {
            for feature in &features {
                let x = feature.evaluate(&integral);
                test::black_box(x);
            }
        });
    }
}
