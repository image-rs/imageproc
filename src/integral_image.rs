//! Functions for computing [integral images](https://en.wikipedia.org/wiki/Summed_area_table)
//! and running sums of rows and columns.

use image::{Luma, GrayImage, GenericImageView, Pixel, Primitive, Rgb, Rgba};
use crate::definitions::Image;
use crate::map::{ChannelMap, WithChannel};
use std::ops::AddAssign;

/// Computes the 2d running sum of an image. Channels are summed independently.
///
/// An integral image I has width and height one greater than its source image F,
/// and is defined by I(x, y) = sum of F(x', y') for x' < x, y' < y, i.e. each pixel
/// in the integral image contains the sum of the pixel intensities of all input pixels
/// that are strictly above it and strictly to its left. In particular, the left column
/// and top row of an integral image are all 0, and the value of the bottom right pixel of
/// an integral image is equal to the sum of all pixels in the source image.
///
/// Integral images have the helpful property of allowing us to
/// compute the sum of pixel intensities in a rectangular region of an image
/// in constant time. Specifically, given a rectangle [l, r] * [t, b] in F,
/// the sum of the pixels in this rectangle is
/// I(r + 1, b + 1) - I(r + 1, t) - I(l, b + 1) + I(l, t).
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::integral_image::{integral_image, sum_image_pixels};
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let integral = gray_image!(type: u32,
///     0,  0,  0,  0;
///     0,  1,  3,  6;
///     0,  5, 12, 21);
///
/// assert_pixels_eq!(integral_image::<_, u32>(&image), integral);
///
/// // Compute the sum of all pixels in the right two columns
/// assert_eq!(sum_image_pixels(&integral, 1, 0, 2, 1)[0], 2 + 3 + 5 + 6);
///
/// // Compute the sum of all pixels in the top row
/// assert_eq!(sum_image_pixels(&integral, 0, 0, 2, 0)[0], 1 + 2 + 3);
/// # }
/// ```
pub fn integral_image<P, T>(image: &Image<P>) -> Image<ChannelMap<P, T>>
where
    P: Pixel<Subpixel = u8> + WithChannel<T> + 'static,
    T: From<u8> + Primitive + AddAssign + 'static
{
    integral_image_impl(image, false)
}

/// Computes the 2d running sum of the squares of the intensities in an image. Channels are summed
/// independently.
///
/// See the [`integral_image`](fn.integral_image.html) documentation for more information on integral images.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::integral_image::{integral_squared_image, sum_image_pixels};
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let integral = gray_image!(type: u32,
///     0,  0,  0,  0;
///     0,  1,  5, 14;
///     0, 17, 46, 91);
///
/// assert_pixels_eq!(integral_squared_image::<_, u32>(&image), integral);
///
/// // Compute the sum of the squares of all pixels in the right two columns
/// assert_eq!(sum_image_pixels(&integral, 1, 0, 2, 1)[0], 4 + 9 + 25 + 36);
///
/// // Compute the sum of the squares of all pixels in the top row
/// assert_eq!(sum_image_pixels(&integral, 0, 0, 2, 0)[0], 1 + 4 + 9);
/// # }
/// ```
pub fn integral_squared_image<P, T>(image: &Image<P>) -> Image<ChannelMap<P, T>>
where
    P: Pixel<Subpixel = u8> + WithChannel<T> + 'static,
    T: From<u8> + Primitive + AddAssign + 'static
{
    integral_image_impl(image, true)
}

/// Implementation of `integral_image` and `integral_squared_image`.
fn integral_image_impl<P, T>(image: &Image<P>, square: bool) -> Image<ChannelMap<P, T>>
where
    P: Pixel<Subpixel = u8> + WithChannel<T> + 'static,
    T: From<u8> + Primitive + AddAssign + 'static
{
    // TODO: Make faster, add a new IntegralImage type
    // TODO: to make it harder to make off-by-one errors when computing sums of regions.
    let (in_width, in_height) = image.dimensions();
    let out_width = in_width + 1;
    let out_height = in_height + 1;

    let mut out = Image::<ChannelMap<P, T>>::new(out_width, out_height);

    if in_width == 0 || in_height == 0 {
        return out;
    }

    for y in 1..out_height {
        let mut sum = vec![T::zero(); P::channel_count() as usize];
        for x in 1..out_width {
            unsafe {
                for c in 0..P::channel_count() {
                    let pix: T = (image.unsafe_get_pixel(x - 1, y - 1).channels()[c as usize]).into();
                    if square {
                        sum[c as usize] += pix * pix;
                    } else {
                        sum[c as usize] += pix;
                    }
                }

                let above = out.unsafe_get_pixel(x, y - 1);
                // For some reason there's no unsafe_get_pixel_mut, so to update the existing
                // pixel here we need to use the method with bounds checking
                let current = out.get_pixel_mut(x, y);
                for c in 0..P::channel_count() {
                    current.channels_mut()[c as usize] = above.channels()[c as usize] + sum[c as usize];
                }
            }
        }
    }

    out
}

/// Hack to get around lack of const generics. See comment on `sum_image_pixels`.
pub trait ArrayData {
    /// The type of the data for this array.
    /// e.g. `[T; 1]` for `Luma`, `[T; 3]` for `Rgb`.
    type DataType;

    /// Get the data from this pixel as a constant length array.
    fn data(&self) -> Self::DataType;

    /// Add the elements of two data arrays elementwise.
    fn add(lhs: Self::DataType, other: Self::DataType) -> Self::DataType;

    /// Subtract the elements of two data arrays elementwise.
    fn sub(lhs: Self::DataType, other: Self::DataType) -> Self::DataType;
}

impl<T: Primitive + 'static> ArrayData for Luma<T> {
    type DataType = [T; 1];

    fn data(&self) -> Self::DataType {
        [self.channels()[0]]
    }

    fn add(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] + rhs[0]]
    }

    fn sub(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] - rhs[0]]
    }
}

impl<T: Primitive + 'static> ArrayData for Rgb<T> {
    type DataType = [T; 3];

    fn data(&self) -> Self::DataType {
        [self.channels()[0], self.channels()[1], self.channels()[2]]
    }

    fn add(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
    }

    fn sub(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
    }
}

impl<T: Primitive + 'static> ArrayData for Rgba<T> {
    type DataType = [T; 4];

    fn data(&self) -> Self::DataType {
        [self.channels()[0], self.channels()[1], self.channels()[2], self.channels()[3]]
    }

    fn add(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]]
    }

    fn sub(lhs: Self::DataType, rhs: Self::DataType) -> Self::DataType {
        [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]]
    }
}

/// Sums the pixels in positions [left, right] * [top, bottom] in F, where `integral_image` is the
/// integral image of F.
///
/// The of `ArrayData` here is due to lack of const generics. This library contains
/// implementations of `ArrayData` for `Luma`, `Rgb` and `Rgba` for any element type `T` that
/// implements `Primitive`. In that case, this function returns `[T; 1]` for an image
/// whose pixels are of type `Luma`, `[T; 3]` for `Rgb` pixels and `[T; 4]` for `Rgba` pixels.
///
/// See the [`integral_image`](fn.integral_image.html) documentation for examples.
pub fn sum_image_pixels<P>(
    integral_image: &Image<P>,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> P::DataType
where
    P: Pixel + ArrayData + Copy + 'static
{
    // TODO: better type-safety. It's too easy to pass the original image in here by mistake.
    // TODO: it's also hard to see what the four u32s mean at the call site - use a Rect instead.
    let (a, b, c, d) =
    (
        integral_image.get_pixel(right + 1, bottom + 1).data(),
        integral_image.get_pixel(left, top).data(),
        integral_image.get_pixel(right + 1, top).data(),
        integral_image.get_pixel(left, bottom + 1).data()
    );
    P::sub(P::sub(P::add(a, b), c), d)
}

/// Computes the variance of [left, right] * [top, bottom] in F, where `integral_image` is the
/// integral image of F and `integral_squared_image` is the integral image of the squares of the
/// pixels in F.
///
/// See the [`integral_image`](fn.integral_image.html) documentation for more information on integral images.
///
///# Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use std::f64;
/// use imageproc::integral_image::{integral_image, integral_squared_image, variance};
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// let integral = integral_image(&image);
/// let integral_squared = integral_squared_image(&image);
///
/// // Compute the variance of the pixels in the right two columns
/// let mean: f64 = (2.0 + 3.0 + 5.0 + 6.0) / 4.0;
/// let var = ((2.0 - mean).powi(2)
///     + (3.0 - mean).powi(2)
///     + (5.0 - mean).powi(2)
///     + (6.0 - mean).powi(2)) / 4.0;
///
/// assert_eq!(variance(&integral, &integral_squared, 1, 0, 2, 1), var);
/// # }
/// ```
pub fn variance(
    integral_image: &Image<Luma<u32>>,
    integral_squared_image: &Image<Luma<u32>>,
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
) -> f64 {
    // TODO: same improvements as for sum_image_pixels, plus check that the given rect is valid.
    let n = (right - left + 1) as f64 * (bottom - top + 1) as f64;
    let sum_sq = sum_image_pixels(integral_squared_image, left, top, right, bottom)[0];
    let sum = sum_image_pixels(integral_image, left, top, right, bottom)[0];
    (sum_sq as f64 - (sum as f64).powi(2) / n) / n
}

/// Computes the running sum of one row of image, padded
/// at the beginning and end. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all rows in an image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::integral_image::row_running_sum;
///
/// let image = gray_image!(
///     1, 2, 3;
///     4, 5, 6);
///
/// // Buffer has length two greater than image width, hence padding of 1
/// let mut buffer = [0; 5];
/// row_running_sum(&image, 0, &mut buffer, 1);
///
/// // The image is padded by continuity on either side
/// assert_eq!(buffer, [1, 2, 4, 7, 10]);
/// # }
/// ```
pub fn row_running_sum(image: &GrayImage, row: u32, buffer: &mut [u32], padding: u32) {
    // TODO: faster, more formats
    let (width, height) = image.dimensions();
    assert!(
        buffer.len() >= (width + 2 * padding) as usize,
        format!(
            "Buffer length {} is less than {} + 2 * {}",
            buffer.len(),
            width,
            padding
        )
    );
    assert!(
        row < height,
        format!("row out of bounds: {} >= {}", row, height)
    );

    unsafe {
        let mut sum = 0;
        for x in 0..padding {
            sum += image.unsafe_get_pixel(0, row)[0] as u32;
            *buffer.get_unchecked_mut(x as usize) = sum;
        }

        for x in 0..width {
            sum += image.unsafe_get_pixel(x, row)[0] as u32;
            *buffer.get_unchecked_mut((x + padding) as usize) = sum;
        }

        for x in 0..padding {
            sum += image.unsafe_get_pixel(width - 1, row)[0] as u32;
            *buffer.get_unchecked_mut((x + width + padding) as usize) = sum;
        }
    }
}

/// Computes the running sum of one column of image, padded
/// at the top and bottom. The padding is by continuity.
/// Takes a reference to buffer so that this can be reused
/// for all columns in an image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::integral_image::column_running_sum;
///
/// let image = gray_image!(
///     1, 4;
///     2, 5;
///     3, 6);
///
/// // Buffer has length two greater than image height, hence padding of 1
/// let mut buffer = [0; 5];
/// column_running_sum(&image, 0, &mut buffer, 1);
///
/// // The image is padded by continuity on top and bottom
/// assert_eq!(buffer, [1, 2, 4, 7, 10]);
/// # }
/// ```
pub fn column_running_sum(image: &GrayImage, column: u32, buffer: &mut [u32], padding: u32) {
    // TODO: faster, more formats
    let (width, height) = image.dimensions();
    assert!(
        buffer.len() >= (height + 2 * padding) as usize,
        format!(
            "Buffer length {} is less than {} + 2 * {}",
            buffer.len(),
            height,
            padding
        )
    );
    assert!(
        column < width,
        format!("column out of bounds: {} >= {}", column, width)
    );

    unsafe {
        let mut sum = 0;
        for y in 0..padding {
            sum += image.unsafe_get_pixel(column, 0)[0] as u32;
            *buffer.get_unchecked_mut(y as usize) = sum;
        }

        for y in 0..height {
            sum += image.unsafe_get_pixel(column, y)[0] as u32;
            *buffer.get_unchecked_mut((y + padding) as usize) = sum;
        }

        for y in 0..padding {
            sum += image.unsafe_get_pixel(column, height - 1)[0] as u32;
            *buffer.get_unchecked_mut((y + height + padding) as usize) = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property_testing::GrayTestImage;
    use crate::utils::{gray_bench_image, pixel_diff_summary, rgb_bench_image};
    use image::{GenericImage, ImageBuffer, Luma};
    use quickcheck::{quickcheck, TestResult};
    use crate::definitions::Image;
    use ::test;

    #[test]
    fn test_integral_image_gray() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6);

        let expected = gray_image!(type: u32,
            0,  0,  0,  0;
            0,  1,  3,  6;
            0,  5, 12, 21);

        assert_pixels_eq!(integral_image::<_, u32>(&image), expected);
    }

    #[test]
    fn test_integral_image_rgb() {
        let image = rgb_image!(
            [1, 11, 21], [2, 12, 22], [3, 13, 23];
            [4, 14, 24], [5, 15, 25], [6, 16, 26]);

        let expected = rgb_image!(type: u32,
            [0, 0, 0],  [0,  0,  0], [ 0,  0,  0], [ 0,  0,   0];
            [0, 0, 0],  [1, 11, 21], [ 3, 23, 43], [ 6, 36,  66];
            [0, 0, 0],  [5, 25, 45], [12, 52, 92], [21, 81, 141]);

        assert_pixels_eq!(integral_image::<_, u32>(&image), expected);
    }

    #[test]
    fn test_sum_image_pixels() {
        let image = gray_image!(
            1, 2;
            3, 4);

        let integral = integral_image::<_, u32>(&image);

        // Top left
        assert_eq!(sum_image_pixels(&integral, 0, 0, 0, 0)[0], 1);
        // Top row
        assert_eq!(sum_image_pixels(&integral, 0, 0, 1, 0)[0], 3);
        // Left column
        assert_eq!(sum_image_pixels(&integral, 0, 0, 0, 1)[0], 4);
        // Whole image
        assert_eq!(sum_image_pixels(&integral, 0, 0, 1, 1)[0], 10);
        // Top right
        assert_eq!(sum_image_pixels(&integral, 1, 0, 1, 0)[0], 2);
        // Right column
        assert_eq!(sum_image_pixels(&integral, 1, 0, 1, 1)[0], 6);
        // Bottom left
        assert_eq!(sum_image_pixels(&integral, 0, 1, 0, 1)[0], 3);
        // Bottom row
        assert_eq!(sum_image_pixels(&integral, 0, 1, 1, 1)[0], 7);
        // Bottom right
        assert_eq!(sum_image_pixels(&integral, 1, 1, 1, 1)[0], 4);
    }

    #[test]
    fn test_sum_image_pixels_rgb() {
        let image = rgb_image!(
            [1,  2,  3], [ 4,  5,  6];
            [7,  8,  9], [10, 11, 12]);

        let integral = integral_image::<_, u32>(&image);

        // Top left
        assert_eq!(
            sum_image_pixels(&integral, 0, 0, 0, 0),
            [1, 2, 3]);
        // Top row
        assert_eq!(
            sum_image_pixels(&integral, 0, 0, 1, 0),
            [5, 7, 9]);
        // Left column
        assert_eq!(
            sum_image_pixels(&integral, 0, 0, 0, 1),
            [8, 10, 12]);
        // Whole image
        assert_eq!(
            sum_image_pixels(&integral, 0, 0, 1, 1),
            [22, 26, 30]);
        // Top right
        assert_eq!(
            sum_image_pixels(&integral, 1, 0, 1, 0),
            [4, 5, 6]);
        // Right column
        assert_eq!(
            sum_image_pixels(&integral, 1, 0, 1, 1),
            [14, 16, 18]);
        // Bottom left
        assert_eq!(
            sum_image_pixels(&integral, 0, 1, 0, 1),
            [7, 8, 9]);
        // Bottom row
        assert_eq!(
            sum_image_pixels(&integral, 0, 1, 1, 1),
            [17, 19, 21]);
        // Bottom right
        assert_eq!(
            sum_image_pixels(&integral, 1, 1, 1, 1),
            [10, 11, 12]);
    }

    #[bench]
    fn bench_integral_image_gray(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let integral = integral_image::<_, u32>(&image);
            test::black_box(integral);
        });
    }

    #[bench]
    fn bench_integral_image_rgb(b: &mut test::Bencher) {
        let image = rgb_bench_image(500, 500);
        b.iter(|| {
            let integral = integral_image::<_, u32>(&image);
            test::black_box(integral);
        });
    }

    /// Simple implementation of integral_image to validate faster versions against.
    fn integral_image_ref<I>(image: &I) -> Image<Luma<u32>>
    where
        I: GenericImage<Pixel = Luma<u8>>,
    {
        let (in_width, in_height) = image.dimensions();
        let (out_width, out_height) = (in_width + 1, in_height + 1);
        let mut out = ImageBuffer::from_pixel(out_width, out_height, Luma([0u32]));

        for y in 1..out_height {
            for x in 0..out_width {
                let mut sum = 0u32;

                for iy in 0..y {
                    for ix in 0..x {
                        sum += image.get_pixel(ix, iy)[0] as u32;
                    }
                }

                out.put_pixel(x, y, Luma([sum]));
            }
        }

        out
    }

    #[test]
    fn test_integral_image_matches_reference_implementation() {
        fn prop(image: GrayTestImage) -> TestResult {
            let expected = integral_image_ref(&image.0);
            let actual = integral_image(&image.0);
            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err),
            }
        }
        quickcheck(prop as fn(GrayTestImage) -> TestResult);
    }

    #[bench]
    fn bench_row_running_sum(b: &mut test::Bencher) {
        let image = gray_bench_image(1000, 1);
        let mut buffer = [0; 1010];
        b.iter(|| { row_running_sum(&image, 0, &mut buffer, 5); });
    }

    #[bench]
    fn bench_column_running_sum(b: &mut test::Bencher) {
        let image = gray_bench_image(100, 1000);
        let mut buffer = [0; 1010];
        b.iter(|| { column_running_sum(&image, 0, &mut buffer, 5); });
    }
}
