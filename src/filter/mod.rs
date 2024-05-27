//! Functions for filtering images.

pub mod bilateral;
mod median;
pub use self::bilateral::bilateral_filter;
pub use self::median::median_filter;

mod sharpen;
pub use self::sharpen::*;

use image::{GenericImage, GenericImageView, GrayImage, Luma, Pixel, Primitive};
use itertools::Itertools;

use crate::definitions::{Clamp, Image};
use crate::integral_image::{column_running_sum, row_running_sum};
use crate::kernel::{self, Kernel};
use crate::map::{ChannelMap, WithChannel};
use num::Num;

use std::cmp::{max, min};
use std::f32;

/// Convolves an 8bpp grayscale image with a kernel of width (2 * `x_radius` + 1)
/// and height (2 * `y_radius` + 1) whose entries are equal and
/// sum to one. i.e. each output pixel is the unweighted mean of
/// a rectangular region surrounding its corresponding input pixel.
/// We handle locations where the kernel would extend past the image's
/// boundary by treating the image as if its boundary pixels were
/// repeated indefinitely.
// TODO: for small kernels we probably want to do the convolution
// TODO: directly instead of using an integral image.
// TODO: more formats!
#[must_use = "the function does not modify the original image"]
pub fn box_filter(image: &GrayImage, x_radius: u32, y_radius: u32) -> Image<Luma<u8>> {
    let (width, height) = image.dimensions();
    let mut out = Image::new(width, height);
    if width == 0 || height == 0 {
        return out;
    }

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;

    let mut row_buffer = vec![0; (width + 2 * x_radius) as usize];
    for y in 0..height {
        row_running_sum(image, y, &mut row_buffer, x_radius);
        let val = row_buffer[(2 * x_radius) as usize] / kernel_width;
        unsafe {
            debug_assert!(out.in_bounds(0, y));
            out.unsafe_put_pixel(0, y, Luma([val as u8]));
        }
        for x in 1..width {
            // TODO: This way we pay rounding errors for each of the
            // TODO: x and y convolutions. Is there a better way?
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = (row_buffer[u] - row_buffer[l]) / kernel_width;
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Luma([val as u8]));
            }
        }
    }

    let mut col_buffer = vec![0; (height + 2 * y_radius) as usize];
    for x in 0..width {
        column_running_sum(&out, x, &mut col_buffer, y_radius);
        let val = col_buffer[(2 * y_radius) as usize] / kernel_height;
        unsafe {
            debug_assert!(out.in_bounds(x, 0));
            out.unsafe_put_pixel(x, 0, Luma([val as u8]));
        }
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = (col_buffer[u] - col_buffer[l]) / kernel_height;
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Luma([val as u8]));
            }
        }
    }

    out
}

/// Calculates the new pixel value for a particular pixel and kernel.
fn filter_pixel<P, K, F, Q>(x: u32, y: u32, kernel: Kernel<K>, f: F, image: &Image<P>) -> Q
where
    P: Pixel,
    Q: Pixel,
    F: Fn(K) -> Q::Subpixel,
    K: num::Num + Copy + From<P::Subpixel>,
{
    let (width, height) = (image.width() as i64, image.height() as i64);
    let (k_width, k_height) = (kernel.width as i64, kernel.height as i64);
    let (x, y) = (i64::from(x), i64::from(y));

    let weighted_pixels = (0..kernel.height as i64)
        .cartesian_product(0..kernel.width as i64)
        .map(|(k_y, k_x)| {
            let kernel_weight = *kernel.at(k_x as u32, k_y as u32);

            let window_y = (y + k_y - k_height / 2).clamp(0, height - 1);
            let window_x = (x + k_x - k_width / 2).clamp(0, width - 1);

            debug_assert!(image.in_bounds(window_x as u32, window_y as u32));

            // Safety: we clamped `window_x` and `window_y` to be in bounds.
            let window_pixel = unsafe { image.unsafe_get_pixel(window_x as u32, window_y as u32) };

            //optimisation: remove allocation when `Pixel::map` allows mapping to a different
            //type
            #[allow(clippy::unnecessary_to_owned)]
            let weighted_pixel = window_pixel
                .channels()
                .to_vec()
                .into_iter()
                .map(move |c| kernel_weight * K::from(c));

            weighted_pixel
        });

    let final_channel_sum = weighted_pixels.fold(
        //optimisation: do this without allocation when `Pixel` gains a method of constant initialization
        vec![K::zero(); Q::CHANNEL_COUNT as usize],
        |mut accumulator, weighted_pixel| {
            for (i, weighted_subpixel) in weighted_pixel.enumerate() {
                accumulator[i] = accumulator[i] + weighted_subpixel;
            }

            accumulator
        },
    );

    *Q::from_slice(&final_channel_sum.into_iter().map(f).collect_vec())
}

/// Returns 2d correlation of an image. Intermediate calculations are performed
/// at type K, and the results converted to pixel Q via f. Pads by continuity.
pub fn filter<P, K, F, Q>(image: &Image<P>, kernel: Kernel<K>, f: F) -> Image<Q>
where
    P: Pixel,
    Q: Pixel,
    F: Fn(K) -> Q::Subpixel,
    K: num::Num + Copy + From<P::Subpixel>,
{
    //TODO refactor this to use the `crate::maps` functions once that lands

    let (width, height) = image.dimensions();

    let mut out = Image::<Q>::new(width, height);

    for y in 0..height {
        for x in 0..width {
            out.put_pixel(x, y, filter_pixel(x, y, kernel, &f, image));
        }
    }

    out
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("filter")]
pub fn filter_parallel<P, K, F, Q>(image: &Image<P>, kernel: Kernel<K>, f: F) -> Image<Q>
where
    P: Pixel + Sync,
    Q: Pixel + Send + Sync,
    P::Subpixel: Sync,
    Q::Subpixel: Send + Sync,
    F: Fn(K) -> Q::Subpixel + Send + Sync,
    K: Num + Copy + From<P::Subpixel> + Sync,
{
    //TODO refactor this to use the `crate::maps` functions once that lands

    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::ParallelIterator;

    let (width, height) = image.dimensions();

    let mut out: Image<Q> = Image::new(width, height);

    image
        .par_enumerate_pixels()
        .zip_eq(out.par_pixels_mut())
        .for_each(move |((x, y, _), output_pixel)| {
            *output_pixel = filter_pixel(x, y, kernel, &f, image);
        });

    out
}

#[inline]
fn gaussian_pdf(x: f32, sigma: f32) -> f32 {
    // Equation derived from <https://en.wikipedia.org/wiki/Gaussian_function>
    (sigma * (2.0 * f32::consts::PI).sqrt()).recip() * (-x.powi(2) / (2.0 * sigma.powi(2))).exp()
}

/// Construct a one dimensional float-valued kernel for performing a Gaussian blur
/// with standard deviation sigma.
fn gaussian_kernel_f32(sigma: f32) -> Vec<f32> {
    let kernel_radius = (2.0 * sigma).ceil() as usize;
    let mut kernel_data = vec![0.0; 2 * kernel_radius + 1];
    for i in 0..kernel_radius + 1 {
        let value = gaussian_pdf(i as f32, sigma);
        kernel_data[kernel_radius + i] = value;
        kernel_data[kernel_radius - i] = value;
    }
    let sum: f32 = kernel_data.iter().sum();
    kernel_data.iter_mut().for_each(|x| *x /= sum);

    kernel_data
}

/// Blurs an image using a Gaussian of standard deviation sigma.
/// The kernel used has type f32 and all intermediate calculations are performed
/// at this type.
///
/// # Panics
///
/// Panics if `sigma <= 0.0`.
// TODO: Integer type kernel, approximations via repeated box filter.
#[must_use = "the function does not modify the original image"]
pub fn gaussian_blur_f32<P>(image: &Image<P>, sigma: f32) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    assert!(sigma > 0.0, "sigma must be > 0.0");
    let kernel = gaussian_kernel_f32(sigma);
    separable_filter_equal(image, &kernel)
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels `h_kernel` and `v_kernel`.
#[must_use = "the function does not modify the original image"]
pub fn separable_filter<P, K>(image: &Image<P>, h_kernel: &[K], v_kernel: &[K]) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K> + Clamp<K>,
    K: Num + Copy,
{
    assert_eq!(
        h_kernel.len(),
        v_kernel.len(),
        "the two 1D kernels must be the same length"
    );

    let h = horizontal_filter(image, h_kernel);
    vertical_filter(&h, v_kernel)
}

/// Returns 2d correlation of an image with the outer product of the 1d
/// kernel filter with itself.
#[must_use = "the function does not modify the original image"]
pub fn separable_filter_equal<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K> + Clamp<K>,
    K: Num + Copy,
{
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of an image with a row-major kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
///
/// A parallelized version of this function exists with [`filter_clamped_parallel`] when
/// the crate `rayon` feature is enabled.
pub fn filter_clamped<P, K, S>(image: &Image<P>, kernel: Kernel<K>) -> Image<ChannelMap<P, S>>
where
    P::Subpixel: Into<K>,
    S: Clamp<K> + Primitive,
    P: WithChannel<S>,
    K: Num + Copy + From<<P as image::Pixel>::Subpixel>,
{
    filter(image, kernel, S::clamp)
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("filter_clamped")]
pub fn filter_clamped_parallel<P, K, S>(
    image: &Image<P>,
    kernel: Kernel<K>,
) -> Image<ChannelMap<P, S>>
where
    P: Sync,
    P::Subpixel: Send + Sync,
    <P as WithChannel<S>>::Pixel: Send + Sync,
    S: Clamp<K> + Primitive + Send + Sync,
    P: WithChannel<S>,
    K: Num + Copy + Send + Sync + From<P::Subpixel>,
{
    filter_parallel(image, kernel, S::clamp)
}

/// Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
#[must_use = "the function does not modify the original image"]
pub fn horizontal_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to filter() without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::CHANNEL_COUNT as usize];
    let k_width = kernel.len() as i32;
    let half_k = k_width / 2;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_width >= width as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let x_unchecked = (x as i32) + i as i32 - half_k;
                    let x_p = max(0, min(x_unchecked, width as i32 - 1)) as u32;
                    debug_assert!(image.in_bounds(x_p, y));
                    let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
            }
        }

        return out;
    }

    for y in 0..height {
        // Left margin - need to check lower bound only
        for x in 0..half_k {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = x + i as i32 - half_k;
                let x_p = max(0, x_unchecked) as u32;
                debug_assert!(image.in_bounds(x_p, y));
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }

        // Neither margin - don't need bounds check on either side
        for x in half_k..(width as i32 - half_k) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = x + i as i32 - half_k;
                let x_p = x_unchecked as u32;
                debug_assert!(image.in_bounds(x_p, y));
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }

        // Right margin - need to check upper bound only
        for x in (width as i32 - half_k)..(width as i32) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = x + i as i32 - half_k;
                let x_p = min(x_unchecked, width as i32 - 1) as u32;
                debug_assert!(image.in_bounds(x_p, y));
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }
    }

    out
}

/// Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
#[must_use = "the function does not modify the original image"]
pub fn vertical_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to filter() without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::CHANNEL_COUNT as usize];
    let k_height = kernel.len() as i32;
    let half_k = k_height / 2;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_height >= height as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let y_unchecked = (y as i32) + i as i32 - half_k;
                    let y_p = max(0, min(y_unchecked, height as i32 - 1)) as u32;
                    debug_assert!(image.in_bounds(x, y_p));
                    let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
            }
        }

        return out;
    }

    // Top margin - need to check lower bound only
    for y in 0..half_k {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = y + i as i32 - half_k;
                let y_p = max(0, y_unchecked) as u32;
                debug_assert!(image.in_bounds(x, y_p));
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }
    }

    // Neither margin - don't need bounds check on either side
    for y in half_k..(height as i32 - half_k) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = y + i as i32 - half_k;
                let y_p = y_unchecked as u32;
                debug_assert!(image.in_bounds(x, y_p));
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }
    }

    // Right margin - need to check upper bound only
    for y in (height as i32 - half_k)..(height as i32) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = y + i as i32 - half_k;
                let y_p = min(y_unchecked, height as i32 - 1) as u32;
                debug_assert!(image.in_bounds(x, y_p));
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            clamp_and_reset::<P, K>(&mut acc, out_channels, zero);
        }
    }

    out
}

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K)
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K>,
    K: Num + Copy,
{
    for i in 0..(P::CHANNEL_COUNT as usize) {
        acc[i] = acc[i] + pixel.channels()[i].into() * weight;
    }
}

fn clamp_and_reset<P, K>(acc: &mut [K], out_channels: &mut [P::Subpixel], zero: K)
where
    P: Pixel,
    <P as Pixel>::Subpixel: Into<K> + Clamp<K>,
    K: Num + Copy,
{
    for (accumulator, pixel_channel) in acc.iter_mut().zip(out_channels.iter_mut()) {
        let clamped_pixel = P::Subpixel::clamp(*accumulator);
        *pixel_channel = clamped_pixel;
        *accumulator = zero;
    }
}

/// Calculates the Laplacian of an image.
///
/// The Laplacian is computed by filtering the image using the following 3x3 kernel:
/// ```notrust
/// 0, 1, 0,
/// 1,-4, 1,
/// 0, 1, 0
/// ```
#[must_use = "the function does not modify the original image"]
pub fn laplacian_filter(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped(image, kernel::FOUR_LAPLACIAN_3X3)
}
#[must_use = "the function does not modify the original image"]
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("laplacian_filter")]
pub fn laplacian_filter_parallel(image: &GrayImage) -> Image<Luma<i16>> {
    filter_clamped_parallel(image, kernel::FOUR_LAPLACIAN_3X3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::{Clamp, Image};
    use crate::utils::gray_bench_image;
    use image::{GrayImage, Luma};
    use std::cmp::{max, min};
    use test::black_box;

    #[test]
    fn test_laplacian() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9);

        #[rustfmt::skip]
        let laplacian = Kernel::new(&[
            0, 1, 0,
            1, -4, 1,
            0, 1, 0
        ], 3, 3);

        let actual = laplacian_filter(&image);
        let expected = filter_clamped(&image, laplacian);
        assert_eq!(actual.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_box_filter_handles_empty_images() {
        let _ = box_filter(&GrayImage::new(0, 0), 3, 3);
        let _ = box_filter(&GrayImage::new(1, 0), 3, 3);
        let _ = box_filter(&GrayImage::new(0, 1), 3, 3);
    }

    #[test]
    fn test_box_filter() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9);

        // For this image we get the same answer from the two 1d
        // convolutions as from doing the 2d convolution in one step
        // (but we needn't in general, as in the former case we're
        // clipping to an integer value twice).
        let expected = gray_image!(
            2, 3, 3;
            4, 5, 5;
            6, 7, 7);

        assert_pixels_eq!(box_filter(&image, 1, 1), expected);
    }
    #[test]
    fn test_separable_filter() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9);

        // Lazily copying the box_filter test case
        let expected = gray_image!(
            2, 3, 3;
            4, 5, 5;
            6, 7, 7);

        let kernel = [1f32 / 3f32; 3];
        let filtered = separable_filter_equal(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_separable_filter_integer_kernel() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9);

        let expected = gray_image!(
            21, 27, 33;
            39, 45, 51;
            57, 63, 69);

        let kernel = [1i32; 3];
        let filtered = separable_filter_equal(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    /// Reference implementation of horizontal_filter. Used to validate
    /// the (presumably faster) actual implementation.
    fn horizontal_filter_reference(image: &GrayImage, kernel: &[f32]) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut out = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut acc = 0f32;

                for (i, k) in kernel.iter().enumerate() {
                    let mut x_unchecked = x as i32 + i as i32 - (kernel.len() / 2) as i32;
                    x_unchecked = max(0, x_unchecked);
                    x_unchecked = min(x_unchecked, width as i32 - 1);

                    let x_checked = x_unchecked as u32;
                    let color = image.get_pixel(x_checked, y)[0];
                    let weight = k;

                    acc += color as f32 * weight;
                }

                let clamped = <u8 as Clamp<f32>>::clamp(acc);
                out.put_pixel(x, y, Luma([clamped]));
            }
        }

        out
    }

    /// Reference implementation of vertical_filter. Used to validate
    /// the (presumably faster) actual implementation.
    fn vertical_filter_reference(image: &GrayImage, kernel: &[f32]) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut out = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut acc = 0f32;

                for (i, k) in kernel.iter().enumerate() {
                    let mut y_unchecked = y as i32 + i as i32 - (kernel.len() / 2) as i32;
                    y_unchecked = max(0, y_unchecked);
                    y_unchecked = min(y_unchecked, height as i32 - 1);

                    let y_checked = y_unchecked as u32;
                    let color = image.get_pixel(x, y_checked)[0];
                    let weight = k;

                    acc += color as f32 * weight;
                }

                let clamped = <u8 as Clamp<f32>>::clamp(acc);
                out.put_pixel(x, y, Luma([clamped]));
            }
        }

        out
    }

    macro_rules! test_against_reference_implementation {
        ($test_name:ident, $under_test:ident, $reference_impl:ident) => {
            #[test]
            fn $test_name() {
                // I think the interesting edge cases here are determined entirely
                // by the relative sizes of the kernel and the image side length, so
                // I'm just enumerating over small values instead of generating random
                // examples via quickcheck.
                for height in 0..5 {
                    for width in 0..5 {
                        for kernel_length in 0..15 {
                            let image = gray_bench_image(width, height);
                            let kernel: Vec<f32> =
                                (0..kernel_length).map(|i| i as f32 % 1.35).collect();

                            let expected = $reference_impl(&image, &kernel);
                            let actual = $under_test(&image, &kernel);

                            assert_pixels_eq!(actual, expected);
                        }
                    }
                }
            }
        };
    }

    test_against_reference_implementation!(
        test_horizontal_filter_matches_reference_implementation,
        horizontal_filter,
        horizontal_filter_reference
    );

    test_against_reference_implementation!(
        test_vertical_filter_matches_reference_implementation,
        vertical_filter,
        vertical_filter_reference
    );

    #[test]
    fn test_horizontal_filter() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let expected = gray_image!(
            2, 2, 2;
            5, 5, 5;
            2, 2, 2);

        let kernel = [1f32 / 3f32; 3];
        let filtered = horizontal_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_filter_with_kernel_wider_than_image_does_not_panic() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let kernel = [1f32 / 10f32; 10];
        black_box(horizontal_filter(&image, &kernel));
    }
    #[test]
    fn test_vertical_filter() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let expected = gray_image!(
            2, 5, 2;
            2, 5, 2;
            2, 5, 2);

        let kernel = [1f32 / 3f32; 3];
        let filtered = vertical_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_filter_with_kernel_taller_than_image_does_not_panic() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let kernel = [1f32 / 10f32; 10];
        black_box(vertical_filter(&image, &kernel));
    }
    #[test]
    fn test_filter_clamped_with_results_outside_input_channel_range() {
        #[rustfmt::skip]
        let kernel: Kernel<i32> = Kernel::new(&[
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1
        ], 3, 3);

        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -4, -8, -4;
            -4, -8, -4;
            -4, -8, -4
        );

        let filtered = filter_clamped(&image, kernel);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_filter_clamped_parallel_with_results_outside_input_channel_range() {
        #[rustfmt::skip]
            let kernel: Kernel<i32> = Kernel::new(&[
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1
        ], 3, 3);

        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image!(type: i16,
            -4, -8, -4;
            -4, -8, -4;
            -4, -8, -4
        );

        let filtered = filter_clamped_parallel(&image, kernel);
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    #[should_panic]
    fn test_kernel_must_be_nonempty() {
        let k: &[u8] = &[];
        let _ = Kernel::new(k, 0, 0);
    }

    #[test]
    fn test_kernel_filter_with_even_kernel_side() {
        let image = gray_image!(
            3, 2;
            4, 1);

        let k = &[1u8, 2u8];
        let kernel = Kernel::new(k, 2, 1);
        let filtered = filter(&image, kernel, |x| x);

        let expected = gray_image!(
             9,  7;
            12,  6);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_kernel_filter_with_empty_image() {
        let image = gray_image!();

        let k = &[2u8];
        let kernel = Kernel::new(k, 1, 1);
        let filtered = filter(&image, kernel, |x| x);

        let expected = gray_image!();
        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_kernel_filter_with_kernel_dimensions_larger_than_image() {
        let image = gray_image!(
            9, 4;
            8, 1);

        #[rustfmt::skip]
        let k: &[f32] = &[
            0.1, 0.2, 0.1,
            0.2, 0.4, 0.2,
            0.1, 0.2, 0.1
        ];
        let kernel = Kernel::new(k, 3, 3);
        let filtered: Image<Luma<u8>> = filter(&image, kernel, <u8 as Clamp<f32>>::clamp);

        let expected = gray_image!(
            11,  7;
            10,  5);

        assert_pixels_eq!(filtered, expected);
    }
    #[test]
    #[should_panic]
    fn test_gaussian_blur_f32_rejects_zero_sigma() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        );
        let _ = gaussian_blur_f32(&image, 0.0);
    }

    #[test]
    #[should_panic]
    fn test_gaussian_blur_f32_rejects_negative_sigma() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9
        );
        let _ = gaussian_blur_f32(&image, -0.5);
    }

    #[test]
    fn test_gaussian_on_u8_white_idempotent() {
        let image = Image::<Luma<u8>>::from_pixel(12, 12, Luma([255]));
        let image2 = gaussian_blur_f32(&image, 6f32);
        assert_pixels_eq_within!(image2, image, 0);
    }

    #[test]
    fn test_gaussian_on_f32_white_idempotent() {
        let image = Image::<Luma<f32>>::from_pixel(12, 12, Luma([1.0]));
        let image2 = gaussian_blur_f32(&image, 6f32);
        assert_pixels_eq_within!(image2, image, 1e-6);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::proptest_utils::arbitrary_image;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_box_filter(
            img in arbitrary_image::<Luma<u8>>(0..200, 0..200),
            x_radius in 0..100u32,
            y_radius in 0..100u32,
        ) {
            let out = box_filter(&img, x_radius, y_radius);
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_gaussian_blur_f32(
            img in arbitrary_image::<Luma<u8>>(0..20, 0..20),
            sigma in (0.0..150f32).prop_filter("contract", |&x| x > 0.0),
        ) {
            let out = gaussian_blur_f32(&img, sigma);
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_kernel_luma_f32(
            img in arbitrary_image::<Luma<f32>>(0..30, 0..30),
            ker in arbitrary_image::<Luma<f32>>(1..20, 1..20),
        ) {
            let kernel = Kernel::new(&ker, ker.width(), ker.height());
            let out: Image<Luma<f32>> = filter(&img, kernel, |x| {
                x
            });
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_horizontal_filter_luma_f32(
            img in arbitrary_image::<Luma<f32>>(0..50, 0..50),
            ker in proptest::collection::vec(any::<f32>(), 0..50),
        ) {
            let out = horizontal_filter(&img, &ker);
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_vertical_filter_luma_f32(
            img in arbitrary_image::<Luma<f32>>(0..50, 0..50),
            ker in proptest::collection::vec(any::<f32>(), 0..50),
        ) {
            let out = vertical_filter(&img, &ker);
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_laplacian_filter(
            img in arbitrary_image::<Luma<u8>>(0..120, 0..120),
        ) {
            let out = laplacian_filter(&img);
            assert_eq!(out.dimensions(), img.dimensions());
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::definitions::Image;
    use crate::utils::{gray_bench_image, rgb_bench_image};
    use image::imageops::blur;
    use image::{GenericImage, Luma, Rgb};
    use test::{black_box, Bencher};

    #[bench]
    fn bench_box_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let filtered = box_filter(&image, 7, 7);
            black_box(filtered);
        });
    }

    #[bench]
    fn bench_separable_filter(b: &mut Bencher) {
        let image = gray_bench_image(300, 300);
        let h_kernel = [1f32 / 5f32; 5];
        let v_kernel = [0.1f32, 0.4f32, 0.3f32, 0.1f32, 0.1f32];
        b.iter(|| {
            let filtered = separable_filter(&image, &h_kernel, &v_kernel);
            black_box(filtered);
        });
    }

    #[bench]
    fn bench_horizontal_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = [1f32 / 5f32; 5];
        b.iter(|| {
            let filtered = horizontal_filter(&image, &kernel);
            black_box(filtered);
        });
    }

    #[bench]
    fn bench_vertical_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = [1f32 / 5f32; 5];
        b.iter(|| {
            let filtered = vertical_filter(&image, &kernel);
            black_box(filtered);
        });
    }

    macro_rules! bench_filter_clamped_gray {
        ($name:ident, $kernel_size:expr, $image_size:expr, $function_name:ident) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image =
                    GrayImage::from_fn($image_size, $image_size, |x, y| Luma([(x + y % 3) as u8]));

                let kernel: Vec<i32> = (0..$kernel_size * $kernel_size).collect();
                let kernel = Kernel::new(&kernel, $kernel_size, $kernel_size);
                b.iter(|| {
                    let filtered: Image<Luma<i16>> = $function_name::<_, _, i16>(&image, kernel);
                    black_box(filtered);
                });
            }
        };
    }
    macro_rules! bench_filter_clamped_rgb {
        ($name:ident, $kernel_size:expr, $image_size:expr, $function_name:ident) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = rgb_bench_image(500, 500);

                let kernel: Vec<i32> = (0..$kernel_size * $kernel_size).collect();
                let kernel = Kernel::new(&kernel, $kernel_size, $kernel_size);
                b.iter(|| {
                    let filtered: Image<Rgb<i16>> = $function_name::<_, _, i16>(&image, kernel);
                    black_box(filtered);
                });
            }
        };
    }

    bench_filter_clamped_gray!(bench_filter_clamped_gray_3x3, 3, 300, filter_clamped);
    bench_filter_clamped_gray!(bench_filter_clamped_gray_5x5, 5, 300, filter_clamped);
    bench_filter_clamped_gray!(bench_filter_clamped_gray_7x7, 7, 300, filter_clamped);
    bench_filter_clamped_rgb!(bench_filter_clamped_rgb_3x3, 3, 300, filter_clamped);
    bench_filter_clamped_rgb!(bench_filter_clamped_rgb_5x5, 5, 300, filter_clamped);
    bench_filter_clamped_rgb!(bench_filter_clamped_rgb_7x7, 7, 300, filter_clamped);

    bench_filter_clamped_gray!(
        bench_filter_clamped_parallel_gray_3x3,
        3,
        300,
        filter_clamped_parallel
    );
    bench_filter_clamped_gray!(
        bench_filter_clamped_parallel_gray_5x5,
        5,
        300,
        filter_clamped_parallel
    );
    bench_filter_clamped_gray!(
        bench_filter_clamped_parallel_gray_7x7,
        7,
        300,
        filter_clamped_parallel
    );
    bench_filter_clamped_rgb!(
        bench_filter_clamped_parallel_rgb_3x3,
        3,
        300,
        filter_clamped_parallel
    );
    bench_filter_clamped_rgb!(
        bench_filter_clamped_parallel_rgb_5x5,
        5,
        300,
        filter_clamped_parallel
    );
    bench_filter_clamped_rgb!(
        bench_filter_clamped_parallel_rgb_7x7,
        7,
        300,
        filter_clamped_parallel
    );

    /// Baseline implementation of Gaussian blur is that provided by image::imageops.
    /// We can also use this to validate correctness of any implementations we add here.
    fn gaussian_baseline_rgb<I>(image: &I, stdev: f32) -> Image<Rgb<u8>>
    where
        I: GenericImage<Pixel = Rgb<u8>>,
    {
        blur(image, stdev)
    }

    #[bench]
    #[ignore] // Gives a baseline performance using code from another library
    fn bench_baseline_gaussian_stdev_1(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_baseline_rgb(&image, 1f32);
            black_box(blurred);
        });
    }

    #[bench]
    #[ignore] // Gives a baseline performance using code from another library
    fn bench_baseline_gaussian_stdev_3(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_baseline_rgb(&image, 3f32);
            black_box(blurred);
        });
    }

    #[bench]
    #[ignore] // Gives a baseline performance using code from another library
    fn bench_baseline_gaussian_stdev_10(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_baseline_rgb(&image, 10f32);
            black_box(blurred);
        });
    }

    #[bench]
    fn bench_gaussian_f32_stdev_1(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_blur_f32(&image, 1f32);
            black_box(blurred);
        });
    }

    #[bench]
    fn bench_gaussian_f32_stdev_3(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_blur_f32(&image, 3f32);
            black_box(blurred);
        });
    }

    #[bench]
    fn bench_gaussian_f32_stdev_10(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let blurred = gaussian_blur_f32(&image, 10f32);
            black_box(blurred);
        });
    }
}
