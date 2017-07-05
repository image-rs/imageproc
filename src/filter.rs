//! Functions for filtering images.

use image::{GrayImage, GenericImage, ImageBuffer, Luma, Pixel, Primitive};

use integral_image::{column_running_sum, row_running_sum};
use map::{WithChannel, ChannelMap};
use definitions::{Clamp, Image};
use num::Num;

use conv::ValueInto;
use math::cast;
use std::cmp::{min, max};
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
pub fn box_filter(image: &GrayImage, x_radius: u32, y_radius: u32) -> Image<Luma<u8>> {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;

    let mut row_buffer = vec![0; (width + 2 * x_radius) as usize];
    for y in 0..height {
        row_running_sum(image, y, &mut row_buffer, x_radius);
        let val = row_buffer[(2 * x_radius) as usize] / kernel_width;
        unsafe {
            out.unsafe_put_pixel(0, y, Luma([val as u8]));
        }
        for x in 1..width {
            // TODO: This way we pay rounding errors for each of the
            // TODO: x and y convolutions. Is there a better way?
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = (row_buffer[u] - row_buffer[l]) / kernel_width;
            unsafe {
                out.unsafe_put_pixel(x, y, Luma([val as u8]));
            }
        }
    }

    let mut col_buffer = vec![0; (height + 2 * y_radius) as usize];
    for x in 0..width {
        column_running_sum(&out, x, &mut col_buffer, y_radius);
        let val = col_buffer[(2 * y_radius) as usize] / kernel_height;
        unsafe {
            out.unsafe_put_pixel(x, 0, Luma([val as u8]));
        }
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = (col_buffer[u] - col_buffer[l]) / kernel_height;
            unsafe {
                out.unsafe_put_pixel(x, y, Luma([val as u8]));
            }
        }
    }

    out
}

/// A 2D kernel, used to filter images via convolution.
pub struct Kernel<'a, K: 'a> {
    data: &'a [K],
    width: u32,
    height: u32,
}

impl<'a, K: Num + Copy + 'a> Kernel<'a, K> {
    /// Construct a kernel from a slice and its dimensions. The input slice is
    /// in row-major form.
    pub fn new(data: &'a [K], width: u32, height: u32) -> Kernel<'a, K> {
        assert!(
            width * height == data.len() as u32,
            format!(
                "Invalid kernel len: expecting {}, found {}",
                width * height,
                data.len()
            )
        );
        Kernel {
            data: data,
            width: width,
            height: height,
        }
    }

    /// Returns 2d correlation of an image. Intermediate calculations are performed
    /// at type K, and the results converted to pixel Q via f. Pads by continuity.
    pub fn filter<P, F, Q>(&self, image: &Image<P>, mut f: F) -> Image<Q>
    where
        P: Pixel + 'static,
        <P as Pixel>::Subpixel: ValueInto<K>,
        Q: Pixel + 'static,
        F: FnMut(&mut Q::Subpixel, K) -> (),
    {
        let (width, height) = image.dimensions();
        let mut out = Image::<Q>::new(width, height);
        let num_channels = P::channel_count() as usize;
        let zero = K::zero();
        let mut acc = vec![zero; num_channels];
        let (k_width, k_height) = (self.width, self.height);

        for y in 0..height {
            for x in 0..width {
                for k_y in 0..k_height {
                    let y_p = min(
                        height + height - 1,
                        max(height, (height + y + k_y - k_height / 2)),
                    ) - height;
                    for k_x in 0..k_width {
                        let x_p = min(
                            width + width - 1,
                            max(width, (width + x + k_x - k_width / 2)),
                        ) - width;
                        let (p, k) = unsafe {
                            (
                                image.unsafe_get_pixel(x_p, y_p),
                                *self.data.get_unchecked((k_y * k_width + k_x) as usize),
                            )
                        };
                        accumulate(&mut acc, &p, k);
                    }
                }
                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    f(c, *a);
                    *a = zero;
                }
            }
        }

        out
    }
}

fn gaussian(x: f32, r: f32) -> f32 {
    ((2.0 * f32::consts::PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

/// Construct a one dimensional float-valued kernel for performing a Gausian blur
/// with standard deviation sigma.
fn gaussian_kernel_f32(sigma: f32) -> Vec<f32> {
    let kernel_radius = (2.0 * sigma).ceil() as usize;
    let mut kernel_data = vec![0.0; 2 * kernel_radius + 1];
    for i in 0..kernel_radius + 1 {
        let value = gaussian(i as f32, sigma);
        kernel_data[kernel_radius + i] = value;
        kernel_data[kernel_radius - i] = value;
    }
    kernel_data
}

/// Blurs an image using a Gausian of standard deviation sigma.
/// The kernel used has type f32 and all intermediate calculations are performed
/// at this type.
// TODO: Integer type kernel, approximations via repeated box filter.
pub fn gaussian_blur_f32<P>(image: &Image<P>, sigma: f32) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let kernel = gaussian_kernel_f32(sigma);
    separable_filter_equal(image, &kernel)
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels `h_kernel` and `v_kernel`.
pub fn separable_filter<P, K>(image: &Image<P>, h_kernel: &[K], v_kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    let h = horizontal_filter(image, h_kernel);
    vertical_filter(&h, v_kernel)
}

/// Returns 2d correlation of an image with the outer product of the 1d
/// kernel filter with itself.
pub fn separable_filter_equal<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of an image with a 3x3 row-major kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
pub fn filter3x3<P, K, S>(image: &Image<P>, kernel: &[K]) -> Image<ChannelMap<P, S>>
where
    P::Subpixel: ValueInto<K>,
    S: Clamp<K> + Primitive + 'static,
    P: WithChannel<S> + 'static,
    K: Num + Copy,
{
    let kernel = Kernel::new(kernel, 3, 3);
    kernel.filter(image, |channel, acc| *channel = S::clamp(acc))
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
pub fn horizontal_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::channel_count() as usize];
    let k_width = kernel.len() as i32;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_width >= width as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                    let x_p = max(0, min(x_unchecked, width as i32 - 1)) as u32;
                    let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    *c = <P as Pixel>::Subpixel::clamp(*a);
                    *a = zero;
                }
            }
        }

        return out;
    }

    let half_k = k_width / 2;

    for y in 0..height {
        // Left margin - need to check lower bound only
        for x in 0..half_k {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = max(0, x_unchecked) as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }

        // Neither margin - don't need bounds check on either side
        for x in half_k..(width as i32 - half_k) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = x_unchecked as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }

        // Right margin - need to check upper bound only
        for x in (width as i32 - half_k)..(width as i32) {
            for (i, k) in kernel.iter().enumerate() {
                let x_unchecked = (x as i32) + i as i32 - k_width / 2;
                let x_p = min(x_unchecked, width as i32 - 1) as u32;
                let p = unsafe { image.unsafe_get_pixel(x_p, y) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x as u32, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    out
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
pub fn vertical_filter<P, K>(image: &Image<P>, kernel: &[K]) -> Image<P>
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
    K: Num + Copy,
{
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::channel_count() as usize];
    let k_height = kernel.len() as i32;

    // Typically the image side will be much larger than the kernel length.
    // In that case we can remove a lot of bounds checks for most pixels.
    if k_height >= height as i32 {
        for y in 0..height {
            for x in 0..width {
                for (i, k) in kernel.iter().enumerate() {
                    let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                    let y_p = max(0, min(y_unchecked, height as i32 - 1)) as u32;
                    let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                    accumulate(&mut acc, &p, *k);
                }

                let out_channels = out.get_pixel_mut(x, y).channels_mut();
                for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                    *c = <P as Pixel>::Subpixel::clamp(*a);
                    *a = zero;
                }
            }
        }

        return out;
    }

    let half_k = k_height / 2;

    // Top margin - need to check lower bound only
    for y in 0..half_k {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = max(0, y_unchecked) as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    // Neither margin - don't need bounds check on either side
    for y in half_k..(height as i32 - half_k) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = y_unchecked as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    // Right margin - need to check upper bound only
    for y in (height as i32 - half_k)..(height as i32) {
        for x in 0..width {
            for (i, k) in kernel.iter().enumerate() {
                let y_unchecked = (y as i32) + i as i32 - k_height / 2;
                let y_p = min(y_unchecked, height as i32 - 1) as u32;
                let p = unsafe { image.unsafe_get_pixel(x, y_p) };
                accumulate(&mut acc, &p, *k);
            }

            let out_channels = out.get_pixel_mut(x, y as u32).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                *c = <P as Pixel>::Subpixel::clamp(*a);
                *a = zero;
            }
        }
    }

    out
}

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K)
where
    P: Pixel,
    <P as Pixel>::Subpixel: ValueInto<K>,
    K: Num + Copy,
{
    for i in 0..(P::channel_count() as usize) {
        acc[i as usize] = acc[i as usize] + cast(pixel.channels()[i]) * weight;
    }
}

/// Applies a median filter of given `radius` to an image. Each output pixels is the median
/// of the pixels in a `2 * radius + 1` square of pixels in the input image.
///
/// Pads by continuity. Performs O(radius) operations per pixel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
///
/// let image = gray_image!(
///     1,   2,   3;
///   200,   6,   7;
///     9, 100,  11
/// );
///
/// // Padding by continuity means that the values we use
/// // for computing medians of boundary pixels are:
/// //
/// //   1     1     2     3     3
/// //      -----------------
/// //   1 |   1     2     3 |   3
/// //
/// // 200 | 200     6     7 |   7
/// //
/// //   9 |   9   100    11 |  11
/// //      -----------------
/// //   9     9   100    11    11
///
/// let filtered = gray_image!(
///     2,  3,  3;
///     9,  7,  7;
///     9, 11, 11
/// );
///
/// assert_pixels_eq!(median_filter(&image, 1), filtered);
/// # }
/// ```
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::filter::median_filter;
///
/// // Image channels are handled independently.
/// // This example sets the red channel to have the same
/// // contents as the image from the grayscale example,
/// // the green channel to a vertically inverted copy of that
/// // image and the blue channel to be constant.
/// //
/// // See the image grayscale example for an explanation of how
/// // boundary conditions are handled.
///
/// let image = rgb_image!(
///     [  1,   9, 10], [  2, 100,  10], [  3,  11,  10];
///     [200, 200, 10], [  6,   6,  10], [  7,   7,  10];
///     [  9,   1, 10], [100,   2,  10], [ 11,   3,  10]
/// );
///
/// let filtered = rgb_image!(
///     [ 2,  9, 10], [ 3, 11, 10], [ 3, 11, 10];
///     [ 9,  9, 10], [ 7,  7, 10], [ 7,  7, 10];
///     [ 9,  2, 10], [11,  3, 10], [11,  3, 10]
/// );
///
/// assert_pixels_eq!(median_filter(&image, 1), filtered);
/// # }
/// ```
pub fn median_filter<P>(image: &Image<P>, radius: u32) -> Image<P>
where
    P: Pixel<Subpixel=u8> + 'static
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        return image.clone();
    }

    let mut out = Image::<P>::new(width, height);
    let r = radius as i32;

    let mut hist = initialise_histogram_for_top_left_pixel(&image, radius);
    slide_down_column(&mut hist, &image, &mut out, 0, r);

    for x in 1..width {
        if x % 2 == 0 {
            slide_right(&mut hist, &image, x, 0, r);
            slide_down_column(&mut hist, &image, &mut out, x, r);
        }
        else {
            slide_right(&mut hist, &image, x, height - 1, r);
            slide_up_column(&mut hist, &image, &mut out, x, r);
        }
    }

    out
}

fn initialise_histogram_for_top_left_pixel<P>(image: &Image<P>, radius: u32) -> HistSet
where
    P: Pixel<Subpixel=u8> + 'static
{
    let (width, height) = image.dimensions();
    let kernel_size = (2 * radius + 1) * (2 * radius + 1);
    let num_channels = P::channel_count();

    let mut hist = HistSet::new(num_channels, kernel_size);
    let r = radius as i32;

    for dy in -r..(r + 1) {
        let py = min(max(0, dy), (height as i32 - 1)) as u32;

        for dx in -r..(r + 1) {
            let px = min(max(0, dx), (width as i32 - 1)) as u32;

            let p = unsafe {
                image.unsafe_get_pixel(px, py)
            };

            hist.incr(p.channels());
        }
    }

    hist
}

fn slide_right<P>(hist: &mut HistSet, image: &Image<P>, x: u32, y: u32, r: i32)
where
    P: Pixel<Subpixel=u8> + 'static
{
    let (width, height) = image.dimensions();

    let prev_x = max(0, x as i32 - r - 1) as u32;
    let next_x = min(x as i32 + r, width as i32 - 1) as u32;

    for dy in -r..(r + 1) {
        let py = min(max(0, y as i32 + dy), (height - 1) as i32) as u32;

        let p = unsafe {
            image.unsafe_get_pixel(prev_x, py)
        };

        let q = unsafe {
            image.unsafe_get_pixel(next_x, py)
        };

        hist.decr(p.channels());
        hist.incr(q.channels());
    }
}

fn slide_down_column<P>(hist: &mut HistSet, image: &Image<P>, out: &mut Image<P>, x: u32, r: i32)
where
    P: Pixel<Subpixel=u8> + 'static
{
    let (width, height) = image.dimensions();

    hist.median(out.get_pixel_mut(x, 0).channels_mut());

    for y in 1..height {
        let prev_y = max(0, y as i32 - r - 1) as u32;
        let next_y = min(y as i32 + r, height as i32 - 1) as u32;

        for dx in -r..(r + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            let p = unsafe {
                image.unsafe_get_pixel(px, prev_y)
            };

            let q = unsafe {
                image.unsafe_get_pixel(px, next_y)
            };

            hist.decr(p.channels());
            hist.incr(q.channels());
        }

        hist.median(out.get_pixel_mut(x, y).channels_mut());
    }
}

fn slide_up_column<P>(hist: &mut HistSet, image: &Image<P>, out: &mut Image<P>, x: u32, r: i32)
where
    P: Pixel<Subpixel=u8> + 'static
{
    let (width, height) = image.dimensions();

    hist.median(out.get_pixel_mut(x, height - 1).channels_mut());

    for y in (0..(height-1)).rev() {
        let prev_y = min(y as i32 + r + 1, height as i32 - 1) as u32;
        let next_y = max(0, y as i32 - r) as u32;

        for dx in -r..(r + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            let p = unsafe {
                image.unsafe_get_pixel(px, prev_y)
            };

            let q = unsafe {
                image.unsafe_get_pixel(px, next_y)
            };

            hist.decr(p.channels());
            hist.incr(q.channels());
        }

        hist.median(out.get_pixel_mut(x, y).channels_mut());
    }
}

// A collection of 256-slot histograms, one per image channel.
// Used to implement median_filter.
struct HistSet {
    // One histogram per image channel.
    data: Vec<[u32; 256]>,
    // Calls to `median` will only return the correct answer
    // if there are `expected_count` entries in the relevant
    // histogram in `data`.
    expected_count: u32,
}

impl HistSet {
    fn new(num_channels: u8, expected_count: u32) -> HistSet {
        // Can't use vec![[0u32; 256], num_channels as usize]
        // because arrays of length > 32 aren't cloneable.
        let mut data = vec![];
        for _ in 0..num_channels {
            data.push([0u32; 256]);
        }

        HistSet {
            data: data,
            expected_count: expected_count
        }
    }

    fn incr(&mut self, pixel: &[u8]) {
        unsafe {
            for c in 0..pixel.len() {
                let p = *pixel.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) += 1;
            }
        }
    }

    fn decr(&mut self, pixel: &[u8]) {
        unsafe {
            for c in 0..pixel.len() {
                let p = *pixel.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) -= 1;
            }
        }
    }

    fn median(&self, pixel: &mut [u8]) {
        unsafe {
            for c in 0..pixel.len() {
                *pixel.get_unchecked_mut(c) = self.channel_median(c as u8);
            }
        }
    }

    fn channel_median(&self, c: u8) -> u8 {
        let hist = unsafe {
            self.data.get_unchecked(c as usize)
        };

        let mut count = 0;

        for i in 0..256 {
            unsafe {
                count += *hist.get_unchecked(i);
            }

            if 2 * count >= self.expected_count {
                return i as u8;
            }
        }

        255
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use utils::{gray_bench_image, rgb_bench_image};
    use image::{GenericImage, GrayImage, ImageBuffer, Luma, Rgb};
    use definitions::{Clamp, Image};
    use image::imageops::blur;
    use property_testing::GrayTestImage;
    use utils::pixel_diff_summary;
    use quickcheck::{quickcheck, TestResult};
    use test::{Bencher, black_box};
    use std::cmp::{min, max};

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

    #[bench]
    fn bench_box_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let filtered = box_filter(&image, 7, 7);
            black_box(filtered);
        });
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

        let kernel = vec![1f32 / 3f32; 3];
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

        let kernel = vec![1i32; 3];
        let filtered = separable_filter_equal(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_separable_filter(b: &mut Bencher) {
        let image = gray_bench_image(300, 300);
        let h_kernel = vec![1f32 / 5f32; 5];
        let v_kernel = vec![0.1f32, 0.4f32, 0.3f32, 0.1f32, 0.1f32];
        b.iter(|| {
            let filtered = separable_filter(&image, &h_kernel, &v_kernel);
            black_box(filtered);
        });
    }

    /// Reference implementation of horizontal_filter. Used to validate
    /// the (presumably faster) actual implementation.
    fn horizontal_filter_reference(image: &GrayImage, kernel: &[f32]) -> GrayImage {
        let (width, height) = image.dimensions();
        let mut out = GrayImage::new(width, height);

        for y in 0..height {
            for x in 0..width {

                let mut acc = 0f32;

                for k in 0..kernel.len() {

                    let mut x_unchecked = x as i32 + k as i32 - (kernel.len() / 2) as i32;
                    x_unchecked = max(0, x_unchecked);
                    x_unchecked = min(x_unchecked, width as i32 - 1);

                    let x_checked = x_unchecked as u32;
                    let color = image.get_pixel(x_checked, y)[0];
                    let weight = kernel[k];

                    acc += color as f32 * weight;
                }

                let clamped = u8::clamp(acc);
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

                for k in 0..kernel.len() {

                    let mut y_unchecked = y as i32 + k as i32 - (kernel.len() / 2) as i32;
                    y_unchecked = max(0, y_unchecked);
                    y_unchecked = min(y_unchecked, height as i32 - 1);

                    let y_checked = y_unchecked as u32;
                    let color = image.get_pixel(x, y_checked)[0];
                    let weight = kernel[k];

                    acc += color as f32 * weight;
                }

                let clamped = u8::clamp(acc);
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
        }
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

        let kernel = vec![1f32 / 3f32; 3];
        let filtered = horizontal_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_filter_with_kernel_wider_than_image_does_not_panic() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let kernel = vec![1f32 / 10f32; 10];
        black_box(horizontal_filter(&image, &kernel));
    }

    #[bench]
    fn bench_horizontal_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32 / 5f32; 5];
        b.iter(|| {
            let filtered = horizontal_filter(&image, &kernel);
            black_box(filtered);
        });
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

        let kernel = vec![1f32 / 3f32; 3];
        let filtered = vertical_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_filter_with_kernel_taller_than_image_does_not_panic() {
        let image = gray_image!(
            1, 4, 1;
            4, 7, 4;
            1, 4, 1);

        let kernel = vec![1f32 / 10f32; 10];
        black_box(vertical_filter(&image, &kernel));
    }

    #[bench]
    fn bench_vertical_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32 / 5f32; 5];
        b.iter(|| {
            let filtered = vertical_filter(&image, &kernel);
            black_box(filtered);
        });
    }

    #[test]
    fn test_filter3x3_with_results_outside_input_channel_range() {
        let kernel: Vec<i32> = vec![
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1];

        let image = gray_image!(
            3, 2, 1;
            6, 5, 4;
            9, 8, 7);

        let expected = gray_image_i16!(
            -4i16, -8i16, -4i16;
            -4i16, -8i16, -4i16;
            -4i16, -8i16, -4i16
        );

        let filtered = filter3x3(&image, &kernel);
        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_filter3x3_i32_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel: Vec<i32> = vec![
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1];

        b.iter(|| {
            let filtered: ImageBuffer<Luma<i16>, Vec<i16>> =
                filter3x3::<_, _, i16>(&image, &kernel);
            black_box(filtered);
        });
    }

    /// Baseline implementation of Gaussian blur is that provided by image::imageops.
    /// We can also use this to validate correctnes of any implementations we add here.
    fn gaussian_baseline_rgb<I>(image: &I, stdev: f32) -> Image<Rgb<u8>>
    where
        I: GenericImage<Pixel = Rgb<u8>> + 'static,
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

    macro_rules! bench_median_filter {
        ($name:ident, side: $s:expr, radius: $r:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let filtered = median_filter(&image, $r);
                    black_box(filtered);
                })
            }
        }
    }

    bench_median_filter!(bench_median_filter_s100_r1, side: 100, radius: 1);
    bench_median_filter!(bench_median_filter_s100_r4, side: 100, radius: 4);
    bench_median_filter!(bench_median_filter_s100_r8, side: 100, radius: 8);

    // Reference implementation of median filter - written to be as simple as possible,
    // to validate faster versions against.
    fn reference_median_filter(image: &GrayImage, radius: u32) -> GrayImage {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return image.clone();
        }

        let mut out = GrayImage::new(width, height);
        let filter_side = (2 * radius + 1) as usize;
        let mut neighbors = vec![0u8; filter_side * filter_side];

        let r = radius as i32;

        for y in 0..height {
            for x in 0..width {
                let mut idx = 0;

                for dy in -r..(r + 1) {
                    for dx in -r..(r + 1) {
                        let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;
                        let py = min(max(0, y as i32 + dy), (height - 1) as i32) as u32;

                        neighbors[idx] = image.get_pixel(px, py)[0] as u8;

                        idx += 1;
                    }
                }

                neighbors.sort();

                let m = median(&neighbors);
                out.put_pixel(x, y, Luma([m]));
            }
        }

        out
    }

    fn median(sorted: &[u8]) -> u8 {
        let mid = sorted.len() / 2;
        sorted[mid]
    }

    #[test]
    fn test_median_filter_matches_reference_implementation() {
        fn prop(image: GrayTestImage, radius: u32) -> TestResult {
            let radius = radius % 5;
            let expected = reference_median_filter(&image.0, radius);
            let actual = median_filter(&image.0, radius);

            match pixel_diff_summary(&actual, &expected) {
                None => TestResult::passed(),
                Some(err) => TestResult::error(err),
            }
        }
        quickcheck(prop as fn(GrayTestImage, u32) -> TestResult);
    }
}
