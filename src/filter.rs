//! Functions for filtering images.

use image::{
    GrayImage,
    GenericImage,
    ImageBuffer,
    Luma,
    Pixel,
    Primitive
};

use integralimage::{
    column_running_sum,
    row_running_sum
};

use map::{
    WithChannel,
    ChannelMap
};

use definitions::{
    Clamp,
    Image
};

use num::{
    Num
};

use conv::ValueInto;
use math::cast;
use std::cmp;
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
pub fn box_filter(image: &GrayImage, x_radius: u32, y_radius: u32)
        -> Image<Luma<u8>> {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;

    let mut row_buffer = vec![0; (width + 2 * x_radius) as usize];
    for y in 0..height {
        row_running_sum(image, y, &mut row_buffer, x_radius);
        let val = row_buffer[(2 * x_radius) as usize]/kernel_width;
        unsafe { out.unsafe_put_pixel(0, y, Luma([val as u8])); }
        for x in 1..width {
            // TODO: This way we pay rounding errors for each of the
            // TODO: x and y convolutions. Is there a better way?
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = (row_buffer[u] - row_buffer[l])/kernel_width;
            unsafe { out.unsafe_put_pixel(x, y, Luma([val as u8])); }
        }
    }

    let mut col_buffer = vec![0; (height + 2 * y_radius) as usize];
    for x in 0..width {
        column_running_sum(&out, x, &mut col_buffer, y_radius);
        let val = col_buffer[(2 * y_radius) as usize]/kernel_height;
        unsafe { out.unsafe_put_pixel(x, 0, Luma([val as u8])); }
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = (col_buffer[u] - col_buffer[l])/kernel_height;
            unsafe { out.unsafe_put_pixel(x, y, Luma([val as u8])); }
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
        assert!(width * height == data.len() as u32,
            format!("Invalid kernel len: expecting {}, found {}", width * height, data.len()));
        Kernel {
            data: data,
            width: width,
            height: height,
        }
    }

    /// Returns 2d correlation of an image. Intermediate calculations are performed
    /// at type K, and the results converted to pixel Q via f. Pads by continuity.
    pub fn filter<P, F, Q>(&self, image: &Image<P>, mut f: F) -> Image<Q>
        where P: Pixel + 'static,
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
                    let y_p = cmp::min(height + height - 1,
                                       cmp::max(height, (height + y + k_y - k_height / 2))) - height;
                    for k_x in 0..k_width {
                        let x_p = cmp::min(width + width - 1,
                                           cmp::max(width, (width + x + k_x - k_width / 2))) - width;
                        let (p, k) = unsafe {
                            (image.unsafe_get_pixel(x_p, y_p),
                             *self.data.get_unchecked((k_y * k_width + k_x) as usize))
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
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>
{
    let kernel = gaussian_kernel_f32(sigma);
    separable_filter_equal(image, &kernel)
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels `h_kernel` and `v_kernel`.
pub fn separable_filter<P, K>(image: &Image<P>, h_kernel: &[K], v_kernel: &[K])
        -> Image<P>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy
{
    let h = horizontal_filter(image, h_kernel);
    vertical_filter(&h, v_kernel)
}

/// Returns 2d correlation of an image with the outer product of the 1d
/// kernel filter with itself.
pub fn separable_filter_equal<P, K>(image: &Image<P>, kernel: &[K])
        -> Image<P>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of an image with a 3x3 row-major kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
pub fn filter3x3<P, K, S>(image: &Image<P>, kernel: &[K]) -> Image<ChannelMap<P, S>>
    where P::Subpixel: ValueInto<K>,
          S: Clamp<K> + Primitive + 'static,
          P: WithChannel<S> + 'static,
          K: Num + Copy {
    let kernel = Kernel::new(kernel, 3, 3);
    kernel.filter(image, |channel, acc| *channel = S::clamp(acc))
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
pub fn horizontal_filter<P, K>(image: &Image<P>, kernel: &[K])
        -> Image<P>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::channel_count() as usize];
    let k_width = kernel.len() as i32;

    for y in 0..height {
        for x in 0..width {
              for k_x in 0..k_width {
                  let x_unchecked = (x as i32) + k_x - k_width / 2;
                  let x_p = cmp::max(0, cmp::min(x_unchecked, width as i32 - 1)) as u32;
                  let (p, k) = unsafe {
                      (image.unsafe_get_pixel(x_p, y),
                       *kernel.get_unchecked(k_x as usize))
                  };
                  accumulate(&mut acc, &p, k);
              }

              let out_channels = out.get_pixel_mut(x, y).channels_mut();
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
pub fn vertical_filter<P, K>(image: &Image<P>, kernel: &[K])
        -> Image<P>
    where P: Pixel + 'static,
          <P as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    // Don't replace this with a call to Kernel::filter without
    // checking the benchmark results. At the time of writing this
    // specialised implementation is faster.
    let (width, height) = image.dimensions();
    let mut out = Image::<P>::new(width, height);
    let zero = K::zero();
    let mut acc = vec![zero; P::channel_count() as usize];
    let k_height = kernel.len() as i32;

    for y in 0..height {
        for x in 0..width {
              for k_y in 0..k_height {
                  let y_unchecked = (y as i32) + k_y - k_height / 2;
                  let y_p = cmp::max(0, cmp::min(y_unchecked, height as i32 - 1)) as u32;
                  let (p, k) = unsafe {
                      (image.unsafe_get_pixel(x, y_p),
                       *kernel.get_unchecked(k_y as usize))
                  };
                  accumulate(&mut acc, &p, k);
              }

              let out_channels = out.get_pixel_mut(x, y).channels_mut();
              for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                  *c = <P as Pixel>::Subpixel::clamp(*a);
                  *a = zero;
              }
        }
    }

    out
}

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K)
    where P: Pixel, <P as Pixel>::Subpixel : ValueInto<K>, K: Num + Copy {
    for i in 0..(P::channel_count() as usize) {
        acc[i as usize] = acc[i as usize] + cast(pixel.channels()[i]) * weight;
    }
}

#[cfg(test)]
mod test {

    use super::{
        box_filter,
        filter3x3,
        gaussian_blur_f32,
        horizontal_filter,
        separable_filter,
        separable_filter_equal,
        vertical_filter
    };
    use utils::{
        gray_bench_image,
        rgb_bench_image
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma,
        Rgb
    };
    use definitions::Image;
    use image::imageops::blur;
    use test::{
        Bencher,
        black_box
    };

    #[test]
    fn test_box_filter() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9]).unwrap();

        // For this image we get the same answer from the two 1d
        // convolutions as from doing the 2d convolution in one step
        // (but we needn't in general, as in the former case we're
        // clipping to an integer value twice).
        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2, 3, 3,
            4, 5, 5,
            6, 7, 7]).unwrap();

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
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9]).unwrap();

        // Lazily copying the box_filter test case
        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2, 3, 3,
            4, 5, 5,
            6, 7, 7]).unwrap();

        let kernel = vec![1f32/3f32; 3];
        let filtered = separable_filter_equal(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_separable_filter_integer_kernel() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            21, 27, 33,
            39, 45, 51,
            57, 63, 69]).unwrap();

        let kernel = vec![1i32; 3];
        let filtered = separable_filter_equal(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_separable_filter(b: &mut Bencher) {
        let image = gray_bench_image(300, 300);
        let h_kernel = vec![1f32/5f32; 5];
        let v_kernel = vec![0.1f32, 0.4f32, 0.3f32, 0.1f32, 0.1f32];
        b.iter(|| {
            let filtered = separable_filter(&image, &h_kernel, &v_kernel);
            black_box(filtered);
            });
    }

    #[test]
    fn test_horizontal_filter() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 4, 1,
            4, 7, 4,
            1, 4, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2, 2, 2,
            5, 5, 5,
            2, 2, 2]).unwrap();

        let kernel = vec![1f32/3f32; 3];
        let filtered = horizontal_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_horizontal_filter_with_kernel_wider_than_image_does_not_panic() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 4, 1,
            4, 7, 4,
            1, 4, 1]).unwrap();

        let kernel = vec![1f32/10f32; 10];
        black_box(horizontal_filter(&image, &kernel));
    }

    #[bench]
    fn bench_horizontal_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32/5f32; 5];
        b.iter(|| {
            let filtered = horizontal_filter(&image, &kernel);
            black_box(filtered);
            });
    }

    #[test]
    fn test_vertical_filter() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 4, 1,
            4, 7, 4,
            1, 4, 1]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2, 5, 2,
            2, 5, 2,
            2, 5, 2]).unwrap();

        let kernel = vec![1f32/3f32; 3];
        let filtered = vertical_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[test]
    fn test_vertical_filter_with_kernel_taller_than_image_does_not_panic() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1, 4, 1,
            4, 7, 4,
            1, 4, 1]).unwrap();

        let kernel = vec![1f32/10f32; 10];
        black_box(vertical_filter(&image, &kernel));
    }

    #[bench]
    fn bench_vertical_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32/5f32; 5];
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

        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            3, 2, 1,
            6, 5, 4,
            9, 8, 7]).unwrap();

        let expected = ImageBuffer::from_raw(3, 3, vec![
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16,
            -4i16, -8i16, -4i16]).unwrap();

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
            let filtered: ImageBuffer<Luma<i16>, Vec<i16>>
                = filter3x3::<_, _, i16>(&image, &kernel);
            black_box(filtered);
            });
    }

    /// Baseline implementation of Gaussian blur is that provided by image::imageops.
    /// We can also use this to validate correctnes of any implementations we add here.
    fn gaussian_baseline_rgb<I>(image: &I, stdev: f32) -> Image<Rgb<u8>>
        where I: GenericImage<Pixel=Rgb<u8>> + 'static
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
