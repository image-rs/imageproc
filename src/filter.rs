//! Functions for filtering images.
// TODO: unify with image::sample

use image::{
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

use definitions::{
    Clamp,
    HasBlack,
    VecBuffer,
    WithChannel,
    ChannelMap
};

use num::{
    Num,
    Zero
};

use conv::ValueInto;
use math::cast;
use std::cmp;

/// Convolves an 8bpp grayscale image with a kernel of width (2 * x_radius + 1)
/// and height (2 * y_radius + 1) whose entries are equal and
/// sum to one. i.e. each output pixel is the unweighted mean of
/// a rectangular region surrounding its corresponding input pixel.
/// We handle locations where the kernel would extend past the image's
/// boundary by treating the image as if its boundary pixels were
/// repeated indefinitely.
// TODO: for small kernels we probably want to do the convolution
// TODO: directly instead of using an integral image.
// TODO: more formats!
// TODO: number of operations is constant with kernel size,
// TODO: but this is still _really_ slow. fix!
pub fn box_filter<I>(image: &I, x_radius: u32, y_radius: u32)
        -> VecBuffer<Luma<u8>>
    where I: GenericImage<Pixel=Luma<u8>>{

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;

    let mut row_buffer = vec![0; (width + 2 * x_radius) as usize];
    for y in 0..height {
        row_running_sum(image, y, &mut row_buffer, x_radius);
        let val = row_buffer[(2 * x_radius) as usize]/kernel_width;
        out.put_pixel(0, y, Luma([val as u8]));
        for x in 1..width {
            // TODO: This way we pay rounding errors for each of the
            // TODO: x and y convolutions. Is there a better way?
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = (row_buffer[u] - row_buffer[l])/kernel_width;
            out.put_pixel(x, y, Luma([val as u8]));
        }
    }

    let mut col_buffer = vec![0; (height + 2 * y_radius) as usize];
    for x in 0..width {
        column_running_sum(&out, x, &mut col_buffer, y_radius);
        let val = col_buffer[(2 * y_radius) as usize]/kernel_height;
        out.put_pixel(x, 0, Luma([val as u8]));
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = (col_buffer[u] - col_buffer[l])/kernel_height;
            out.put_pixel(x, y, Luma([val as u8]));
        }
    }

    out
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels h_filter and v_filter.
pub fn separable_filter<I, K>(image: &I, h_kernel: &[K], v_kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K> ,
          K: Num + Copy {
    let h = horizontal_filter(image, h_kernel);
    let v = vertical_filter(&h, v_kernel);
    v
}

/// Returns 2d correlation of an image with the outer product of the 1d
/// kernel filter with itself.
pub fn separable_filter_equal<I, K>(image: &I, kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of an image with a 3x3 kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
// TODO: factor out the accumulation code from this, horizontal_filter and vertical_filter
pub fn filter3x3<I, P, K, S>(image: &I, kernel: &[K]) -> VecBuffer<ChannelMap<P, S>>
    where I: GenericImage<Pixel=P>,
          P::Subpixel: ValueInto<K>,
          S: Clamp<K> + Primitive + 'static,
          P: WithChannel<S> + 'static,
          K: Num + Copy {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::<ChannelMap<P, S>>::new(width, height);
    let num_channels = I::Pixel::channel_count() as usize;

    // TODO: Should we handle images with height or width < 2? Feels clunky to return Results
    // TODO: everywhere. Could just document as requirement and leave it at that.
    for y in 0..height {
        let y_prev = cmp::max(1, y) - 1;
        let y_next = cmp::min(height - 2, y) + 1;

        for x in 0..width {
            let x_prev = cmp::max(1, x) - 1;
            let x_next = cmp::min(width - 2, x) + 1;

            let mut acc = vec![Zero::zero(); num_channels];
            accumulate(&mut acc, &image.get_pixel(x_prev, y_prev), kernel[0], num_channels);
            accumulate(&mut acc, &image.get_pixel(x,      y_prev), kernel[1], num_channels);
            accumulate(&mut acc, &image.get_pixel(x_next, y_prev), kernel[2], num_channels);
            accumulate(&mut acc, &image.get_pixel(x_prev, y     ), kernel[3], num_channels);
            accumulate(&mut acc, &image.get_pixel(x     , y     ), kernel[4], num_channels);
            accumulate(&mut acc, &image.get_pixel(x_next, y     ), kernel[5], num_channels);
            accumulate(&mut acc, &image.get_pixel(x_prev, y_next), kernel[6], num_channels);
            accumulate(&mut acc, &image.get_pixel(x     , y_next), kernel[7], num_channels);
            accumulate(&mut acc, &image.get_pixel(x_next, y_next), kernel[8], num_channels);

            clamp_acc(
                &acc,
                out.get_pixel_mut(x, y).channels_mut(),
                num_channels);
        }
    }

    out
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
pub fn horizontal_filter<I, K>(image: &I, kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {

    let (width, height) = image.dimensions();
    let mut out = copy(image);

    let num_channels = I::Pixel::channel_count() as usize;
    let k = kernel.len() as u32;
    let mut buffer = vec![I::Pixel::black(); (width + k/2 + k/2) as usize];

	for y in 0..height {
        let left_pad  = image.get_pixel(0, y);
        let right_pad = image.get_pixel(width - 1, y);

	    pad_buffer(&mut buffer, k/2, left_pad, right_pad);

        for x in 0..width {
            buffer[(x + k/2) as usize] = image.get_pixel(x, y);
        }

        for x in k/2..(width + k/2) {
            let mut acc = vec![Zero::zero(); num_channels];

            for z in 0..k {
                let p = buffer[(x + z - k/2) as usize];
                let weight = kernel[z as usize];
                accumulate(&mut acc, &p, weight, num_channels);
            }

            clamp_acc(
                &acc,
                out.get_pixel_mut(x - k/2, y).channels_mut(),
                num_channels);
        }
	}

    out
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
// TODO: shares too much code with horizontal_filter
pub fn vertical_filter<I, K>(image: &I, kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: HasBlack + 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {

    let (width, height) = image.dimensions();
    let mut out = copy(image);

    let num_channels = I::Pixel::channel_count() as usize;
    let k = kernel.len() as u32;
    let mut buffer = vec![I::Pixel::black(); (height + k/2 + k/2) as usize];

	for x in 0..width {
        let left_pad  = image.get_pixel(x, 0);
        let right_pad = image.get_pixel(x, height - 1);

	    pad_buffer(&mut buffer, k/2, left_pad, right_pad);

        for y in 0..height {
            buffer[(y + k/2) as usize] = image.get_pixel(x, y);
        }

        for y in k/2..(height + k/2) {
            let mut acc = vec![Zero::zero(); num_channels];

            for z in 0..k {
                let p = buffer[(y + z - k/2) as usize];
                let weight = kernel[z as usize];
                accumulate(&mut acc, &p, weight, num_channels);
            }

            clamp_acc(
                &acc,
                out.get_pixel_mut(x, y - k/2).channels_mut(),
                num_channels);
        }
	}

    out
}

pub fn copy<I>(image: &I) -> VecBuffer<I::Pixel>
    where I: GenericImage, I::Pixel: 'static {
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    out
}

fn accumulate<P, K>(acc: &mut [K], pixel: &P, weight: K, num_channels: usize)
    where P: Pixel, <P as Pixel>::Subpixel : ValueInto<K>, K: Num + Copy {
    for i in 0..num_channels {
        acc[i as usize] = acc[i as usize] + cast(pixel.channels()[i]) * weight;
    }
}

fn clamp_acc<C, K>(acc: &[K], channels: &mut [C], num_channels: usize)
    where C: Clamp<K>,
          K: Copy {
    for i in 0..num_channels {
        channels[i] = C::clamp(acc[i]);
    }
}

/// Fills the left margin entries in buffer with left_val and the
// right margin entries with right_val
fn pad_buffer<T>(buffer: &mut[T], margin: u32, left_val: T, right_val: T)
    where T : Copy {
    for i in 0..margin {
        buffer[i as usize] = left_val;
    }

    for i in (buffer.len() - margin as usize)..buffer.len() {
        buffer[i] = right_val;
    }
}

#[cfg(test)]
mod test {

    use super::{
        box_filter,
        filter3x3,
        horizontal_filter,
        pad_buffer,
        separable_filter,
        separable_filter_equal,
        vertical_filter
    };
    use utils::{
        gray_bench_image
    };
    use image::{
        GrayImage,
        ImageBuffer,
        Luma,
        Pixel
    };
    use test;

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
    fn bench_box_filter(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let filtered = box_filter(&image, 7, 7);
            test::black_box(filtered);
            });
    }

    #[test]
    fn test_pad_buffer() {
        let mut a = [0, 1, 2, 3, 0];
        pad_buffer(&mut a, 1, 4, 5);
        assert_eq!(a, [4, 1, 2, 3, 5]);
        pad_buffer(&mut a, 2, 8, 9);
        assert_eq!(a, [8, 8, 2, 9, 9]);

        let mut b = [0, 1, 2, 0];
        pad_buffer(&mut b, 1, 4, 5);
        assert_eq!(b, [4, 1, 2, 5]);
        pad_buffer(&mut b, 2, 8, 9);
        assert_eq!(b, [8, 8, 9, 9]);
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
    fn bench_separable_filter(b: &mut test::Bencher) {
        let image = gray_bench_image(300, 300);
        let h_kernel = vec![1f32/5f32; 5];
        let v_kernel = vec![0.1f32, 0.4f32, 0.3f32, 0.1f32, 0.1f32];
        b.iter(|| {
            let filtered = separable_filter(&image, &h_kernel, &v_kernel);
            test::black_box(filtered);
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

        let kernel = vec![1f32/3f32, 1f32/3f32, 1f32/3f32];
        let filtered = horizontal_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_horizontal_filter(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32/5f32; 5];
        b.iter(|| {
            let filtered = horizontal_filter(&image, &kernel);
            test::black_box(filtered);
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

        let kernel = vec![1f32/3f32, 1f32/3f32, 1f32/3f32];
        let filtered = vertical_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }

    #[bench]
    fn bench_vertical_filter(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel = vec![1f32/5f32; 5];
        b.iter(|| {
            let filtered = vertical_filter(&image, &kernel);
            test::black_box(filtered);
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
    fn bench_filter3x3_i32_filter(b: &mut test::Bencher) {
        let image = gray_bench_image(500, 500);
        let kernel: Vec<i32> = vec![
            -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1];

        b.iter(|| {
            let filtered: ImageBuffer<Luma<i16>, Vec<i16>>
                = filter3x3::<_, _, _, i16>(&image, &kernel);
            test::black_box(filtered);
            });
    }
}
