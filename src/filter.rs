//! Functions for filtering images.
// TODO: unify with image::sample

use image::{
    GenericImage,
    ImageBuffer,
    Luma,
    Pixel
};

use integralimage::{
    column_running_sum,
    row_running_sum
};

use traits::{
    Clamp,
    ToFloat
};

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
        -> ImageBuffer<Luma<u8>, Vec<u8>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

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
pub fn separable_filter<I>(image: &I, h_kernel: &[f32], v_kernel: &[f32])
        -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ToFloat + Clamp + 'static {
    let h = horizontal_filter(image, h_kernel);
    let v = vertical_filter(&h, v_kernel);
    v
}

/// Returns 2d correlation of view with the outer product of the 1d
/// kernel filter with itself.
pub fn separable_filter_equal<I>(image: &I, kernel: &[f32])
        -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ToFloat + Clamp + 'static {
    separable_filter(image, kernel, kernel)
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
pub fn horizontal_filter<I>(image: &I, kernel: &[f32])
        -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ToFloat + Clamp + 'static {

    let (width, height) = image.dimensions();
    let mut out = copy(image);

    if width == 0 || height == 0 {
        return out;
    }

    // TODO: Add Default instances for pixels
    let pix = image.get_pixel(0, 0);
    let num_channels = I::Pixel::channel_count() as usize;
    let k = kernel.len() as u32;
    let mut buffer = vec![pix; (width + k/2 + k/2) as usize];

	for y in 0..height {
        let left_pad  = image.get_pixel(0, y);
        let right_pad = image.get_pixel(width - 1, y);

	    pad_buffer(&mut buffer, k/2, left_pad, right_pad);

        for x in 0..width {
            buffer[(x + k/2) as usize] = image.get_pixel(x, y);
        }

        for x in k/2..(width + k/2) {
            let mut acc = vec![0f32; num_channels];

            for z in 0..k {
                let p = buffer[(x + z - k/2) as usize];
                let weight = kernel[z as usize];
                accumulate(&mut acc, &p, weight, num_channels);
            }

            let mut out_pixel = pix;
            clamp_acc(&acc, &mut out_pixel, num_channels);

            out.put_pixel(x - k/2, y, out_pixel);
        }
	}

    out
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
// TODO: shares too much code with horizontal_filter
pub fn vertical_filter<I>(
    image: &I, kernel: &[f32])
    -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ToFloat + Clamp + 'static {

    let (width, height) = image.dimensions();
    let mut out = copy(image);

    if width == 0 || height == 0 {
        return out;
    }

    let pix = image.get_pixel(0, 0);
    let num_channels = I::Pixel::channel_count() as usize;
    let k = kernel.len() as u32;
    let mut buffer = vec![pix; (height + k/2 + k/2) as usize];

	for x in 0..width {
        let left_pad  = image.get_pixel(x, 0);
        let right_pad = image.get_pixel(x, height - 1);

	    pad_buffer(&mut buffer, k/2, left_pad, right_pad);

        for y in 0..height {
            buffer[(y + k/2) as usize] = image.get_pixel(x, y);
        }

        for y in k/2..(height + k/2) {
            let mut acc = vec![0f32; num_channels];

            for z in 0..k {
                let p = buffer[(y + z - k/2) as usize];
                let weight = kernel[z as usize];
                accumulate(&mut acc, &p, weight, num_channels);
            }

            let mut out_pixel = pix;
            clamp_acc(&acc, &mut out_pixel, num_channels);

            out.put_pixel(x, y - k/2, out_pixel);
        }
	}

    out
}

pub fn copy<I>(image: &I) -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    out
}

fn accumulate<P>(acc: &mut [f32], pixel: &P, weight: f32, num_channels: usize)
    where P: Pixel + 'static, <P as Pixel>::Subpixel : ToFloat {
    for i in 0..num_channels {
        acc[i as usize]
            += pixel.channels()[i].to_float() * weight;
    }
}

fn clamp_acc<P>(acc: &[f32], pixel: &mut P, num_channels: usize)
    where P: Pixel + 'static, <P as Pixel>::Subpixel: Clamp{
    for i in 0..num_channels {
        pixel.channels_mut()[i] = P::Subpixel::clamp(acc[i]);
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
        ImageBuffer
    };
    use test;

    #[test]
    fn test_box_filter() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1u8, 2u8, 3u8,
            4u8, 5u8, 6u8,
            7u8, 8u8, 9u8]).unwrap();

        // For this image we get the same answer from the two 1d
        // convolutions as from doing the 2d convolution in one step
        // (but we needn't in general, as in the former case we're
        // clipping to an integer value twice).
        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2u8, 3u8, 3u8,
            4u8, 5u8, 5u8,
            6u8, 7u8, 7u8]).unwrap();

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
            1u8, 2u8, 3u8,
            4u8, 5u8, 6u8,
            7u8, 8u8, 9u8]).unwrap();

        // Lazily copying the box_filter test case
        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2u8, 3u8, 3u8,
            4u8, 5u8, 5u8,
            6u8, 7u8, 7u8]).unwrap();

        let kernel = vec![1f32/3f32; 3];
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
            1u8, 4u8, 1u8,
            4u8, 7u8, 4u8,
            1u8, 4u8, 1u8]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2u8, 2u8, 2u8,
            5u8, 5u8, 5u8,
            2u8, 2u8, 2u8]).unwrap();

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
            1u8, 4u8, 1u8,
            4u8, 7u8, 4u8,
            1u8, 4u8, 1u8]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2u8, 5u8, 2u8,
            2u8, 5u8, 2u8,
            2u8, 5u8, 2u8]).unwrap();

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
}
