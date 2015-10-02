//! Functions for filtering images.

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
pub fn box_filter<I: GenericImage<Pixel=Luma<u8>> + 'static>(
    image: &I,
    x_radius: u32,
    y_radius: u32)
    -> ImageBuffer<Luma<u8>, Vec<u8>>
    where I::Pixel: 'static,
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

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
pub fn horizontal_filter<I: GenericImage + 'static>(
    image: &I, kernel: &[f32])
    -> ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>
    where I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ToFloat + Clamp + 'static {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    out.copy_from(image, 0, 0);

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
                for i in 0..num_channels {
                    acc[i] += p.channels()[i].to_float() * kernel[z as usize];
                }
            }

            for c in acc.iter_mut() {
                *c /= k as f32;
            }

            let mut out_pixel = pix;

            for i in 0..num_channels {
                out_pixel.channels_mut()[i]
                    = <I::Pixel as Pixel>::Subpixel::clamp(acc[i]);
            }

            out.put_pixel(x - k/2, y, out_pixel);
        }
	}

    out
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
        pad_buffer
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
    fn test_horizontal_filter() {
        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            1u8, 4u8, 1u8,
            4u8, 7u8, 4u8,
            1u8, 4u8, 1u8]).unwrap();

        let expected: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            2u8, 2u8, 2u8,
            5u8, 5u8, 5u8,
            2u8, 2u8, 2u8]).unwrap();

        let kernel = vec![1f32, 1f32, 1f32];
        let filtered = horizontal_filter(&image, &kernel);

        assert_pixels_eq!(filtered, expected);
    }
}
