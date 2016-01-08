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

use map::{
    WithChannel,
    ChannelMap
};

use definitions::{
    Clamp,
    VecBuffer
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

/// Returns 2d correlation of view with the outer product of the 1d
/// kernels h_filter and v_filter.
pub fn separable_filter<I, K>(image: &I, h_kernel: &[K], v_kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
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
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    separable_filter(image, kernel, kernel)
}

/// Returns 2d correlation of an image with a kernel. Intermediate calculations are
/// performed at type K, and the results converted to pixel Q via f. Pads by continuity.
pub fn filter_kernel<I, K, F, Q>(image: &I, kernel: &[K], 
                                 k_width: u32, k_height: u32, mut f: F) 
        -> VecBuffer<Q>
    where I: GenericImage,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K>,
          Q: Pixel + 'static,
          F: FnMut(&mut Q::Subpixel, K) -> (),
          K: Num + Copy {

    assert!(kernel.len() as u32 >= k_width * k_height,
            "incompatible kernel lengths".to_owned());

    let (w, h) = image.dimensions();
    let mut out = VecBuffer::<Q>::new(w, h);
    let num_channels = I::Pixel::channel_count() as usize;
    let zero = K::zero();
    let mut acc = vec![zero; num_channels];

    for y in 0..h {
        for x in 0..w {
            for k_y in 0..k_height {
                let y_p = cmp::min(h + h - 1, cmp::max(h, (h + y + k_y - k_height / 2))) - h;
                for k_x in 0..k_width {
                    let x_p = cmp::min(w + w - 1, cmp::max(w, (w + x + k_x - k_width / 2))) - w;
                    let (p, k) = unsafe {
                        (image.unsafe_get_pixel(x_p, y_p),
                         *kernel.get_unchecked((k_y * k_width + k_x) as usize))
                    };
                    accumulate(&mut acc, &p, k);
                }
            }
            let out_channels = out.get_pixel_mut(x, y).channels_mut();
            for (a, c) in acc.iter_mut().zip(out_channels.iter_mut()) {
                f(c, a.clone());
                *a = zero;
            }
        }
    }

    out
}

/// Returns 2d correlation of an image with a 3x3 kernel. Intermediate calculations are
/// performed at type K, and the results clamped to subpixel type S. Pads by continuity.
pub fn filter3x3<I, P, K, S>(image: &I, kernel: &[K]) -> VecBuffer<ChannelMap<P, S>>
    where I: GenericImage<Pixel=P>,
          P::Subpixel: ValueInto<K>,
          S: Clamp<K> + Primitive + 'static,
          P: WithChannel<S> + 'static,
          K: Num + Copy {
    filter_kernel(image, kernel, 3, 3, |channel, acc| *channel = S::clamp(acc))
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity. Intermediate calculations are performed at
/// type K.
pub fn horizontal_filter<I, K>(image: &I, kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    filter_kernel(image, kernel, kernel.len() as u32, 1,
        |channel, acc| *channel = <I::Pixel as Pixel>::Subpixel::clamp(acc))
}

///	Returns horizontal correlations between an image and a 1d kernel.
/// Pads by continuity.
pub fn vertical_filter<I, K>(image: &I, kernel: &[K])
        -> VecBuffer<I::Pixel>
    where I: GenericImage,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<K> + Clamp<K>,
          K: Num + Copy {
    filter_kernel(image, kernel, 1, kernel.len() as u32,
        |channel, acc| *channel = <I::Pixel as Pixel>::Subpixel::clamp(acc))
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
        horizontal_filter,
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
        Luma
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
