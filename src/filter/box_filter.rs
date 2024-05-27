use image::{GenericImage, GenericImageView, GrayImage, Luma};

use crate::{
    definitions::Image,
    integral_image::{column_running_sum, row_running_sum},
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

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
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_box_filter(b: &mut Bencher) {
        let image = gray_bench_image(500, 500);
        b.iter(|| {
            let filtered = box_filter(&image, 7, 7);
            black_box(filtered);
        });
    }
}
