//! An implementation of [seam carving]. Currently in a pretty rough state.
//! See examples/seam_carving.rs for an example.
//!
//! [seam carving]: https://en.wikipedia.org/wiki/Seam_carving

use crate::gradients::sobel_gradient_map;
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Rgb};
use crate::definitions::{HasBlack, Image};
use crate::map::{map_colors, WithChannel};
use std::cmp::min;

/// An image seam connecting the bottom of an image to its top (in that order).
pub struct VerticalSeam(Vec<u32>);

/// Workaround for lack of custom stride support in ImageBuffer.
/// Store an original image and shrink the perceived width.
pub struct PartialImage<I> {
    image: I,
    partial_width: u32
}

impl<I> PartialImage<I> {
    fn reduce_width(&mut self, by: u32) {
        assert!(by >= self.partial_width);
        self.partial_width -= by;
    }
}

impl<I: GenericImageView> GenericImageView for PartialImage<I> {
    type Pixel = I::Pixel;
    type InnerImageView = I;

    fn dimensions(&self) -> (u32, u32) {
        (self.partial_width, self.image.height())
    }

    fn bounds(&self) -> (u32, u32, u32, u32) {
        (0, 0, 0, 0) // TODO (check other impls - docs unhelpfully just say 'the bounding box')
    }

    fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
        // TODO: assert x < partial_width?
        self.image.get_pixel(x, y)
    }

    fn inner(&self) -> &Self::InnerImageView {
        &self.image
    }
}

impl<I: GenericImage> GenericImage for PartialImage<I> {
    type InnerImage = I;

    fn get_pixel_mut(&mut self, x: u32, y: u32) -> &mut Self::Pixel {
        self.image.get_pixel_mut(x, y)
    }

    fn put_pixel(&mut self, x: u32, y: u32, p: Self::Pixel) {
        // TODO: assert x < partial_width?
        self.image.put_pixel(x, y, p);
    }

    fn blend_pixel(&mut self, x: u32, y: u32, p: Self::Pixel) {
        // TODO: assert x < partial_width?
        self.image.blend_pixel(x, y, p);
    }

    fn inner_mut(&mut self) -> &mut Self::InnerImage {
        &mut self.image
    }
}

/// Reduces the width of an image using seam carving.
///
/// Warning: this is very slow! It implements the algorithm from
/// https://inst.eecs.berkeley.edu/~cs194-26/fa16/hw/proj4-seamcarving/imret.pdf, with some
/// extra unnecessary allocations thrown in. Rather than attempting to optimise the implementation
/// of this inherently slow algorithm, the planned next step is to switch to the algorithm from
/// https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/seam-carving-ChinaF.pdf.
pub fn shrink_width<I>(image: &I, target_width: u32) -> Image<I::Pixel>
// TODO: this is pretty silly! We should just be able to express that we want a pixel which is a slice of integral values
where
    I: GenericImageView,
    I::Pixel: Pixel<Subpixel=u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    <I::Pixel as WithChannel<u16>>::Pixel: HasBlack
{
    assert!(target_width <= image.width(), "target_width must be <= input image width");

    let iterations = image.width() - target_width;
    let mut result = ImageBuffer::new(image.width(), image.height());
    result.copy_from(image, 0, 0);
    let mut result = PartialImage {
        image: result,
        partial_width: image.width()
    };

    for _ in 0..iterations {
        let seam = find_vertical_seam(&result);
        remove_vertical_seam(&mut result, &seam);
    }

    result.image
}

/// Computes an 8-connected path from the bottom of the image to the top whose sum of
/// gradient magnitudes is minimal.
pub fn find_vertical_seam<I>(image: &I) -> VerticalSeam
where
    I: GenericImageView,
    I::Pixel: Pixel<Subpixel=u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    <I::Pixel as WithChannel<u16>>::Pixel: HasBlack
{
    let (width, height) = image.dimensions();
    assert!(image.width() >= 2, "Cannot find seams if image width is < 2");

    let mut gradients = sobel_gradient_map(image, |p| {
        let gradient_sum: u16 = p.channels().iter().sum();
        let gradient_mean: u16 = gradient_sum / I::Pixel::channel_count() as u16;
        Luma([gradient_mean as u32])
    });

    // Find the least energy path through the gradient image.
    for y in 1..height {
        for x in 0..width {
            set_path_energy(&mut gradients, x, y);
        }
    }

    // Retrace our steps to find the vertical seam.
    let mut min_x = 0;
    let mut min_energy = gradients.get_pixel(0, height - 1)[0];

    for x in 1..width {
        let c = gradients.get_pixel(x, height - 1)[0];
        if c < min_energy {
            min_x = x;
            min_energy = c;
        }
    }

    let mut seam = Vec::with_capacity(height as usize);

    seam.push(min_x);

    let mut last_x = min_x;

    for y in (1..height).rev() {
        let above = gradients.get_pixel(last_x, y - 1)[0];
        if last_x > 0 {
            let left = gradients.get_pixel(last_x - 1, y - 1)[0];
            if left < above {
                min_x = last_x - 1;
                min_energy = left;
            }
        }
        if last_x < width - 1 {
            let right = gradients.get_pixel(last_x + 1, y - 1)[0];
            if right < min_energy {
                min_x = last_x + 1;
                min_energy = right;
            }
        }

        last_x = min_x;
        seam.push(min_x);
    }

    VerticalSeam(seam)
}

/// Assumes that the previous rows have all been processed.
fn set_path_energy<I>(path_energies: &mut I, x: u32, y: u32)
where
    I: GenericImage<Pixel = Luma<u32>>
{
    let above = path_energies.get_pixel(x, y - 1)[0];
    let mut min_energy = above;

    if x > 0 {
        let above_left = path_energies.get_pixel(x - 1, y - 1)[0];
        min_energy = min(above, above_left);
    }
    if x < path_energies.width() - 1 {
        let above_right = path_energies.get_pixel(x + 1, y - 1)[0];
        min_energy = min(min_energy, above_right);
    }

    let current = path_energies.get_pixel(x, y)[0];
    path_energies.put_pixel(x, y, Luma([min_energy + current]));
}

/// Returns the result of removing `seam` from `image`.
// This should just mutate an image in place. The problem is that we don't have a
// way of talking about views of ImageBuffer without devolving into supporting
// arbitrary GenericImages. And a lot of other functions don't support those because
// it would make them a lot slower.
pub fn remove_vertical_seam<I>(image: &mut PartialImage<I>, seam: &VerticalSeam)
where
    I: GenericImage,
    I::Pixel: Pixel + 'static
{
    assert!(seam.0.len() as u32 == image.height(), "seam length does not match image height");
    let (width, height) = image.dimensions();

    for y in 0..height {
        let x_seam = seam.0[(height - y - 1) as usize];
        // Should memcpy here, but pixelwise will do for now
        for x in (x_seam + 1)..width {
            image.put_pixel(x - 1, y, image.get_pixel(x, y));
        }
    }

    image.reduce_width(1);
}

/// Draws a series of `seams` on `image` in red. Assumes that the provided seams were
/// removed in the given order from the input image.
pub fn draw_vertical_seams(image: &GrayImage, seams: &[VerticalSeam]) -> Image<Rgb<u8>> {
    let height = image.height();

    let mut offsets = vec![vec![]; height as usize];
    let mut out = map_colors(image, |p| p.to_rgb());

    for seam in seams {
        for (y, x) in(0..height).rev().zip(&seam.0) {
            let mut x_original = *x;
            for o in &offsets[y as usize] {
                if *o < *x {
                    x_original += 1;
                }
            }
            out.put_pixel(x_original, y, Rgb([255, 0, 0]));
            offsets[y as usize].push(x_original);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::{Bencher, black_box};
    use crate::utils::gray_bench_image;

    macro_rules! bench_shrink_width {
        ($name:ident, side: $s:expr, shrink_by: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                b.iter(|| {
                    let filtered = shrink_width(&image, $s - $m);
                    black_box(filtered);
                })
            }
        }
    }

    bench_shrink_width!(bench_shrink_width_s100_r1, side: 100, shrink_by: 1);
    bench_shrink_width!(bench_shrink_width_s100_r4, side: 100, shrink_by: 4);
    bench_shrink_width!(bench_shrink_width_s100_r8, side: 100, shrink_by: 8);
}
