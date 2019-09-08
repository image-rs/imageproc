//! An implementation of [seam carving]. Currently in a pretty rough state.
//! See examples/seam_carving.rs for an example.
//!
//! [seam carving]: https://en.wikipedia.org/wiki/Seam_carving

use crate::definitions::{HasBlack, Image};
use crate::gradients::sobel_gradient_map;
use crate::map::{map_colors, WithChannel};
use image::{GrayImage, Luma, Pixel, Rgb};
use std::cmp::min;

/// An image seam connecting the bottom of an image to its top (in that order).
pub struct VerticalSeam(Vec<u32>);

/// Reduces the width of an image using seam carving.
///
/// Warning: this is very slow! It implements the algorithm from
/// https://inst.eecs.berkeley.edu/~cs194-26/fa16/hw/proj4-seamcarving/imret.pdf, with some
/// extra unnecessary allocations thrown in. Rather than attempting to optimise the implementation
/// of this inherently slow algorithm, the planned next step is to switch to the algorithm from
/// https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/seam-carving-ChinaF.pdf.
pub fn shrink_width<P>(image: &Image<P>, target_width: u32) -> Image<P>
// TODO: this is pretty silly! We should just be able to express that we want a pixel which is a slice of integral values
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    <P as WithChannel<u16>>::Pixel: HasBlack,
{
    assert!(
        target_width <= image.width(),
        "target_width must be <= input image width"
    );

    let iterations = image.width() - target_width;
    let mut result = image.clone();

    for _ in 0..iterations {
        let seam = find_vertical_seam(&result);
        result = remove_vertical_seam(&result, &seam);
    }

    result
}

/// Computes an 8-connected path from the bottom of the image to the top whose sum of
/// gradient magnitudes is minimal.
pub fn find_vertical_seam<P>(image: &Image<P>) -> VerticalSeam
where
    P: Pixel<Subpixel = u8> + WithChannel<u16> + WithChannel<i16> + 'static,
    <P as WithChannel<u16>>::Pixel: HasBlack,
{
    let (width, height) = image.dimensions();
    assert!(
        image.width() >= 2,
        "Cannot find seams if image width is < 2"
    );

    let mut gradients = sobel_gradient_map(&image, |p| {
        let gradient_sum: u16 = p.channels().iter().sum();
        let gradient_mean: u16 = gradient_sum / P::CHANNEL_COUNT as u16;
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
fn set_path_energy(path_energies: &mut Image<Luma<u32>>, x: u32, y: u32) {
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
pub fn remove_vertical_seam<P>(image: &Image<P>, seam: &VerticalSeam) -> Image<P>
where
    P: Pixel + 'static,
{
    assert!(
        seam.0.len() as u32 == image.height(),
        "seam length does not match image height"
    );

    let (width, height) = image.dimensions();
    let mut out = Image::new(width - 1, height);

    for y in 0..height {
        let x_seam = seam.0[(height - y - 1) as usize];
        for x in 0..x_seam {
            out.put_pixel(x, y, *image.get_pixel(x, y));
        }
        for x in (x_seam + 1)..width {
            out.put_pixel(x - 1, y, *image.get_pixel(x, y));
        }
    }

    out
}

/// Draws a series of `seams` on `image` in red. Assumes that the provided seams were
/// removed in the given order from the input image.
pub fn draw_vertical_seams(image: &GrayImage, seams: &[VerticalSeam]) -> Image<Rgb<u8>> {
    let height = image.height();

    let mut offsets = vec![vec![]; height as usize];
    let mut out = map_colors(image, |p| p.to_rgb());

    for seam in seams {
        for (y, x) in (0..height).rev().zip(&seam.0) {
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
    use crate::utils::gray_bench_image;
    use test::{black_box, Bencher};

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
        };
    }

    bench_shrink_width!(bench_shrink_width_s100_r1, side: 100, shrink_by: 1);
    bench_shrink_width!(bench_shrink_width_s100_r4, side: 100, shrink_by: 4);
    bench_shrink_width!(bench_shrink_width_s100_r8, side: 100, shrink_by: 8);
}
