//! Demonstrates the current incorrect handling of gamma for RGB images

use image::{ImageBuffer, Rgb};
use imageproc::pixelops::interpolate;

fn main() {
    let red = Rgb::<u8>([255, 0, 0]);
    let green = Rgb::<u8>([0, 255, 0]);

    // We'll create an 800 pixel wide gradient image.
    let left_weight = |x| x as f32 / 800.0;

    let naive_blend = |x| interpolate(red, green, left_weight(x));

    let mut naive_image = ImageBuffer::new(800, 400);
    for y in 0..naive_image.height() {
        for x in 0..naive_image.width() {
            naive_image.put_pixel(x, y, naive_blend(x));
        }
    }
    naive_image.save("naive_blend.png").unwrap();

    let gamma = 2.2f32;
    let gamma_inv = 1.0 / gamma;

    let gamma_blend_channel = |l, r, w| {
        let l = (l as f32).powf(gamma);
        let r = (r as f32).powf(gamma);
        let s: f32 = l * w + r * (1.0 - w);
        s.powf(gamma_inv) as u8
    };

    let gamma_blend = |x| {
        let w = left_weight(x);
        Rgb([
            gamma_blend_channel(red[0], green[0], w),
            gamma_blend_channel(red[1], green[1], w),
            gamma_blend_channel(red[2], green[2], w),
        ])
    };

    let mut gamma_image = ImageBuffer::new(800, 400);
    for y in 0..gamma_image.height() {
        for x in 0..gamma_image.width() {
            gamma_image.put_pixel(x, y, gamma_blend(x));
        }
    }
    gamma_image.save("gamma_blend.png").unwrap();
}
