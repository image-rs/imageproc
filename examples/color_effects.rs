//! Demonstrates adding a color tint and applying a color gradient to a grayscale image.

use image::{open, Luma, Rgb};
use imageproc::map::map_colors;
use imageproc::pixelops::weighted_sum;
use std::env;
use std::path::Path;

/// Tint a grayscale value with the given color.
/// Midtones are tinted most heavily.
pub fn tint(gray: Luma<u8>, color: Rgb<u8>) -> Rgb<u8> {
    let dist_from_mid = ((gray[0] as f32 - 128f32).abs()) / 255f32;
    let scale_factor = 1f32 - 4f32 * dist_from_mid.powi(2);
    weighted_sum(Rgb([gray[0]; 3]), color, 1.0, scale_factor)
}

/// Linearly interpolates between low and mid colors for pixel intensities less than
/// half of maximum brightness and between mid and high for those above.
pub fn color_gradient(gray: Luma<u8>, low: Rgb<u8>, mid: Rgb<u8>, high: Rgb<u8>) -> Rgb<u8> {
    let fraction = gray[0] as f32 / 255f32;
    let (lower, upper, offset) = if fraction < 0.5 {
        (low, mid, 0.0)
    } else {
        (mid, high, 0.5)
    };
    let right_weight = 2.0 * (fraction - offset);
    let left_weight = 1.0 - right_weight;
    weighted_sum(lower, upper, left_weight, right_weight)
}

fn main() {
    let arg = if env::args().count() == 2 {
        env::args().nth(1).unwrap()
    } else {
        panic!("Please enter an input file")
    };
    let path = Path::new(&arg);

    // Load a image::DynamicImage and convert it to a image::GrayImage
    let image = open(path)
        .expect(&format!("Could not load image at {:?}", path))
        .to_luma();

    let blue = Rgb([0u8, 0u8, 255u8]);

    // Apply the color tint to every pixel in the grayscale image, producing a image::RgbImage
    let tinted = map_colors(&image, |pix| tint(pix, blue));
    tinted.save(path.with_file_name("tinted.png")).unwrap();

    // Apply color gradient to each image pixel
    let black = Rgb([0u8, 0u8, 0u8]);
    let red = Rgb([255u8, 0u8, 0u8]);
    let yellow = Rgb([255u8, 255u8, 0u8]);
    let gradient = map_colors(&image, |pix| color_gradient(pix, black, red, yellow));
    gradient.save(path.with_file_name("gradient.png")).unwrap();
}
