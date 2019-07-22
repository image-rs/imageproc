//! An example of finding gradients in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! wrench image in /examples by running
//! `cargo run --example scharr ./examples/empire-state-building.jpg ./tmp`

use image::open;
use image::Luma;
use imageproc::definitions::Image;
use imageproc::gradients::{horizontal_scharr, vertical_scharr};
use imageproc::map::map_pixels;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    if env::args().len() != 3 {
        panic!("Please enter an input file and a target directory")
    }

    let input_path = env::args().nth(1).unwrap();
    let output_dir = env::args().nth(2).unwrap();

    let input_path = Path::new(&input_path);
    let output_dir = Path::new(&output_dir);

    if !output_dir.is_dir() {
        fs::create_dir(output_dir).expect("Failed to create output directory")
    }

    if !input_path.is_file() {
        panic!("Input file does not exist");
    }

    // Load image and convert to grayscale
    let input_image = open(input_path)
        .expect(&format!("Could not load image at {:?}", input_path))
        .to_luma();

    // Save grayscale image in output directory
    let gray_path = output_dir.join("grey.png");
    input_image.save(&gray_path).unwrap();

    let horizontal_scharr_gradient = horizontal_scharr(&input_image);
    save_gradient(
        &horizontal_scharr_gradient,
        &output_dir,
        "scharr_horizontal.png",
    );

    let vertical_scharr_gradient = vertical_scharr(&input_image);
    save_gradient(
        &vertical_scharr_gradient,
        &output_dir,
        "scharr_vertical.png",
    );
}

fn save_gradient(gradient: &Image<Luma<i16>>, output_dir: &Path, filename: &str) {
    // for gradient, simply take the absolute value and use as u8
    let gray_scale_image = map_pixels(&gradient, |_, _, p| Luma([(p[0].abs()) as u8]));
    let output_image_filename = output_dir.join("output.png");
    gray_scale_image.save(&output_image_filename).unwrap();
}
