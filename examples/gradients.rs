//! An example of finding gradients in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! building image in /examples by running the following command line.
//!
//! `cargo run --example gradients ./examples/empire-state-building.jpg ./tmp`

use image::{open, GrayImage, Luma};
use imageproc::{
    definitions::Image,
    gradients::{
        horizontal_prewitt, horizontal_scharr, horizontal_sobel, vertical_prewitt, vertical_scharr,
        vertical_sobel,
    },
    map::map_pixels,
};
use std::{env, fs, path::Path};

fn compute_gradients<F>(
    input: &GrayImage,
    gradient_func: F,
    output_dir: &Path,
    file_name: &str,
    scale: i16,
) where
    F: Fn(&GrayImage) -> Image<Luma<i16>>,
{
    let gradient = gradient_func(&input);
    let gradient = map_pixels(&gradient, |_, _, p| Luma([(p[0].abs() / scale) as u8]));
    gradient.save(&output_dir.join(file_name)).unwrap();
}

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

    compute_gradients(
        &input_image,
        horizontal_scharr,
        &output_dir,
        "horizontal_scharr.png",
        32,
    );
    compute_gradients(
        &input_image,
        vertical_scharr,
        &output_dir,
        "vertical_scharr.png",
        32,
    );
    compute_gradients(
        &input_image,
        horizontal_sobel,
        &output_dir,
        "horizontal_sobel.png",
        8,
    );
    compute_gradients(
        &input_image,
        vertical_sobel,
        &output_dir,
        "vertical_sobel.png",
        8,
    );
    compute_gradients(
        &input_image,
        horizontal_prewitt,
        &output_dir,
        "horizontal_prewitt.png",
        6,
    );
    compute_gradients(
        &input_image,
        vertical_prewitt,
        &output_dir,
        "vertical_prewitt.png",
        6,
    );
}
