//! An example of finding gradients in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! building image in /examples by running the following command line.
//!
//! `cargo run --example gradients ./examples/empire-state-building.jpg ./tmp`

use image::{open, GrayImage};
use imageproc::{filter::filter_clamped, kernel::OwnedKernel, map::map_subpixels};
use std::{env, fs, path::Path};

fn save_gradients(
    input: &GrayImage,
    horizontal_kernel: &OwnedKernel<i16>,
    vertical_kernel: &OwnedKernel<i16>,
    output_dir: &Path,
    name: &str,
    scale: i16,
) {
    let horizontal_gradients = filter_clamped(input, horizontal_kernel);
    let vertical_gradients = filter_clamped(input, vertical_kernel);

    let horizontal_scaled = map_subpixels(&horizontal_gradients, |p: i16| (p.abs() / scale) as u8);
    let vertical_scaled = map_subpixels(&vertical_gradients, |p: i16| (p.abs() / scale) as u8);

    horizontal_scaled
        .save(&output_dir.join(format!("{name}_horizontal.png")))
        .unwrap();
    vertical_scaled
        .save(&output_dir.join(format!("{name}_vertical.png")))
        .unwrap();
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
        .unwrap_or_else(|_| panic!("Could not load image at {:?}", input_path))
        .to_luma8();

    // Save grayscale image in output directory
    let gray_path = output_dir.join("grey.png");
    input_image.save(&gray_path).unwrap();

    for (name, horizontal, vertical, scale) in [
        (
            "sobel",
            OwnedKernel::sobel_horizontal_3x3(),
            OwnedKernel::sobel_vertical_3x3(),
            32,
        ),
        (
            "scharr",
            OwnedKernel::scharr_horizontal_3x3(),
            OwnedKernel::scharr_vertical_3x3(),
            8,
        ),
        (
            "prewitt",
            OwnedKernel::prewitt_horizontal_3x3(),
            OwnedKernel::prewitt_vertical_3x3(),
            6,
        ),
        (
            "roberts",
            OwnedKernel::roberts_horizontal_2x2(),
            OwnedKernel::roberts_vertical_2x2(),
            4,
        ),
    ] {
        save_gradients(
            &input_image,
            &horizontal,
            &vertical,
            output_dir,
            name,
            scale,
        );
    }
}
