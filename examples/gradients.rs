//! An example of finding gradients in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! building image in /examples by running the following command line.
//!
//! `cargo run --example gradients ./examples/empire-state-building.jpg ./tmp`

use image::{open, GrayImage};
use imageproc::{filter::filter3x3, gradients::GradientKernel, map::map_subpixels};
use std::{env, fs, path::Path};

fn save_gradients(
    input: &GrayImage,
    gradient_kernel: &GradientKernel,
    output_dir: &Path,
    name: &str,
    scale: i16,
) {
    let horizontal_gradients = filter3x3(input, gradient_kernel.kernel1::<i16>());
    let vertical_gradients = filter3x3(input, gradient_kernel.kernel2::<i16>());

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

    for gradient_kernel in GradientKernel::ALL {
        let scale = match gradient_kernel {
            GradientKernel::Sobel => 32,
            GradientKernel::Scharr => 8,
            GradientKernel::Prewitt => 6,
            GradientKernel::Roberts => 4,
        };

        save_gradients(
            &input_image,
            &gradient_kernel,
            output_dir,
            &format!("{gradient_kernel:?}"),
            scale,
        );
    }
}
