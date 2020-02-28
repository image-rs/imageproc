//! An example of finding lines in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! wrench image in /examples by running
//! `cargo run --example hough ./examples/wrench.jpg ./tmp`

use image::{open, Rgb};
use imageproc::edges::canny;
use imageproc::hough::{detect_lines, draw_polar_lines, LineDetectionOptions, PolarLine};
use imageproc::map::map_colors;
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

    // Detect edges using Canny algorithm
    let edges = canny(&input_image, 50.0, 100.0);
    let canny_path = output_dir.join("canny.png");
    edges.save(&canny_path).unwrap();

    // Detect lines using Hough transform
    let options = LineDetectionOptions {
        vote_threshold: 40,
        suppression_radius: 8,
    };
    let lines: Vec<PolarLine> = detect_lines(&edges, options);

    let white = Rgb::<u8>([255, 255, 255]);
    let green = Rgb::<u8>([0, 255, 0]);
    let black = Rgb::<u8>([0, 0, 0]);

    // Convert edge image to colour
    let color_edges = map_colors(&edges, |p| if p[0] > 0 { white } else { black });

    // Draw lines on top of edge image
    let lines_image = draw_polar_lines(&color_edges, &lines, green);
    let lines_path = output_dir.join("lines.png");
    lines_image.save(&lines_path).unwrap();
}
