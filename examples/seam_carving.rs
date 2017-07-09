extern crate image;
extern crate imageproc;

use image::{GrayImage, open, Luma};
use imageproc::seam_carving::*;
use imageproc::map::map_colors;
use imageproc::definitions::Clamp;
use imageproc::gradients::sobel_gradients;
use std::env;
use std::path::Path;
use std::fs;

fn main() {
    if env::args().len() != 3 {
        panic!("Please enter an input file and a target directory. For example:
    cargo run --release --example seam_carving PATH_TO_IMAGE temp");
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
    
    // We will reduce the image width by this amount, removing one seam at a time.
    let seams_to_remove = 300;

    let mut shrunk = input_image.clone();
    let mut seams = Vec::new();

    // Record each removed seam so that we can draw them on the original image later.
    for i in 0..seams_to_remove {
        println!("Removing seam {}", i);
        let vertical_seam = find_vertical_seam(&shrunk);
        shrunk = remove_vertical_seam(&mut shrunk, &vertical_seam);
        seams.push(vertical_seam);
    }

    // Draw the seams on the original image.
    let annotated = draw_vertical_seams(&input_image, &seams);
    let annotated_path = output_dir.join("annotated.png");
    annotated.save(&annotated_path).unwrap();

    // Draw the seams on the gradient magnitude image.
    let gradients = sobel_gradients(&input_image);
    let clamped_gradients: GrayImage = map_colors(&gradients, |p| Luma([Clamp::clamp(p[0])]));
    let annotated_gradients = draw_vertical_seams(&clamped_gradients, &seams);
    let gradients_path = output_dir.join("gradients.png");
    clamped_gradients.save(&gradients_path).unwrap();
    let annotated_gradients_path = output_dir.join("annotated_gradients.png");
    annotated_gradients.save(&annotated_gradients_path).unwrap();

    // Save the shrunk image.
    let shrunk_path = output_dir.join("shrunk.png");
    shrunk.save(&shrunk_path).unwrap();
}