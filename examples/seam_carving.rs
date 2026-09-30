//! Example usages of the seam carving functionality

use imageproc::{
    geometric_transformations::{rotate90, rotate270},
    seam_carving::*,
};
use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

fn main() {
    if env::args().len() < 3 {
        println!("Usage: cargo run --release --example seam_carving <path_to_image> <output_dir> [num_seams_to_remove] [only_shrink]
Example:
cargo run --release --example seam_carving tests/data/elephant.png ./output 25

<path_to_image> The path to the source image.
<output_dir> The directory in which the resulting images will be output.
[num_seams_to_remove] Optional: the number of seams to remove from the given image. Defaults to 50
[only_shrink] Optional: specifies if the example should only run the shrink operation or the full suite of example functions with annotated images in the output.");
        exit(1);
    }

    let mut args = env::args().skip(1);

    let input_path = args.next().expect("path argument should exist");
    let output_dir = args.next().expect("output argument should exist");
    let seams_to_remove = args
        .next()
        .map_or(Ok(50usize), |n| n.parse())
        .expect("must be a valid number");
    let only_shrink = args.next().is_some_and(|s| s == "true");

    let input_path = Path::new(&input_path);
    let output_dir = Path::new(&output_dir);

    if !output_dir.is_dir() {
        fs::create_dir(output_dir).expect("Failed to create output directory")
    }

    if !input_path.is_file() {
        panic!("Input file does not exist");
    }

    let input = image::open(input_path)
        .unwrap_or_else(|_| panic!("Could not load image at {:?}", input_path))
        .into_rgba8();

    // If all you need to do is shrink, without caring about the seams themselves, use shrink_width directly.
    if only_shrink {
        let target_width = input.width() - (seams_to_remove as u32);
        let shrunk = shrink_width(&input, target_width);

        shrunk.save(&output_dir.join("shrunk.png")).unwrap();
    } else {
        let vertical_seams = find_vertical_seams(&input);

        let shrunk = remove_vertical_seams(&input, &vertical_seams, seams_to_remove);

        let lowest_energy = vertical_seams
            .seam_energies()
            .iter()
            .min()
            .expect("the image must contain at least one seam");

        println!("Lowest seam energy: {lowest_energy}");

        // Draw annotated original showing all lowest-energy seams
        let annotated_by_energy =
            draw_vertical_seams_by_energy(&input, &vertical_seams, seams_to_remove);

        annotated_by_energy
            .save(&output_dir.join("annotated_seams_by_energy.png"))
            .unwrap();

        // Save the image we shrunk the width of before
        shrunk.save(&output_dir.join("shrunk.png")).unwrap();

        // Shrink both dimensions and save
        let shrunk_once = shrink_width(&input, input.width() - seams_to_remove as u32);
        let rotated = rotate90(&shrunk_once);
        let carved_height = shrink_width(&rotated, input.height() - seams_to_remove as u32);
        let shrunk_both = rotate270(&carved_height);

        shrunk_both
            .save(&output_dir.join("shrunk_both.png"))
            .unwrap();
    }
}
