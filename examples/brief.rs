//! An example of a common BRIEF descriptor workflow. First, read the images in.
//! Then, find keypoints and compute descriptors for small patches around them.
//! Match the descriptors and write the output.
//!
//! Use two images of a scene from slightly different positions. Consider
//! downloading images from any of the datasets used in the original BRIEF paper
//! [here][0]. The Wall/Viewpoint dataset seems to work well.
//!
//! [0]: https://www.robots.ox.ac.uk/~vgg/research/affine/
//!
//! From the root of this crate, run:
//!
//! `cargo run --release --example brief -- <first-image> <second-image> <output-image>`

use image::{open, GenericImage, ImageResult, Rgb};
use imageproc::{
    binary_descriptors::{brief, match_binary_descriptors},
    corners::corners_fast9,
    definitions::Image,
    drawing::draw_line_segment_mut,
};

use std::{env, path::Path};

fn main() -> ImageResult<()> {
    if env::args().len() != 4 {
        panic!("Please enter two input files and one output file")
    }

    let first_image_path_arg = env::args().nth(1).unwrap();
    let second_image_path_arg = env::args().nth(2).unwrap();
    let output_image_path_arg = env::args().nth(3).unwrap();

    let first_image_path = Path::new(&first_image_path_arg);
    let second_image_path = Path::new(&second_image_path_arg);

    if !first_image_path.is_file() {
        panic!("First image file does not exist");
    }
    if !second_image_path.is_file() {
        panic!("Second image file does not exist");
    }

    let first_image = open(first_image_path)?.to_luma8();
    let second_image = open(second_image_path)?.to_luma8();

    const CORNER_THRESHOLD: u8 = 84;
    let first_corners = corners_fast9(&first_image, CORNER_THRESHOLD)
        .iter()
        .map(|c| (c.x, c.y))
        .collect::<Vec<(u32, u32)>>();
    let second_corners = corners_fast9(&second_image, CORNER_THRESHOLD)
        .iter()
        .map(|c| (c.x, c.y))
        .collect::<Vec<(u32, u32)>>();

    let start = std::time::Instant::now();
    let (first_descriptors, test_pairs) = brief(&first_image, &first_corners, 256, None);
    let (second_descriptors, _test_pairs) =
        brief(&second_image, &second_corners, 256, Some(test_pairs));
    let elapsed = start.elapsed();
    println!(
        "Computed {} + {} = {} descriptors in {:.2?} ({:.2?} per descriptor)",
        first_descriptors.len(),
        second_descriptors.len(),
        first_descriptors.len() + second_descriptors.len(),
        elapsed,
        elapsed / (first_descriptors.len() + first_descriptors.len()) as u32
    );

    let start = std::time::Instant::now();
    let matches = match_binary_descriptors(&first_descriptors, &second_descriptors, 32);
    let elapsed = start.elapsed();
    println!(
        "Matched {} pairs from {} candidates in {:.2?} ({:?} per candidate)",
        matches.len(),
        first_descriptors.len() * second_descriptors.len(),
        elapsed,
        elapsed / (first_descriptors.len() * second_descriptors.len()) as u32
    );

    let first_image = open(first_image_path)?.to_rgb8();
    let second_image = open(second_image_path)?.to_rgb8();

    let (output_width, output_height) = (
        first_image.width() + second_image.width(),
        u32::max(first_image.height(), second_image.height()),
    );
    let mut output_image = Image::new(output_width, output_height);
    output_image.copy_from(&first_image, 0, 0).unwrap();
    output_image
        .copy_from(&second_image, first_image.width(), 0)
        .unwrap();
    for keypoint_match in matches.iter() {
        let start_point = first_corners[keypoint_match.0];
        let end_point = second_corners[keypoint_match.1];
        draw_line_segment_mut(
            &mut output_image,
            (start_point.0 as f32, start_point.1 as f32),
            (
                (end_point.0 + first_image.width()) as f32,
                end_point.1 as f32,
            ),
            Rgb([0, 255, 0]),
        )
    }
    output_image.save(&output_image_path_arg).unwrap();
    println!("Wrote output image to {}", output_image_path_arg);

    Ok(())
}
