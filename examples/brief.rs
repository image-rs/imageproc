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
    binary_descriptors::{brief::brief, match_binary_descriptors, BinaryDescriptor},
    corners::{corners_fast9, Corner},
    definitions::Image,
    drawing::draw_line_segment_mut,
    point::Point,
};

use std::{env, path::Path};

fn filter_edge_keypoints(keypoint: &Corner, height: u32, width: u32, radius: u32) -> bool {
    keypoint.x >= radius
        && keypoint.x <= width - radius
        && keypoint.y >= radius
        && keypoint.y <= height - radius
}

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

    let (first_image_height, first_image_width) = (first_image.height(), first_image.width());
    let (second_image_height, second_image_width) = (second_image.height(), second_image.width());

    const CORNER_THRESHOLD: u8 = 70;
    let first_corners = corners_fast9(&first_image, CORNER_THRESHOLD)
        .into_iter()
        .filter(|c| filter_edge_keypoints(c, first_image_height, first_image_width, 16))
        .map(|c| c.into())
        .collect::<Vec<Point<u32>>>();
    println!("Found {} corners in the first image", first_corners.len());
    let second_corners = corners_fast9(&second_image, CORNER_THRESHOLD)
        .into_iter()
        .filter(|c| filter_edge_keypoints(c, second_image_height, second_image_width, 16))
        .map(|c| c.into())
        .collect::<Vec<Point<u32>>>();
    println!("Found {} corners in the second image", second_corners.len());

    let (first_descriptors, test_pairs) = brief(&first_image, &first_corners, 256, None).unwrap();
    let (second_descriptors, _test_pairs) =
        brief(&second_image, &second_corners, 256, Some(&test_pairs)).unwrap();
    println!("Computed descriptors");

    let matches = match_binary_descriptors(&first_descriptors, &second_descriptors, 24, Some(0xc0));
    println!("Matched {} descriptor pairs", matches.len());

    // now that we've matched descriptors in both images, put them side by side
    // and draw lines connecting the descriptors together
    let first_image = open(first_image_path)?.to_rgb8();
    let second_image = open(second_image_path)?.to_rgb8();

    let (output_width, output_height) = (
        first_image_width + second_image_width,
        u32::max(first_image_height, second_image_height),
    );
    let mut output_image = Image::new(output_width, output_height);
    output_image.copy_from(&first_image, 0, 0).unwrap();
    output_image
        .copy_from(&second_image, first_image.width(), 0)
        .unwrap();
    for keypoint_match in matches.iter() {
        let start_point = keypoint_match.0.position();
        let end_point = keypoint_match.1.position();
        draw_line_segment_mut(
            &mut output_image,
            (start_point.x as f32, start_point.y as f32),
            (
                (end_point.x + first_image.width()) as f32,
                end_point.y as f32,
            ),
            Rgb([0, 255, 0]),
        )
    }
    output_image.save(&output_image_path_arg).unwrap();
    println!("Wrote output image to {}", output_image_path_arg);

    Ok(())
}
