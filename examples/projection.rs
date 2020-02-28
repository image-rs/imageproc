//! Examples of applying projections to an image.
//!
//! If running from the root directory of this crate you can test on the
//! building image in /examples by running the following command line.
//!
//! `cargo run --release --example projection ./examples/empire-state-building.jpg ./tmp

use image::{error::ImageResult, open, Rgb};
use imageproc::geometric_transformations::{warp, Interpolation, Projection};
use std::{env, fs, path::Path};

fn main() -> ImageResult<()> {
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

    let image = open(input_path)
        .expect(&format!("Could not load image at {:?}", input_path))
        .to_rgb();

    let translate = Projection::translate(90.0, 10.0);
    warp(
        &image,
        &translate,
        Interpolation::Bilinear,
        Rgb([255, 0, 0]),
    )
    .save(output_dir.join("translated.png"))?;

    let inverse_translation = translate.invert();
    warp(
        &image,
        &inverse_translation,
        Interpolation::Bilinear,
        Rgb([255, 0, 0]),
    )
    .save(output_dir.join("translated_inverse.png"))?;

    let rotate = Projection::rotate(45f32.to_radians());
    warp(&image, &rotate, Interpolation::Bilinear, Rgb([255, 0, 0]))
        .save(output_dir.join("rotated.png"))?;

    let rotate_then_translate = translate * rotate;
    warp(
        &image,
        &rotate_then_translate,
        Interpolation::Bilinear,
        Rgb([255, 0, 0]),
    )
    .save(output_dir.join("rotated_then_translated.png"))?;

    let translate_then_rotate = rotate * translate;
    warp(
        &image,
        &translate_then_rotate,
        Interpolation::Bilinear,
        Rgb([255, 0, 0]),
    )
    .save(output_dir.join("translated_then_rotated.png"))?;

    let (cx, cy) = (image.width() as f32 / 2.0, image.height() as f32 / 2.0);
    let rotate_about_center =
        Projection::translate(cx, cy) * rotate * Projection::translate(-cx, -cy);
    warp(
        &image,
        &rotate_about_center,
        Interpolation::Bilinear,
        Rgb([255, 0, 0]),
    )
    .save(output_dir.join("rotated_about_center.png"))?;

    let scale = Projection::scale(2.0, 3.0);
    warp(&image, &scale, Interpolation::Bilinear, Rgb([255, 0, 0]))
        .save(output_dir.join("scaled.png"))?;

    Ok(())
}
