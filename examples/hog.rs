//! Demonstrates computing and visualising HoG gradients.

use image::{open, ImageBuffer};
use imageproc::hog::*;
use std::env;
use std::path::Path;

fn create_hog_image(input: &Path, signed: bool) {
    // Load a image::DynamicImage and convert it to a image::GrayImage
    let image = open(input)
        .expect(&format!("Could not load image at {:?}", input))
        .to_luma();

    // We're not going to do anything interesting with the block sizes here - they're
    // only relevant when combining and normalising per-cell histograms, and in this
    // example we're just going to compute and visualise the histograms. However, to
    // create a HogSpec from these HogOptions we still require that the image is evenly
    // divisible into blocks.
    let opts = HogOptions {
        orientations: 8,
        signed: signed,
        cell_side: 5,
        block_side: 2,
        block_stride: 1,
    };

    let (width, height) = image.dimensions();
    assert!(
        width >= 10 && height >= 10,
        "input file must have width and height both >= 10"
    );

    // Crop image to a suitable size
    let (cropped_width, cropped_height) = (10 * (width / 10), 10 * (height / 10));
    let mut cropped = ImageBuffer::new(cropped_width, cropped_height);
    for y in 0..cropped_height {
        for x in 0..cropped_width {
            cropped.put_pixel(x, y, *image.get_pixel(x, y));
        }
    }

    cropped.save(input.with_file_name("cropped.png")).unwrap();

    let spec = HogSpec::from_options(cropped_width, cropped_height, opts).unwrap();
    let mut hist = cell_histograms(&cropped, spec);

    let star_side = 20;
    let hog = render_hist_grid(star_side, &hist.view_mut(), signed);

    let output_name = if signed {
        "hog_signed.png"
    } else {
        "hog_unsigned.png"
    };
    hog.save(input.with_file_name(output_name)).unwrap();
}

fn main() {
    let arg = if env::args().count() == 2 {
        env::args().nth(1).unwrap()
    } else {
        panic!("Please enter an input file")
    };
    let path = Path::new(&arg);

    create_hog_image(path, true);
    create_hog_image(path, false);
}
