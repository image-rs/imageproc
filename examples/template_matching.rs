//! An example of template matching in a greyscale image.
//! If running from the root directory of this crate you can test on the
//! wrench image in /examples by running
//! `cargo run --example template_matching ./examples/wrench.jpg ./tmp`
extern crate image;
extern crate imageproc;

use std::env;
use std::path::Path;
use std::fs;
use std::f32;
use image::{open, GenericImage, GrayImage, Luma, Rgb};
use imageproc::definitions::Image;
use imageproc::template_matching::match_template;
use imageproc::map::map_colors;
use imageproc::rect::Rect;
use imageproc::drawing::draw_hollow_rect_mut;

/// Convert an f32-valued image to a 8 bit depth, covering the whole
/// available intensity range.
fn convert_to_gray_image(image: &Image<Luma<f32>>) -> GrayImage {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for p in image.iter() {
        lo = if *p < lo { *p } else { lo };
        hi = if *p > hi { *p } else { hi };
    }

    let range = hi - lo;
    let scale = |x| (255.0 * (x - lo) / range) as u8;
    map_colors(image, |p| {
        Luma([scale(p[0])])
    })
}

fn copy_sub_image(image: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    assert!(x + w < image.width() && y + h < image.height(), "invalid sub-image");

    let mut result = GrayImage::new(w, h);
    for sy in 0..h {
        for sx in 0..w {
            result.put_pixel(sx, sy, *image.get_pixel(x + sx, y + sy));
        }
    }

    result
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
    let image = open(input_path)
        .expect(&format!("Could not load image at {:?}", input_path))
        .to_luma();

    // Save grayscale image in output directory
    let gray_path = output_dir.join("image.png");
    image.save(&gray_path).unwrap();

    // Crop a section of the input image as the template
    // This will fail if the input image is too small...
    let template_width = 25;
    let template_height = 25;
    let template_x = 157;
    let template_y = 85;
    let template = copy_sub_image(&image, template_x, template_y, template_width, template_height);
    let template_path = output_dir.join("template.png");
    template.save(&template_path).unwrap();

    // Match the template and convert to u8 depth to display
    let result = match_template(&image, &template);
    let result_scaled = convert_to_gray_image(&result);

    // Pad the result to the same size as the input image, to make
    // it easier to compare the two
    let mut result_padded = GrayImage::new(image.width(), image.height());
    result_padded.copy_from(&result_scaled, template_width / 2, template_height / 2);

    // Show location the template was extracted from
    let mut result_padded = map_colors(&result_padded, |p| Rgb([p[0], p[0], p[0]]));
    draw_hollow_rect_mut(
        &mut result_padded,
        Rect::at(template_x as i32, template_y as i32)
            .of_size(template_width, template_height),
        Rgb([0, 255, 0]));

    let result_path = output_dir.join("result.png");
    result_padded.save(&result_path).unwrap();
}
