
extern crate image;
extern crate imageproc;

use image::{GrayImage, Luma};
use imageproc::definitions::Image;
use imageproc::map::map_colors;
use imageproc::template_matching::{find_extremes, match_template, MatchTemplateMethod};

use std::f32;

fn convert_to_gray_image(image: &Image<Luma<f32>>) -> GrayImage {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;

    for p in image.iter() {
        lo = if *p < lo { *p } else { lo };
        hi = if *p > hi { *p } else { hi };
    }

    let range = hi - lo;
    let scale = |x| (255.0 * (x - lo) / range) as u8;
    map_colors(image, |p| Luma([scale(p[0])]))
}

fn main() {
    let image1 = image::open("/Users/philip/dev/bug/image.png").unwrap().to_luma();
    let template1 = image::open("/Users/philip/dev/bug/template.png").unwrap().to_luma();

    let img = match_template(
        &image1,
        &template1,
        MatchTemplateMethod::SumOfSquaredErrorsNormalized,
    );

    let img = convert_to_gray_image(&img);
    let result = find_extremes(&img);
    println!("result: {:#?}", result);
}