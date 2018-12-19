
extern crate image;
extern crate imageproc;

use image::{GrayImage, Luma, Rgb};
use imageproc::{
    contrast::stretch_contrast,
    definitions::Image,
    drawing::draw_hollow_circle,
    map::map_colors,
    stats::percentile,
    template_matching::{
        find_extremes,
        match_template,
        MatchTemplateMethod
    }
};

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

// See https://github.com/PistonDevelopers/imageproc/issues/293 for some example inputs
fn main() {
    let image1 = image::open("/Users/philip/dev/bug/image.png").unwrap().to_luma();
    let template1 = image::open("/Users/philip/dev/bug/template.png").unwrap().to_luma();

    let img = match_template(
        &image1,
        &template1,
        MatchTemplateMethod::SumOfSquaredErrorsNormalized,
        //MatchTemplateMethod::SumOfSquaredErrors
    );

    let img = convert_to_gray_image(&img);
    let result = find_extremes(&img);
    println!("result: {:#?}", result);

    // Invert intensities and stretch contrast so that brighter pixels indicate better
    // matches and the distinction between the good matches is increased.
    let img = map_colors(&img, |p| Luma([255u8 - p[0]]));
    let img = stretch_contrast(&img, percentile(&img, 80), percentile(&img, 100));
    // Draw a circle around the best value, for extra garishness
    let img = map_colors(&img, |p| Rgb([p[0], p[0], p[0]]));
    let img = draw_hollow_circle(
        &img,
        (result.min_value_location.0 as i32, result.min_value_location.1 as i32),
        (img.width() / 15) as i32,
        Rgb([0, 255, 0])
    );

    img.save(&"/Users/philip/dev/bug/result.png").unwrap();
    image1.save(&"/Users/philip/dev/bug/image_grey.png").unwrap();
    template1.save(&"/Users/philip/dev/bug/template_grey.png").unwrap();
}