//! An example using the drawing functions. Writes to the user-provided target file.

use image::{Rgb, RgbImage};
use imageproc::drawing::{
    draw_cross_mut, draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_circle_mut,
    draw_hollow_rect_mut, draw_line_segment_mut,
};
use imageproc::rect::Rect;
use std::env;
use std::path::Path;

#[rustfmt::skip]
fn main() {
    let arg = if env::args().count() == 2 {
        env::args().nth(1).unwrap()
    } else {
        panic!("Please enter a target file path")
    };

    let path = Path::new(&arg);

    let red   = Rgb([255u8, 0u8,   0u8]);
    let green = Rgb([0u8,   255u8, 0u8]);
    let blue  = Rgb([0u8,   0u8,   255u8]);
    let white = Rgb([255u8, 255u8, 255u8]);

    let mut image = RgbImage::new(200, 200);

    // Draw some crosses within bounds
    draw_cross_mut(&mut image, white, 5, 5);
    draw_cross_mut(&mut image, red, 9, 9);
    draw_cross_mut(&mut image, blue, 9, 5);
    draw_cross_mut(&mut image, green, 5, 9);
    // Draw a cross totally outside image bounds - does not panic but nothing is rendered
    draw_cross_mut(&mut image, white, 250, 0);
    // Draw a cross partially out of bounds - the part in bounds is rendered
    draw_cross_mut(&mut image, white, 2, 0);

    // Draw a line segment wholly within bounds
    draw_line_segment_mut(&mut image, (20f32, 12f32), (40f32, 60f32), white);
    // Draw a line segment totally outside image bounds - does not panic but nothing is rendered
    draw_line_segment_mut(&mut image, (0f32, -30f32), (40f32, -20f32), white);
    // Draw a line segment partially out of bounds - the part in bounds is rendered
    draw_line_segment_mut(&mut image, (20f32, 180f32), (20f32, 220f32), white);

    // Draw a hollow rect within bounds
    draw_hollow_rect_mut(&mut image, Rect::at(60, 10).of_size(20, 20), white);
    // Outside bounds
    draw_hollow_rect_mut(&mut image, Rect::at(300, 10).of_size(20, 20), white);
    // Partially outside bounds
    draw_hollow_rect_mut(&mut image, Rect::at(90, -10).of_size(30, 20), white);

    // Draw a filled rect within bounds
    draw_filled_rect_mut(&mut image, Rect::at(130, 10).of_size(20, 20), white);
    // Outside bounds
    draw_filled_rect_mut(&mut image, Rect::at(300, 10).of_size(20, 20), white);
    // Partially outside bounds
    draw_filled_rect_mut(&mut image, Rect::at(180, -10).of_size(30, 20), white);

    // Draw a hollow circle within bounds
    draw_hollow_circle_mut(&mut image, (100, 100), 15, white);
    // Outside bounds
    draw_hollow_circle_mut(&mut image, (400, 400), 20, white);
    // Partially outside bounds
    draw_hollow_circle_mut(&mut image, (100, 190), 20, white);

    // Draw a filled circle within bounds
    draw_filled_circle_mut(&mut image, (150, 100), 15, white);
    // Outside bounds
    draw_filled_circle_mut(&mut image, (450, 400), 20, white);
    // Partially outside bounds
    draw_filled_circle_mut(&mut image, (150, 190), 20, white);

    image.save(path).unwrap();
}
