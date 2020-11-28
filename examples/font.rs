//! An example of drawing text. Writes to the user-provided target file.

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use rusttype::{Font, Scale};
use std::env;
use std::path::Path;

fn main() {
    let arg = if env::args().count() == 2 {
        env::args().nth(1).unwrap()
    } else {
        panic!("Please enter a target file path")
    };

    let path = Path::new(&arg);

    let mut image = RgbImage::new(200, 200);

    let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();

    let height = 12.4;
    let scale = Scale {
        x: height * 2.0,
        y: height,
    };

    let text = "Hello, world!";
    draw_text_mut(&mut image, Rgb([0u8, 0u8, 255u8]), 0, 0, scale, &font, text);
    let (w, h) = text_size(scale, &font, text);
    println!("Text size: {}x{}", w, h);

    let _ = image.save(path).unwrap();
}
