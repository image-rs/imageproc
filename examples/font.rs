//! An example of drawing text. Writes to the user-provided target file.

extern crate image;
extern crate imageproc;
extern crate rusttype;

use std::path::Path;
use std::env;
use imageproc::drawing::draw_text_mut;
use image::{Rgb, RgbImage};
use rusttype::{FontCollection, Scale};

fn main() {

    let arg = if env::args().count() == 2 {
            env::args().nth(1).unwrap()
        } else {
            panic!("Please enter a target file path")
        };

    let path = Path::new(&arg);

    let mut image = RgbImage::new(200, 200);

    let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
    let font = FontCollection::from_bytes(font).into_font().unwrap();

    let height = 12.4;
    let scale = Scale { x: height * 2.0, y: height };
    draw_text_mut(&mut image, Rgb([0u8, 0u8, 255u8]), 0, 0, scale, &font, "Hello, world!");

    let _ = image.save(path).unwrap();
}
