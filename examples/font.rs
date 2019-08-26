//! An example of drawing text. Writes to the user-provided target file.

use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rusttype::{FontCollection, Scale};
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
    let font = FontCollection::from_bytes(font)
        .unwrap()
        .into_font()
        .unwrap();

    let height = 12.4;
    let scale = Scale {
        x: height * 2.0,
        y: height,
    };
    draw_text_mut(
        &mut image,
        Rgb([0u8, 0u8, 255u8]),
        0,
        0,
        scale,
        &font,
        "Hello, world!",
    );

    let _ = image.save(path).unwrap();
}
