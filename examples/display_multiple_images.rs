//! Run this example from your root directory, enable the display_image feature and
//! provide a path to an image file as an argument.
//!
//! `cargo run --release --features display-window --example display_multiple_images examples/wrench.jpg examples/empire-state-building.jpg`

#[cfg(feature = "display-window")]
fn main() {
    use imageproc::window::display_multiple_images;
    use std::env;

    let first_image_path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            println!("No path provided for first image. Using default image.");
            "examples/wrench.jpg".to_owned()
        }
    };

    let second_image_path = match env::args().nth(2) {
        Some(path) => path,
        None => {
            println!("No path provided for second image. Using default image.");
            "examples/empire-state-building.jpg".to_owned()
        }
    };

    let first_image = image::open(&first_image_path)
        .expect("No image found at provided path")
        .to_rgba();
    let second_image = image::open(&second_image_path)
        .expect("No image found at provided path")
        .to_rgba();

    display_multiple_images("", &vec![&first_image, &second_image], 500, 500);
}

#[cfg(not(feature = "display-window"))]
fn main() {
    panic!("Displaying images is only supported if the display-window feature is enabled.");
}
