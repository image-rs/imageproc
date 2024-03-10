//! An example of displaying an image in a window using the display_image function.
//! Run this example from your root directory, enabled the display_image feature and
//! provide a path to an image file as an argument.
//!
//! `cargo run --release --features display-window --example display_image examples/wrench.jpg`

#[cfg(feature = "display-window")]
fn main() {
    use imageproc::window::display_image;
    use std::env;

    let image_path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            println!("No image path provided. Using default image.");
            "examples/wrench.jpg".to_owned()
        }
    };

    let image = image::open(image_path)
        .expect("No image found at provided path")
        .to_rgba8();

    display_image("", &image, 500, 500);
}

#[cfg(not(feature = "display-window"))]
fn main() {
    panic!("Displaying images is only supported if the display-window feature is enabled.");
}
