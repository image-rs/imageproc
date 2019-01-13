use imageproc::window::display_image;
use std::env;

// run this example from your root directory, use the "--features" flag and
// provide a path to an image file as an argument
// "cargo run --release --features "display-window" --example display_image examples/wrench.jpg" from your root directory
fn main() {
    let img_path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            println!("No image path provided. Using default image."); 
            "examples/wrench.jpg".to_owned()
        }
    };
    let img = image::open(&img_path).expect("no image found at that path").to_rgba();    
    display_image("", &img, 10, 10);
}
