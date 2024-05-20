use std::env;

use image::open;
use imageproc::filter::{bilateral::GaussianEuclideanColorDistance, bilateral_filter};

fn main() {
    if env::args().len() != 2 {
        panic!("Please enter an input image file path")
    }

    let input_path = env::args().nth(1).unwrap();

    let dynamic_image =
        open(&input_path).unwrap_or_else(|_| panic!("Could not load image at {:?}", input_path));

    let image_grey = dynamic_image.to_luma8();
    let image_color = dynamic_image.to_rgb8();

    let radius = 5;
    let color_sigma = 40.0;
    let spatial_sigma = 140.0;

    let bilateral_grey = bilateral_filter(
        &image_grey,
        radius,
        spatial_sigma,
        GaussianEuclideanColorDistance::new(color_sigma),
    );
    let bilateral_color = bilateral_filter(
        &image_color,
        radius,
        spatial_sigma,
        GaussianEuclideanColorDistance::new(color_sigma),
    );

    bilateral_grey.save("bilateral_grey.png").unwrap();
    bilateral_color.save("bilateral_color.png").unwrap();
}
