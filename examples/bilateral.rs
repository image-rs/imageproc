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

    let radius = 16;
    let sigma_color = 40.0;
    let sigma_spatial = 40.0;

    let bilateral_grey = bilateral_filter(
        &image_grey,
        radius,
        sigma_spatial,
        GaussianEuclideanColorDistance {
            sigma_squared: sigma_color,
        },
    );
    let bilateral_color = bilateral_filter(
        &image_color,
        radius,
        sigma_spatial,
        GaussianEuclideanColorDistance {
            sigma_squared: sigma_color,
        },
    );

    bilateral_grey.save("bilateral_grey.png").unwrap();
    bilateral_color.save("bilateral_color.png").unwrap();
}
