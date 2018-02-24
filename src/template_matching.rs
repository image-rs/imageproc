//! Functions for performing template matching.
use definitions::Image;
use image::{GrayImage, Luma};

/// Slides a `template` over an image and computes the sum of squared pixel intensity
/// differences at each point.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// # Panics
///
/// If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
pub fn match_template(image: &GrayImage, template: &GrayImage) -> Image<Luma<f32>> {
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(image_width > template_width, "image width must strictly exceed template width");
    assert!(image_height > template_height, "image height must strictly exceed template height");

    let mut result = Image::<Luma<f32>>::new(image_width - template_width + 1, image_height - template_height + 1);

    for y in 0..result.height() {
        for x in 0..result.width() {
            let mut sse = 0f32;

            for dy in 0..template_height {
                for dx in 0..template_width {
                    let image_value = image.get_pixel(x + dx, y + dy)[0] as f32;
                    let template_value = image.get_pixel(dx, dy)[0] as f32;
                    sse += (image_value - template_value).powf(2.0);
                }
            }

            result.put_pixel(x, y, Luma([sse]));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO test panic for invalid dimensions. test does not panic on boundary condition
    // TODO test result dimensions
    // TODO test actual results
}