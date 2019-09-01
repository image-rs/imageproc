use super::{filter3x3, gaussian_blur_f32};
use crate::{
    definitions::{Clamp, Image},
    map::{map_colors2, map_subpixels},
};
use image::{GrayImage, Luma};

/// Sharpens a grayscale image by applying a 3x3 approximation to the Laplacian.
pub fn sharpen3x3(image: &GrayImage) -> GrayImage {
    let identity_minus_laplacian = [0, -1, 0, -1, 5, -1, 0, -1, 0];
    filter3x3(image, &identity_minus_laplacian)
}

/// Sharpens a grayscale image using a Gaussian as a low-pass filter.
///
/// * `sigma` is the standard deviation of the Gaussian filter used.
/// * `amount` controls the level of sharpening. `output = input + amount * edges`.
// TODO: remove unnecessary allocations, support colour images
pub fn sharpen_gaussian(image: &GrayImage, sigma: f32, amount: f32) -> GrayImage {
    let image = map_subpixels(image, |x| x as f32);
    let smooth: Image<Luma<f32>> = gaussian_blur_f32(&image, sigma);
    map_colors2(&image, &smooth, |p, q| {
        let v = (1.0 + amount) * p[0] - amount * q[0];
        Luma([<u8 as Clamp<f32>>::clamp(v)])
    })
}
