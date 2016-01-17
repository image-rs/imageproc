//! Compares results of image processing functions to existing "truth" images.
//! All test images are taken from the caltech256 dataset.
//! http://authors.library.caltech.edu/7694/

#![feature(test)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

extern crate image;
extern crate test;
#[macro_use]
extern crate imageproc;

use std::path::Path;
use image::ImageBuffer;

use imageproc::utils::{
    load_image_or_panic
};

use imageproc::affine::{
    Interpolation,
    rotate_about_center
};

fn compare_to_truth_rgb(
    input_path: &Path,
    truth_path: &Path,
    op: &Fn(&image::RgbImage) -> image::RgbImage) {

    let truth = load_image_or_panic(&truth_path).to_rgb();
    let input = load_image_or_panic(&input_path).to_rgb();
    let actual = op.call((&input,));

    assert_pixels_eq!(actual, truth);
}

fn compare_to_truth_rgb_with_tolerance(
    input_path: &Path,
    truth_path: &Path,
    op: &Fn(&image::RgbImage) -> image::RgbImage,
    tolerance: u8) {

    let truth = load_image_or_panic(&truth_path).to_rgb();
    let input = load_image_or_panic(&input_path).to_rgb();
    let actual = op.call((&input,));

    assert_pixels_eq_within!(actual, truth, tolerance);
}

fn compare_to_truth_grayscale(
    input_path: &Path,
    truth_path: &Path,
    op: &Fn(&image::GrayImage) -> image::GrayImage) {

    let truth = load_image_or_panic(&truth_path).to_luma();
    let input = load_image_or_panic(&input_path).to_luma();
    let actual = op.call((&input,));

    assert_pixels_eq!(actual, truth);
}

fn rotate_nearest_about_center(image: &image::RgbImage) -> image::RgbImage {
    rotate_about_center(image, std::f32::consts::PI/4f32, Interpolation::Nearest)
}

#[test]
fn test_rotate_nearest_rgb() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_rotate_nearest.png");
    compare_to_truth_rgb(&ip, &tp, &rotate_nearest_about_center);
}

#[test]
fn test_equalize_histogram_grayscale() {
    let ip = Path::new("./tests/data/lumaphant.png");
    let tp = Path::new("./tests/data/truth/lumaphant_eq.png");
    compare_to_truth_grayscale(&ip, &tp, &imageproc::contrast::equalize_histogram);
}

fn rotate_bilinear_about_center(image: &image::RgbImage) -> image::RgbImage {
    rotate_about_center(image, std::f32::consts::PI/4f32, Interpolation::Bilinear)
}

#[test]
fn test_rotate_bilinear_rgb() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_rotate_bilinear.png");
    compare_to_truth_rgb_with_tolerance(&ip, &tp, &rotate_bilinear_about_center, 1);
}

fn affine_nearest(image: &image::RgbImage) -> image::RgbImage {
    let root_two_inv = 1f32/2f32.sqrt();
    let trans = imageproc::math::Affine2::new(
        imageproc::math::Mat2::new(
            root_two_inv, -root_two_inv,
            root_two_inv, root_two_inv) * 2f32,
        imageproc::math::Vec2::new(50f32, -70f32)
    );

    imageproc::affine::affine(
        image, trans,
        imageproc::affine::Interpolation::Nearest).unwrap()
}

#[test]
fn test_affine_nearest_rgb() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_affine_nearest.png");
    compare_to_truth_rgb(&ip, &tp, &affine_nearest);
}

fn affine_bilinear(image: &image::RgbImage) -> image::RgbImage {
    let root_two_inv = 1f32/2f32.sqrt();
    let trans = imageproc::math::Affine2::new(
        imageproc::math::Mat2::new(
            root_two_inv, -root_two_inv,
            root_two_inv, root_two_inv) * 2f32,
        imageproc::math::Vec2::new(50f32, -70f32)
    );

    imageproc::affine::affine(
        image, trans,
        imageproc::affine::Interpolation::Bilinear).unwrap()
}

#[test]
fn test_affine_bilinear_rgb() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_affine_bilinear.png");
    compare_to_truth_rgb(&ip, &tp, &affine_bilinear);
}

fn sobel_gradients(image: &image::GrayImage) -> image::GrayImage {
    use imageproc::definitions::Clamp;
    use imageproc::gradients;
    imageproc::map::map_subpixels(
        &gradients::sobel_gradients(image),
        |x| u8::clamp(x))
}

#[test]
fn test_sobel_gradients() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_gradients.png");
    compare_to_truth_grayscale(&ip, &tp, &sobel_gradients);
}

fn match_to_zebra_histogram(image: &image::GrayImage) -> image::GrayImage {
    let zebra_path = Path::new("./tests/data/zebra.png");
    let zebra = load_image_or_panic(&zebra_path).to_luma();
    imageproc::contrast::match_histogram(image, &zebra)
}

#[test]
fn test_match_histograms() {
    let ip = Path::new("./tests/data/elephant.png");
    let tp = Path::new("./tests/data/truth/elephant_matched.png");
    compare_to_truth_grayscale(&ip, &tp, &match_to_zebra_histogram);
}
