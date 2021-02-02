//! Compares results of image processing functions to existing "truth" images.
//! All test images are taken from the [caltech256 dataset].
//!
//! To update the truth file for a test, or to generate a truth file for a new test,
//! set the REGENERATE environment variable:
//!
//! ```
//! $ REGENERATE=1 cargo test
//! ```
//!
//! [caltech256 dataset]: http://authors.library.caltech.edu/7694/

#![feature(test)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]

#[macro_use]
extern crate imageproc;

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::{
    definitions::{Clamp, HasBlack, HasWhite},
    edges::canny,
    filter::{gaussian_blur_f32, sharpen3x3},
    geometric_transformations::{rotate_about_center, warp, Interpolation, Projection},
    gradients,
    utils::load_image_or_panic,
};
use std::{env, f32, path::Path};

/// The directory containing the input images used in regression tests.
const INPUT_DIR: &'static str = "./tests/data";

/// The directory containing the truth images to compare against test outputs.
const TRUTH_DIR: &'static str = "./tests/data/truth";

// If the REGENERATE environment variable is set then running tests will update the truth files
// to match the output of the current code.
fn should_regenerate() -> bool {
    env::var("REGENERATE").is_ok()
}

trait FromDynamic {
    fn from_dynamic(image: &DynamicImage) -> Self;
}

impl FromDynamic for GrayImage {
    fn from_dynamic(image: &DynamicImage) -> Self {
        image.to_luma()
    }
}

impl FromDynamic for RgbImage {
    fn from_dynamic(image: &DynamicImage) -> Self {
        image.to_rgb()
    }
}

impl FromDynamic for RgbaImage {
    fn from_dynamic(image: &DynamicImage) -> Self {
        image.to_rgba()
    }
}

/// Loads an input image, applies a function to it and checks that the result matches a 'truth' image.
fn compare_to_truth<P, F>(input_file_name: &str, truth_file_name: &str, op: F)
where
    P: Pixel<Subpixel = u8> + 'static,
    ImageBuffer<P, Vec<u8>>: FromDynamic,
    F: Fn(&ImageBuffer<P, Vec<u8>>) -> ImageBuffer<P, Vec<u8>>,
{
    compare_to_truth_with_tolerance(input_file_name, truth_file_name, op, 0u8);
}

/// Loads an input image, applies a function to it and checks that the result
/// matches a 'truth' image to within a given per-pixel tolerance.
fn compare_to_truth_with_tolerance<P, F>(
    input_file_name: &str,
    truth_file_name: &str,
    op: F,
    tol: u8,
) where
    P: Pixel<Subpixel = u8> + 'static,
    ImageBuffer<P, Vec<u8>>: FromDynamic,
    F: Fn(&ImageBuffer<P, Vec<u8>>) -> ImageBuffer<P, Vec<u8>>,
{
    let input = ImageBuffer::<P, Vec<u8>>::from_dynamic(&load_image_or_panic(
        Path::new(INPUT_DIR).join(input_file_name),
    ));
    let actual = op.call((&input,));
    compare_to_truth_image_with_tolerance(&actual, truth_file_name, tol);
}

/// Checks that an image matches a 'truth' image.
fn compare_to_truth_image<P>(actual: &ImageBuffer<P, Vec<u8>>, truth_file_name: &str)
where
    P: Pixel<Subpixel = u8> + 'static,
    ImageBuffer<P, Vec<u8>>: FromDynamic,
{
    compare_to_truth_image_with_tolerance(actual, truth_file_name, 0u8);
}

/// Checks that an image matches a 'truth' image to within a given per-pixel tolerance.
fn compare_to_truth_image_with_tolerance<P>(
    actual: &ImageBuffer<P, Vec<u8>>,
    truth_file_name: &str,
    tol: u8,
) where
    P: Pixel<Subpixel = u8> + 'static,
    ImageBuffer<P, Vec<u8>>: FromDynamic,
{
    if should_regenerate() {
        actual
            .save(Path::new(TRUTH_DIR).join(truth_file_name))
            .unwrap();
    } else {
        let truth = ImageBuffer::<P, Vec<u8>>::from_dynamic(&load_image_or_panic(
            Path::new(TRUTH_DIR).join(truth_file_name),
        ));
        assert_pixels_eq_within!(*actual, truth, tol);
    }
}

#[test]
fn test_rotate_nearest_rgb() {
    fn rotate_nearest_about_center(image: &RgbImage) -> RgbImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Nearest,
            Rgb::black(),
        )
    }
    compare_to_truth(
        "elephant.png",
        "elephant_rotate_nearest.png",
        rotate_nearest_about_center,
    );
}

#[test]
fn test_rotate_nearest_rgba() {
    fn rotate_nearest_about_center(image: &RgbaImage) -> RgbaImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Nearest,
            Rgba::black(),
        )
    }
    compare_to_truth(
        "elephant_rgba.png",
        "elephant_rotate_nearest_rgba.png",
        rotate_nearest_about_center,
    );
}

#[test]
fn test_equalize_histogram_grayscale() {
    use imageproc::contrast::equalize_histogram;
    compare_to_truth("lumaphant.png", "lumaphant_eq.png", equalize_histogram);
}

#[test]
fn test_rotate_bilinear_rgb() {
    fn rotate_bilinear_about_center(image: &RgbImage) -> RgbImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Bilinear,
            Rgb::black(),
        )
    }
    compare_to_truth_with_tolerance(
        "elephant.png",
        "elephant_rotate_bilinear.png",
        rotate_bilinear_about_center,
        2,
    );
}

#[test]
fn test_rotate_bilinear_rgba() {
    fn rotate_bilinear_about_center(image: &RgbaImage) -> RgbaImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Bilinear,
            Rgba::black(),
        )
    }
    compare_to_truth_with_tolerance(
        "elephant_rgba.png",
        "elephant_rotate_bilinear_rgba.png",
        rotate_bilinear_about_center,
        2,
    );
}

#[test]
fn test_rotate_bicubic_rgb() {
    fn rotate_bicubic_about_center(image: &RgbImage) -> RgbImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Bicubic,
            Rgb::black(),
        )
    }
    compare_to_truth_with_tolerance(
        "elephant.png",
        "elephant_rotate_bicubic.png",
        rotate_bicubic_about_center,
        2,
    );
}

#[test]
fn test_rotate_bicubic_rgba() {
    fn rotate_bicubic_about_center(image: &RgbaImage) -> RgbaImage {
        rotate_about_center(
            image,
            std::f32::consts::PI / 4f32,
            Interpolation::Bicubic,
            Rgba::black(),
        )
    }
    compare_to_truth_with_tolerance(
        "elephant_rgba.png",
        "elephant_rotate_bicubic_rgba.png",
        rotate_bicubic_about_center,
        2,
    );
}

#[test]
fn test_affine_nearest_rgb() {
    fn affine_nearest(image: &RgbImage) -> RgbImage {
        let root_two_inv = 1f32 / 2f32.sqrt() * 2.0;
        #[rustfmt::skip]
        let hom = Projection::from_matrix([
            root_two_inv, -root_two_inv,  50.0,
            root_two_inv,  root_two_inv, -70.0,
                     0.0,           0.0,   1.0,
        ])
        .unwrap();
        warp(image, &hom, Interpolation::Nearest, Rgb::black())
    }
    compare_to_truth(
        "elephant.png",
        "elephant_affine_nearest.png",
        affine_nearest,
    );
}

#[test]
fn test_affine_bilinear_rgb() {
    fn affine_bilinear(image: &RgbImage) -> RgbImage {
        let root_two_inv = 1f32 / 2f32.sqrt() * 2.0;
        #[rustfmt::skip]
        let hom = Projection::from_matrix([
            root_two_inv, -root_two_inv,  50.0,
            root_two_inv,  root_two_inv, -70.0,
                     0.0,           0.0,   1.0,
        ])
        .unwrap();

        warp(image, &hom, Interpolation::Bilinear, Rgb::black())
    }
    compare_to_truth_with_tolerance(
        "elephant.png",
        "elephant_affine_bilinear.png",
        affine_bilinear,
        1,
    );
}

#[test]
fn test_affine_bicubic_rgb() {
    fn affine_bilinear(image: &RgbImage) -> RgbImage {
        let root_two_inv = 1f32 / 2f32.sqrt() * 2.0;
        #[rustfmt::skip]
        let hom = Projection::from_matrix([
            root_two_inv, -root_two_inv,  50.0,
            root_two_inv,  root_two_inv, -70.0,
            0.0         , 0.0          , 1.0,
        ]).unwrap();

        warp(image, &hom, Interpolation::Bicubic, Rgb::black())
    }
    compare_to_truth_with_tolerance(
        "elephant.png",
        "elephant_affine_bicubic.png",
        affine_bilinear,
        1,
    );
}

#[test]
fn test_sobel_gradients() {
    fn sobel_gradients(image: &GrayImage) -> GrayImage {
        imageproc::map::map_subpixels(
            &gradients::sobel_gradients(image),
            <u8 as Clamp<u16>>::clamp,
        )
    }
    compare_to_truth("elephant.png", "elephant_gradients.png", sobel_gradients);
}

#[test]
fn test_sharpen3x3() {
    compare_to_truth("robin.png", "robin_sharpen3x3.png", sharpen3x3);
}

#[test]
fn test_sharpen_gaussian() {
    fn sharpen(image: &GrayImage) -> GrayImage {
        imageproc::filter::sharpen_gaussian(image, 0.7, 7.0)
    }
    compare_to_truth("robin.png", "robin_sharpen_gaussian.png", sharpen);
}

#[test]
fn test_match_histograms() {
    fn match_to_zebra_histogram(image: &GrayImage) -> GrayImage {
        let zebra = load_image_or_panic(Path::new(INPUT_DIR).join("zebra.png")).to_luma();
        imageproc::contrast::match_histogram(image, &zebra)
    }
    compare_to_truth(
        "elephant.png",
        "elephant_matched.png",
        match_to_zebra_histogram,
    );
}

#[test]
fn test_canny() {
    compare_to_truth("zebra.png", "zebra_canny.png", |image| {
        canny(image, 250.0, 300.0)
    });
}

#[test]
fn test_gaussian_blur_stdev_3() {
    compare_to_truth("zebra.png", "zebra_gaussian_3.png", |image: &GrayImage| {
        gaussian_blur_f32(image, 3f32)
    });
}

#[test]
fn test_gaussian_blur_stdev_10() {
    compare_to_truth("zebra.png", "zebra_gaussian_10.png", |image: &GrayImage| {
        gaussian_blur_f32(image, 10f32)
    });
}

#[test]
fn test_adaptive_threshold() {
    use imageproc::contrast::adaptive_threshold;
    compare_to_truth("zebra.png", "zebra_adaptive_threshold.png", |image| {
        adaptive_threshold(image, 41)
    });
}

#[test]
fn test_otsu_threshold() {
    use imageproc::contrast::{otsu_level, threshold};
    fn otsu_threshold(image: &GrayImage) -> GrayImage {
        let level = otsu_level(image);
        threshold(image, level)
    }
    compare_to_truth("zebra.png", "zebra_otsu.png", otsu_threshold);
}

#[test]
fn test_draw_antialiased_line_segment_rgb() {
    use image::Rgb;
    use imageproc::drawing::draw_antialiased_line_segment_mut;
    use imageproc::pixelops::interpolate;

    let blue = Rgb([0, 0, 255]);
    let mut image = RgbImage::from_pixel(200, 200, blue);

    let white = Rgb([255, 255, 255]);
    // Connected path:
    //      - horizontal
    draw_antialiased_line_segment_mut(&mut image, (20, 80), (40, 80), white, interpolate);
    //      - shallow ascent
    draw_antialiased_line_segment_mut(&mut image, (40, 80), (60, 70), white, interpolate);
    //      - diagonal ascent
    draw_antialiased_line_segment_mut(&mut image, (60, 70), (70, 70), white, interpolate);
    //      - steep ascent
    draw_antialiased_line_segment_mut(&mut image, (70, 70), (80, 30), white, interpolate);
    //      - shallow descent
    draw_antialiased_line_segment_mut(&mut image, (80, 30), (110, 45), white, interpolate);
    //      - diagonal descent
    draw_antialiased_line_segment_mut(&mut image, (110, 45), (130, 65), white, interpolate);
    //      - steep descent
    draw_antialiased_line_segment_mut(&mut image, (130, 65), (150, 110), white, interpolate);
    //      - vertical
    draw_antialiased_line_segment_mut(&mut image, (150, 110), (150, 140), white, interpolate);

    // Isolated segment, partially outside of image bounds
    draw_antialiased_line_segment_mut(&mut image, (150, 150), (210, 130), white, interpolate);

    compare_to_truth_image(&image, "antialiased_lines_rgb.png");
}

#[test]
fn test_draw_polygon() {
    use imageproc::drawing::draw_polygon_mut;
    use imageproc::point::Point;

    let mut image = GrayImage::from_pixel(300, 300, Luma::black());
    let white = Luma::white();

    let star = vec![
        Point::new(100, 20),
        Point::new(120, 35),
        Point::new(140, 30),
        Point::new(115, 45),
        Point::new(130, 60),
        Point::new(100, 50),
        Point::new(80, 55),
        Point::new(90, 40),
        Point::new(60, 25),
        Point::new(90, 35),
    ];
    draw_polygon_mut(&mut image, &star, white);

    let partially_out_of_bounds_star = vec![
        Point::new(275, 20),
        Point::new(295, 35),
        Point::new(315, 30),
        Point::new(290, 45),
        Point::new(305, 60),
        Point::new(275, 50),
        Point::new(255, 55),
        Point::new(265, 40),
        Point::new(235, 25),
        Point::new(265, 35),
    ];
    draw_polygon_mut(&mut image, &partially_out_of_bounds_star, white);

    let triangle = vec![Point::new(35, 80), Point::new(145, 110), Point::new(5, 90)];
    draw_polygon_mut(&mut image, &triangle, white);

    let partially_out_of_bounds_triangle = vec![
        Point::new(250, 80),
        Point::new(350, 130),
        Point::new(250, 120),
    ];
    draw_polygon_mut(&mut image, &partially_out_of_bounds_triangle, white);

    let quad = vec![
        Point::new(190, 250),
        Point::new(240, 210),
        Point::new(270, 200),
        Point::new(220, 280),
    ];
    draw_polygon_mut(&mut image, &quad, white);

    let hex: Vec<Point<i32>> = (0..6)
        .map(|i| i as f32 * f32::consts::PI / 3f32)
        .map(|theta| (theta.cos() * 50.0 + 75.0, theta.sin() * 50.0 + 225.0))
        .map(|(x, y)| Point::new(x as i32, y as i32))
        .collect();
    draw_polygon_mut(&mut image, &hex, white);

    compare_to_truth_image(&image, "polygon.png");
}

#[test]
fn test_draw_spiral_polygon() {
    use imageproc::drawing::draw_polygon_mut;
    use imageproc::point::Point;

    let mut image = GrayImage::from_pixel(100, 100, Luma::black());

    let polygon = vec![
        Point::new(20, 20),
        Point::new(80, 20),
        Point::new(80, 70),
        Point::new(20, 70),
        Point::new(20, 40),
        Point::new(60, 40),
        Point::new(60, 50),
        Point::new(30, 50),
        Point::new(30, 60),
        Point::new(70, 60),
        Point::new(70, 30),
        Point::new(20, 30),
    ];
    draw_polygon_mut(&mut image, &polygon, Luma::white());

    compare_to_truth_image(&image, "spiral_polygon.png");
}

#[test]
fn test_draw_cubic_bezier_curve() {
    use imageproc::drawing::draw_cubic_bezier_curve_mut;

    let red = Rgb([255, 0, 0]);
    let green = Rgb([0, 255, 0]);
    let blue = Rgb([0, 0, 255]);
    let mut image = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));

    // Straight line
    draw_cubic_bezier_curve_mut(
        &mut image,
        (0.0, 100.0),
        (200.0, 100.0),
        (-10.0, 100.0),
        (210.0, 100.0),
        red,
    );
    // Straight line off screen
    draw_cubic_bezier_curve_mut(
        &mut image,
        (30.0, -30.0),
        (40.0, 250.0),
        (30.0, -30.0),
        (40.0, 250.0),
        red,
    );
    // Basic curve horizontal
    draw_cubic_bezier_curve_mut(
        &mut image,
        (20.0, 150.0),
        (180.0, 150.0),
        (100.0, 100.0),
        (150.0, 80.0),
        blue,
    );
    // Curve with inflection
    draw_cubic_bezier_curve_mut(
        &mut image,
        (100.0, 0.0),
        (120.0, 200.0),
        (300.0, 20.0),
        (-100.0, 150.0),
        green,
    );
    // Curve that makes a lopsided loop
    draw_cubic_bezier_curve_mut(
        &mut image,
        (150.0, 50.0),
        (150.0, 50.0),
        (200.0, 20.0),
        (50.0, 20.0),
        red,
    );

    compare_to_truth_image(&image, "cubic_bezier_curve.png");
}

#[test]
fn test_draw_hollow_ellipse() {
    use imageproc::drawing::draw_hollow_ellipse_mut;

    let red = Rgb([255, 0, 0]);
    let green = Rgb([0, 255, 0]);
    let blue = Rgb([0, 0, 255]);
    let mut image = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));

    // Circle
    draw_hollow_ellipse_mut(&mut image, (100, 100), 50, 50, red);
    // Vertically stretched
    draw_hollow_ellipse_mut(&mut image, (50, 100), 40, 90, blue);
    // Horizontally stretched
    draw_hollow_ellipse_mut(&mut image, (100, 150), 80, 30, green);
    // Partially off-screen
    draw_hollow_ellipse_mut(&mut image, (150, 150), 100, 60, blue);

    compare_to_truth_image(&image, "hollow_ellipse.png");
}

#[test]
fn test_draw_filled_ellipse() {
    use imageproc::drawing::draw_filled_ellipse_mut;

    let red = Rgb([255, 0, 0]);
    let green = Rgb([0, 255, 0]);
    let blue = Rgb([0, 0, 255]);
    let mut image = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));

    // Circle
    draw_filled_ellipse_mut(&mut image, (100, 100), 50, 50, red);
    // Vertically stretched
    draw_filled_ellipse_mut(&mut image, (50, 100), 40, 90, blue);
    // Horizontally stretched
    draw_filled_ellipse_mut(&mut image, (100, 150), 80, 30, green);
    // Partially off-screen
    draw_filled_ellipse_mut(&mut image, (150, 150), 100, 60, blue);

    compare_to_truth_image(&image, "filled_ellipse.png");
}

#[test]
fn test_hough_line_detection() {
    use imageproc::hough::{detect_lines, draw_polar_lines, LineDetectionOptions, PolarLine};
    use imageproc::map::map_colors;

    let white = Rgb([255u8, 255u8, 255u8]);
    let black = Rgb([0u8, 0u8, 0u8]);
    let green = Rgb([0u8, 255u8, 0u8]);

    let image = GrayImage::new(100, 100);
    let image = draw_polar_lines(
        &image,
        &vec![
            PolarLine {
                r: 50.0,
                angle_in_degrees: 0,
            },
            PolarLine {
                r: 50.0,
                angle_in_degrees: 45,
            },
            PolarLine {
                r: 50.0,
                angle_in_degrees: 90,
            },
            PolarLine {
                r: -10.0,
                angle_in_degrees: 120,
            },
            PolarLine {
                r: 0.01,
                angle_in_degrees: 135,
            },
        ],
        Luma([255u8]),
    );
    let options = LineDetectionOptions {
        vote_threshold: 40,
        suppression_radius: 8,
    };
    let lines: Vec<PolarLine> = detect_lines(&image, options);
    let color_edges = map_colors(&image, |p| if p[0] > 0 { white } else { black });

    // Draw detected lines on top of original image
    let lines_image = draw_polar_lines(&color_edges, &lines, green);

    compare_to_truth_image(&lines_image, "hough_lines.png");
}
