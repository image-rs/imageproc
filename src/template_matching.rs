//! Functions for performing template matching.
use crate::definitions::Image;
use crate::integral_image::{integral_squared_image, sum_image_pixels};
use crate::rect::Rect;
use image::Primitive;
use image::{GrayImage, Luma};

/// Method used to compute the matching score between a template and an image region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel
    /// intensities.
    ///
    /// Smaller values are better.
    SumOfSquaredErrors,
    /// Divides the sum computed using `SumOfSquaredErrors` by a normalization term.
    SumOfSquaredErrorsNormalized,
    /// Cross Correlation
    ///
    /// Higher values are better.
    CrossCorrelation,
    /// Divides the sum computed using `CrossCorrelation` by a normalization term.
    CrossCorrelationNormalized,
}

/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// # Panics
///
/// If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
pub fn match_template(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>> {
    use image::GenericImageView;

    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(
        image_width >= template_width,
        "image width must be greater than or equal to template width"
    );
    assert!(
        image_height >= template_height,
        "image height must be greater than or equal to template height"
    );

    let should_normalize = matches! { method,
    MatchTemplateMethod::SumOfSquaredErrorsNormalized
    | MatchTemplateMethod::CrossCorrelationNormalized };
    let image_squared_integral = if should_normalize {
        Some(integral_squared_image(image))
    } else {
        None
    };
    let template_squared_sum = if should_normalize {
        Some(sum_squares(template))
    } else {
        None
    };

    let mut result = Image::new(
        image_width - template_width + 1,
        image_height - template_height + 1,
    );

    for y in 0..result.height() {
        for x in 0..result.width() {
            let mut score = 0f32;

            for dy in 0..template_height {
                for dx in 0..template_width {
                    let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                    let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };

                    use MatchTemplateMethod::*;

                    score += match method {
                        SumOfSquaredErrors | SumOfSquaredErrorsNormalized => {
                            (image_value - template_value).powf(2.0)
                        }
                        CrossCorrelation | CrossCorrelationNormalized => {
                            image_value * template_value
                        }
                    };
                }
            }

            if let (Some(i), &Some(t)) = (&image_squared_integral, &template_squared_sum) {
                let region = Rect::at(x as i32, y as i32).of_size(template_width, template_height);
                let norm = normalization_term(i, t, region);
                if norm > 0.0 {
                    score /= norm;
                }
            }

            result.put_pixel(x, y, Luma([score]));
        }
    }

    result
}

#[cfg(feature = "rayon")]
/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`. This version uses rayon to parallelize the computation.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// # Panics
///
/// If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
pub fn match_template_parallel(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
) -> Image<Luma<f32>> {
    use image::GenericImageView;
    use rayon::prelude::*;

    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(
        image_width >= template_width,
        "image width must be greater than or equal to template width"
    );
    assert!(
        image_height >= template_height,
        "image height must be greater than or equal to template height"
    );

    let should_normalize = matches! { method,
    MatchTemplateMethod::SumOfSquaredErrorsNormalized
    | MatchTemplateMethod::CrossCorrelationNormalized };
    let image_squared_integral = if should_normalize {
        Some(integral_squared_image(image))
    } else {
        None
    };
    let template_squared_sum = if should_normalize {
        Some(sum_squares(template))
    } else {
        None
    };

    let result_width = image_width - template_width + 1;
    let result_height = image_height - template_height + 1;

    let rows = (0..result_height)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(result_width as usize);
            for x in 0..result_width {
                let mut score = 0f32;

                for dy in 0..template_height {
                    for dx in 0..template_width {
                        let image_value =
                            unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                        let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };

                        use MatchTemplateMethod::*;

                        score += match method {
                            SumOfSquaredErrors | SumOfSquaredErrorsNormalized => {
                                (image_value - template_value).powf(2.0)
                            }
                            CrossCorrelation | CrossCorrelationNormalized => {
                                image_value * template_value
                            }
                        };
                    }
                }

                if let (Some(i), &Some(t)) = (&image_squared_integral, &template_squared_sum) {
                    let region =
                        Rect::at(x as i32, y as i32).of_size(template_width, template_height);
                    let norm = normalization_term(i, t, region);
                    if norm > 0.0 {
                        score /= norm;
                    }
                }

                row.push(Luma([score]));
            }
            row
        })
        .collect::<Vec<_>>();

    Image::from_fn(result_width, result_height, |x, y| {
        rows[y as usize][x as usize]
    })
}

fn sum_squares(template: &GrayImage) -> f32 {
    template.iter().map(|p| *p as f32 * *p as f32).sum()
}

/// Returns the square root of the product of the sum of squares of
/// pixel intensities in template and the provided region of image.
fn normalization_term(
    image_squared_integral: &Image<Luma<u64>>,
    template_squared_sum: f32,
    region: Rect,
) -> f32 {
    let image_sum = sum_image_pixels(
        image_squared_integral,
        region.left() as u32,
        region.top() as u32,
        region.right() as u32,
        region.bottom() as u32,
    )[0] as f32;
    (image_sum * template_squared_sum).sqrt()
}

/// The largest and smallest values in an image,
/// together with their locations.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Extremes<T> {
    /// The largest value in an image.
    pub max_value: T,
    /// The smallest value in an image.
    pub min_value: T,
    /// The coordinates of the largest value in an image.
    pub max_value_location: (u32, u32),
    /// The coordinates of the smallest value in an image.
    pub min_value_location: (u32, u32),
}

/// Finds the largest and smallest values in an image and their locations.
/// If there are multiple such values then the lexicographically smallest is returned.
pub fn find_extremes<T>(image: &Image<Luma<T>>) -> Extremes<T>
where
    T: Primitive,
{
    assert!(
        image.width() > 0 && image.height() > 0,
        "image must be non-empty"
    );

    let mut min_value = image.get_pixel(0, 0)[0];
    let mut max_value = image.get_pixel(0, 0)[0];

    let mut min_value_location = (0, 0);
    let mut max_value_location = (0, 0);

    for (x, y, p) in image.enumerate_pixels() {
        if p[0] < min_value {
            min_value = p[0];
            min_value_location = (x, y);
        }
        if p[0] > max_value {
            max_value = p[0];
            max_value_location = (x, y);
        }
    }

    Extremes {
        max_value,
        min_value,
        max_value_location,
        min_value_location,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_width_does_is_less_than_template_width() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(6, 5),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_height_is_less_than_template_height() {
        let _ = match_template(
            &GrayImage::new(5, 5),
            &GrayImage::new(5, 6),
            MatchTemplateMethod::SumOfSquaredErrors,
        );
    }

    #[test]
    fn match_template_handles_template_of_same_size_as_image() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(5, 5),
                &GrayImage::new(5, 5),
                MatchTemplateMethod::SumOfSquaredErrors
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[test]
    fn match_template_normalization_handles_zero_norm() {
        assert_pixels_eq!(
            match_template(
                &GrayImage::new(1, 1),
                &GrayImage::new(1, 1),
                MatchTemplateMethod::SumOfSquaredErrorsNormalized
            ),
            gray_image!(type: f32, 0.0)
        );
    }

    #[test]
    fn match_template_sum_of_squared_errors() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
        let expected = gray_image!(type: f32,
            14.0, 14.0;
            3.0, 1.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_sum_of_squared_errors_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            14.0 / (22.0 * tss).sqrt(), 14.0 / (30.0 * tss).sqrt();
            3.0 / (23.0 * tss).sqrt(), 1.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_cross_correlation() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template, MatchTemplateMethod::CrossCorrelation);
        let expected = gray_image!(type: f32,
            19.0, 23.0;
            25.0, 32.0
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn match_template_cross_correlation_normalized() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
        );
        let tss = 30f32;
        let expected = gray_image!(type: f32,
            19.0 / (22.0 * tss).sqrt(), 23.0 / (30.0 * tss).sqrt();
            25.0 / (23.0 * tss).sqrt(), 32.0 / (35.0 * tss).sqrt()
        );

        assert_pixels_eq!(actual, expected);
    }

    #[test]
    fn test_find_extremes() {
        let image = gray_image!(
            10,  7,  8,  1;
             9, 15,  4,  2
        );

        let expected = Extremes {
            max_value: 15,
            min_value: 1,
            max_value_location: (1, 1),
            min_value_location: (3, 0),
        };

        assert_eq!(find_extremes(&image), expected);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::gray_bench_image;
    use test::{black_box, Bencher};

    macro_rules! bench_match_template {
        ($name:ident, image_size: $s:expr, template_size: $t:expr, method: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                b.iter(|| {
                    let result =
                        match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
                    black_box(result);
                })
            }
        };
    }

    bench_match_template!(
        bench_match_template_s100_t1_sse,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t4_sse,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t16_sse,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template!(
        bench_match_template_s100_t1_sse_norm,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t4_sse_norm,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template!(
        bench_match_template_s100_t16_sse_norm,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);
}
