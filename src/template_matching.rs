//! Functions for performing template matching.
use definitions::Image;
use rect::Rect;
use integral_image::{integral_squared_image, sum_image_pixels};
use image::{GenericImage, GrayImage, Luma};

/// Method used to compute the matching score between a template and an image region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel
    /// intensities.
    SumOfSquaredErrors,
    /// Divides the sum computed using `SumOfSquaredErrors` by a normalization term.
    SumOfSquaredErrorsNormalized,
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
pub fn match_template(image: &GrayImage, template: &GrayImage, method: MatchTemplateMethod) -> Image<Luma<f32>> {
    let (image_width, image_height) = image.dimensions();
    let (template_width, template_height) = template.dimensions();

    assert!(image_width > template_width, "image width must strictly exceed template width");
    assert!(image_height > template_height, "image height must strictly exceed template height");

    let should_normalize = method == MatchTemplateMethod::SumOfSquaredErrorsNormalized;
    let image_squared_integral = if should_normalize { Some(integral_squared_image(&image)) } else { None };
    let template_squared_integral = if should_normalize { Some(integral_squared_image(&template)) } else { None };

    let mut result = Image::new(image_width - template_width + 1, image_height - template_height + 1);

    for y in 0..result.height() {
        for x in 0..result.width() {
            let mut score = 0f32;

            for dy in 0..template_height {
                for dx in 0..template_width {
                    let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                    let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };
                    score += (image_value - template_value).powf(2.0);
                }
            }

            if let (&Some(ref i), &Some(ref t)) = (&image_squared_integral, &template_squared_integral) {
                let region = Rect::at(x as i32, y as i32).of_size(template_width, template_height);
                let norm = normalization_term(i, t, region);
                score /= norm;
            }

            result.put_pixel(x, y, Luma([score]));
        }
    }

    result
}

/// Returns the square root of the product of the sum of squares of
/// pixel intensities in template and the provided region of image.
fn normalization_term(
    image_squared_integral: &Image<Luma<u32>>,
    template_squared_integral: &Image<Luma<u32>>,
    region: Rect) -> f32 {
    let image_sum = sum_image_pixels(
        image_squared_integral,
        region.left() as u32,
        region.top() as u32,
        region.right() as u32,
        region.bottom() as u32
    ) as f32;
    let template_sum = sum_image_pixels(
        template_squared_integral,
        0,
        0,
        region.width() - 1,
        region.height() - 1
    ) as f32;

    (image_sum * template_sum).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::gray_bench_image;
    use image::GrayImage;
    use test::{Bencher, black_box};

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_width_does_not_exceed_template_width() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(5, 4), MatchTemplateMethod::SumOfSquaredErrors);
    }

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_height_does_not_exceed_template_height() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(4, 5), MatchTemplateMethod::SumOfSquaredErrors);
    }

    #[test]
    fn match_template_accepts_valid_template_size() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(4, 4), MatchTemplateMethod::SumOfSquaredErrors);
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

    macro_rules! bench_match_template {
        ($name:ident, image_size: $s:expr, template_size: $t:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                b.iter(|| {
                    let result = match_template(&image, &template, MatchTemplateMethod::SumOfSquaredErrors);
                    black_box(result);
                })
            }
        }
    }

    bench_match_template!(bench_match_template_s100_t1, image_size: 100, template_size: 1);
    bench_match_template!(bench_match_template_s100_t4, image_size: 100, template_size: 4);
    bench_match_template!(bench_match_template_s100_t16, image_size: 100, template_size: 16);
}