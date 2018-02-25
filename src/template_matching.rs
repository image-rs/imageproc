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
                    let template_value = template.get_pixel(dx, dy)[0] as f32;

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
    use utils::gray_bench_image;
    use image::GrayImage;
    use test::{Bencher, black_box};

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_width_does_not_exceed_template_width() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(5, 4));
    }

    #[test]
    #[should_panic]
    fn match_template_panics_if_image_height_does_not_exceed_template_height() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(4, 5));
    }

    #[test]
    fn match_template_accepts_valid_template_size() {
        let _ = match_template(&GrayImage::new(5, 5), &GrayImage::new(4, 4));
    }

    #[test]
    fn match_template_example() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );

        let actual = match_template(&image, &template);
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
                    let result = match_template(&image, &template);
                    black_box(result);
                })
            }
        }
    }

    bench_match_template!(bench_match_template_s100_t1, image_size: 100, template_size: 1);
    bench_match_template!(bench_match_template_s100_t4, image_size: 100, template_size: 4);
    bench_match_template!(bench_match_template_s100_t16, image_size: 100, template_size: 16);
}