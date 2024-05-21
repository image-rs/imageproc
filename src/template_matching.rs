//! Functions for performing template matching.
use crate::definitions::Image;
use image::{GenericImageView, GrayImage, Luma, Primitive};

#[cfg_attr(feature = "katexit", katexit::katexit)]
/// Scoring functions when comparing a template and an image region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchTemplateMethod {
    /// Sum of the squares of the difference between image and template pixel intensities. Smaller values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') - \text{image}(x+x', y+y') \right)^2
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( (\text{template}(x', y') - \text{image}(x+x', y+y')) \cdot \text{mask}(x', y') \right)^2
    /// $$
    ///
    SumOfSquaredErrors,
    /// Divides the sum computed using `SumOfSquaredErrors` by a normalization term. Smaller values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') - \text{image}(x+x', y+y') \right)^2}
    ///                     {\sqrt{ \sum_{x', y'} {\text{template}(x', y')}^2 \cdot \sum_{x', y'} {\text{image}(x+x', y+y')}^2 }}
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( (\text{template}(x', y') - \text{image}(x+x', y+y')) \cdot \text{mask}(x', y') \right)^2}
    ///         {\sqrt{ \sum_{x', y'}{(\text{template}(x', y') \cdot \text{mask}(x', y'))}^2 \cdot \sum_{x', y'}{(\text{image}(x+x', y+y') \cdot \text{mask}(x', y'))}^2 }}
    /// $$
    SumOfSquaredErrorsNormalized,
    /// Cross Correlation. Larger values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \right)
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \cdot {\text{mask}(x', y')}^2 \right)
    /// $$
    ///
    CrossCorrelation,
    /// Divides the sum computed using `CrossCorrelation` by a normalization term. Larger values indicate a better match.
    ///
    /// Without a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \right)}
    ///                     {\sqrt{ \sum_{x', y'} {\text{template}(x', y')}^2 \cdot \sum_{x', y'} {\text{image}(x+x', y+y')}^2 }}
    /// $$
    ///
    /// With a mask:
    /// $$
    /// \text{output}(x, y) = \frac{\sum_{x', y'} \left( \text{template}(x', y') \cdot \text{image}(x+x', y+y') \cdot {\text{mask}(x', y')}^2 \right)}
    ///         {\sqrt{ \sum_{x', y'}{(\text{template}(x', y') \cdot \text{mask}(x', y'))}^2 \cdot \sum_{x', y'}{(\text{image}(x+x', y+y') \cdot \text{mask}(x', y'))}^2 }}
    /// $$
    ///
    CrossCorrelationNormalized,
}

/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
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
    use MatchTemplateMethod as M;

    let input = &ImageTemplate::new(image, template);
    match method {
        M::SumOfSquaredErrors => methods::Sse::match_template(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalized::match_template(input),
        M::CrossCorrelation => methods::Ccorr::match_template(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalized::match_template(input),
    }
}

#[cfg(feature = "rayon")]
/// Slides a `template` over an `image` and scores the match at each point using
/// the requested `method`. This version uses rayon to parallelize the computation.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
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
    use MatchTemplateMethod as M;

    let input = &ImageTemplate::new(image, template);
    match method {
        M::SumOfSquaredErrors => methods::Sse::match_template_parallel(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalized::match_template_parallel(input),
        M::CrossCorrelation => methods::Ccorr::match_template_parallel(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalized::match_template_parallel(input),
    }
}

/// Slides a `template` and a `mask` over an `image` and scores the match at each point using
/// the requested `method`.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
///
/// - If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
/// - If `template.dimensions() != mask.dimensions()`.
pub fn match_template_with_mask(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
    mask: &GrayImage,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplateMask::new(image, template, mask);
    match method {
        M::SumOfSquaredErrors => methods::SseWithMask::match_template(input),
        M::SumOfSquaredErrorsNormalized => methods::SseNormalizedWithMask::match_template(input),
        M::CrossCorrelation => methods::CcorrWithMask::match_template(input),
        M::CrossCorrelationNormalized => methods::CcorrNormalizedWithMask::match_template(input),
    }
}

#[cfg(feature = "rayon")]
/// Slides a `template` and a `mask` over an `image` and scores the match at each point using
/// the requested `method`. This version uses rayon to parallelize the computation.
///
/// The returned image has dimensions `image.width() - template.width() + 1` by
/// `image.height() - template.height() + 1`.
///
/// See [`MatchTemplateMethod`] for details of the matching methods.
///
/// # Panics
/// - If either dimension of `template` is not strictly less than the corresponding dimension
/// of `image`.
/// - If `template.dimensions() != mask.dimensions()`.
pub fn match_template_with_mask_parallel(
    image: &GrayImage,
    template: &GrayImage,
    method: MatchTemplateMethod,
    mask: &GrayImage,
) -> Image<Luma<f32>> {
    use MatchTemplateMethod as M;

    let input = &ImageTemplateMask::new(image, template, mask);
    match method {
        M::SumOfSquaredErrors => methods::SseWithMask::match_template_parallel(input),
        M::SumOfSquaredErrorsNormalized => {
            methods::SseNormalizedWithMask::match_template_parallel(input)
        }
        M::CrossCorrelation => methods::CcorrWithMask::match_template_parallel(input),
        M::CrossCorrelationNormalized => {
            methods::CcorrNormalizedWithMask::match_template_parallel(input)
        }
    }
}

trait MatchTemplate<'a>
where
    Self: Sync + Sized,
{
    type Input: Sync + OutputDims;

    fn init(input: &Self::Input) -> Self;
    fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32;

    fn match_template(input: &Self::Input) -> Image<Luma<f32>> {
        let method = Self::init(input);
        let (width, height) = input.output_dims();

        Image::from_fn(width, height, |x, y| {
            let score = method.score_at((x, y), input);
            Luma([score])
        })
    }
    #[cfg(feature = "rayon")]
    fn match_template_parallel(input: &Self::Input) -> Image<Luma<f32>> {
        use rayon::prelude::*;

        let method = Self::init(input);
        let (width, height) = input.output_dims();

        let rows = (0..height)
            .into_par_iter()
            .map(|y| {
                (0..width)
                    .map(|x| method.score_at((x, y), input))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Image::from_fn(width, height, |x, y| {
            let score = rows[y as usize][x as usize];
            Luma([score])
        })
    }
}

trait OutputDims {
    fn output_dims(&self) -> (u32, u32);
}

mod methods {
    use super::*;

    pub struct Sse;
    impl<'a> MatchTemplate<'a> for Sse {
        type Input = ImageTemplate<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += (t - i).powi(2);
                })
            };
            score
        }
    }

    pub struct SseNormalized {
        template_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for SseNormalized {
        type Input = ImageTemplate<'a>;
        fn init(input: &Self::Input) -> Self {
            Self {
                template_squared_sum: square_sum(input.template),
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut ii = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += (t - i).powi(2);
                    ii += i * i;
                })
            };
            let norm = (ii * self.template_squared_sum).sqrt();
            if norm > 0.0 {
                score / norm
            } else {
                score
            }
        }
    }

    pub struct Ccorr;
    impl<'a> MatchTemplate<'a> for Ccorr {
        type Input = ImageTemplate<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += i * t;
                })
            };
            score
        }
    }

    pub struct CcorrNormalized {
        template_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for CcorrNormalized {
        type Input = ImageTemplate<'a>;
        fn init(input: &Self::Input) -> Self {
            Self {
                template_squared_sum: square_sum(input.template),
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut ii = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t| {
                    score += i * t;
                    ii += i * i;
                })
            };
            let norm = (ii * self.template_squared_sum).sqrt();
            if norm > 0.0 {
                score / norm
            } else {
                score
            }
        }
    }

    pub struct SseWithMask;
    impl<'a> MatchTemplate<'a> for SseWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += ((t - i) * m).powi(2);
                })
            };
            score
        }
    }

    pub struct SseNormalizedWithMask {
        template_mask_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for SseNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(input: &Self::Input) -> Self {
            let template_mask_squared_sum = mult_square_sum(input.inner.template, input.mask);
            Self {
                template_mask_squared_sum,
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut im_im = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += ((t - i) * m).powi(2);
                    im_im += (i * m).powi(2);
                })
            };
            let norm = (self.template_mask_squared_sum * im_im).sqrt();
            if norm > 0.0 {
                score / norm
            } else {
                score
            }
        }
    }
    pub struct CcorrWithMask;
    impl<'a> MatchTemplate<'a> for CcorrWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(_: &Self::Input) -> Self {
            Self
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += t * i * m * m;
                })
            };
            score
        }
    }

    pub struct CcorrNormalizedWithMask {
        template_mask_squared_sum: f32,
    }
    impl<'a> MatchTemplate<'a> for CcorrNormalizedWithMask {
        type Input = ImageTemplateMask<'a>;
        fn init(input: &Self::Input) -> Self {
            let template_mask_squared_sum = mult_square_sum(input.inner.template, input.mask);
            Self {
                template_mask_squared_sum,
            }
        }
        fn score_at(&self, at: (u32, u32), input: &Self::Input) -> f32 {
            let mut score = 0f32;
            let mut im_im = 0f32;
            unsafe {
                input.slide_window_at(at, |i, t, m| {
                    score += t * i * m * m;
                    im_im += (i * m).powi(2);
                })
            };
            let norm = (self.template_mask_squared_sum * im_im).sqrt();
            if norm > 0.0 {
                score / norm
            } else {
                score
            }
        }
    }

    fn square_sum(input: &GrayImage) -> f32 {
        input.iter().map(|&x| (x as f32 * x as f32)).sum()
    }
    fn mult_square_sum(a: &GrayImage, b: &GrayImage) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as f32 * y as f32).powi(2))
            .sum()
    }
}

struct ImageTemplate<'a> {
    image: &'a GrayImage,
    template: &'a GrayImage,
}
impl<'a> ImageTemplate<'a> {
    fn new(image: &'a GrayImage, template: &'a GrayImage) -> Self {
        assert!(
            image.width() >= template.width(),
            "image width must be greater than or equal to template width"
        );
        assert!(
            image.height() >= template.height(),
            "image height must be greater than or equal to template height"
        );
        Self { image, template }
    }
    unsafe fn slide_window_at(&self, (x, y): (u32, u32), mut for_each: impl FnMut(f32, f32)) {
        let (image, template) = (self.image, self.template);
        debug_assert!(x + template.width() - 1 < image.width());
        debug_assert!(y + template.height() - 1 < image.height());

        for dy in 0..template.height() {
            for dx in 0..template.width() {
                let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };
                for_each(image_value, template_value);
            }
        }
    }
}
impl<'a> OutputDims for ImageTemplate<'a> {
    fn output_dims(&self) -> (u32, u32) {
        let width = self.image.width() - self.template.width() + 1;
        let height = self.image.height() - self.template.height() + 1;
        (width, height)
    }
}

struct ImageTemplateMask<'a> {
    inner: ImageTemplate<'a>,
    mask: &'a GrayImage,
}
impl<'a> ImageTemplateMask<'a> {
    fn new(image: &'a GrayImage, template: &'a GrayImage, mask: &'a GrayImage) -> Self {
        assert_eq!(
            template.dimensions(),
            mask.dimensions(),
            "the template and mask must be the same size"
        );
        let inner = ImageTemplate::new(image, template);
        Self { inner, mask }
    }
    unsafe fn slide_window_at(&self, (x, y): (u32, u32), mut for_each: impl FnMut(f32, f32, f32)) {
        let Self { mask, inner } = self;
        let (image, template) = (inner.image, inner.template);
        debug_assert!(x + template.width() - 1 < image.width());
        debug_assert!(y + template.height() - 1 < image.height());

        for dy in 0..template.height() {
            for dx in 0..template.width() {
                let image_value = unsafe { image.unsafe_get_pixel(x + dx, y + dy)[0] as f32 };
                let template_value = unsafe { template.unsafe_get_pixel(dx, dy)[0] as f32 };
                let mask_value = unsafe { mask.unsafe_get_pixel(dx, dy)[0] as f32 };
                for_each(image_value, template_value, mask_value);
            }
        }
    }
}
impl<'a> OutputDims for ImageTemplateMask<'a> {
    fn output_dims(&self) -> (u32, u32) {
        self.inner.output_dims()
    }
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
    fn match_template_sum_of_squared_errors_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            89., 25.;
            10., 1.
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrors,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::SumOfSquaredErrors,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[test]
    fn match_template_sum_of_squared_errors_normalized_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            1.0246822 , 0.19536021;
            0.067865655, 0.005362412
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::SumOfSquaredErrorsNormalized,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::SumOfSquaredErrorsNormalized,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[test]
    fn match_template_cross_correlation_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            68., 124.;
            146., 186.
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelation,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::CrossCorrelation,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
    }

    #[test]
    fn match_template_cross_correlation_normalized_with_mask() {
        let image = gray_image!(
            1, 4, 2;
            2, 1, 3;
            3, 3, 4
        );
        let template = gray_image!(
            1, 2;
            3, 4
        );
        let mask = gray_image!(
            0, 1;
            2, 3
        );
        let expected = gray_image!(type: f32,
            0.78290325, 0.96898663;
            0.9908386, 0.9974086
        );
        let actual = match_template_with_mask(
            &image,
            &template,
            MatchTemplateMethod::CrossCorrelationNormalized,
            &mask,
        );
        assert_pixels_eq!(actual, expected);

        #[cfg(feature = "rayon")]
        {
            let actual_parallel = match_template_with_mask_parallel(
                &image,
                &template,
                MatchTemplateMethod::CrossCorrelationNormalized,
                &mask,
            );
            assert_pixels_eq!(actual_parallel, expected);
        }
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

    macro_rules! bench_match_template_with_mask {
        ($name:ident, image_size: $s:expr, template_size: $t:expr, method: $m:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let image = gray_bench_image($s, $s);
                let template = gray_bench_image($t, $t);
                let mask = gray_bench_image($t, $t);
                b.iter(|| {
                    let result = match_template_with_mask(
                        &image,
                        &template,
                        MatchTemplateMethod::SumOfSquaredErrors,
                        &mask,
                    );
                    black_box(result);
                })
            }
        };
    }

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t1_sse,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t4_sse,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t16_sse,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrors);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t1_sse_norm,
        image_size: 100,
        template_size: 1,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t4_sse_norm,
        image_size: 100,
        template_size: 4,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);

    bench_match_template_with_mask!(
        bench_match_template_with_mask_s100_t16_sse_norm,
        image_size: 100,
        template_size: 16,
        method: MatchTemplateMethod::SumOfSquaredErrorsNormalized);
}
