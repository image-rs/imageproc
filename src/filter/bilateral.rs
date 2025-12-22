//! Bilateral Filter and associated items.

use image::{GenericImage, Pixel};
use itertools::Itertools;
use num::cast::AsPrimitive;

use crate::definitions::Image;

/// A trait which provides a distance metric between two pixels based on their colors.
///
/// This trait is used with the `bilateral_filter()` function.
pub trait ColorDistance<P> {
    /// Returns a distance measure between the two pixels based on their colors
    fn color_distance(&self, pixel1: &P, pixel2: &P) -> f64;
}

/// A gaussian function of the euclidean distance between two pixel's colors.
///
/// This implements [`ColorDistance`].
pub struct GaussianEuclideanColorDistance {
    sigma_squared: f64,
}
impl GaussianEuclideanColorDistance {
    /// Creates a new [`GaussianEuclideanColorDistance`] using a given sigma value.
    ///
    /// Internally, this is stored as sigma squared for performance.
    pub fn new(sigma: f64) -> Self {
        GaussianEuclideanColorDistance {
            sigma_squared: sigma.powi(2),
        }
    }
}

impl<P> ColorDistance<P> for GaussianEuclideanColorDistance
where
    P: Pixel,
    P::Subpixel: Into<f64>,
{
    fn color_distance(&self, pixel1: &P, pixel2: &P) -> f64 {
        let euclidean_distance_squared = pixel1
            .channels()
            .iter()
            .zip(pixel2.channels().iter())
            .map(|(c1, c2)| ((*c1).into() - (*c2).into()).powi(2))
            .sum::<f64>();

        gaussian_weight(euclidean_distance_squared, self.sigma_squared)
    }
}

/// Denoise an 8-bit image while preserving edges using bilateral filtering.
///
/// # Arguments
///
/// * `image` - Image to be filtered.
/// * `radius` - The radius of the kernel used for the filtering. 0 -> 1x1, 1 -> 3x3, 2 -> 5x5, 3
///     -> 7x7, etc..
/// * `spatial_sigma` - Standard deviation for euclidean spatial distance. A larger value results in
///     averaging of pixels with larger spatial distances.
/// * `color_distance` - A type which implements [`ColorDistance`]. This defines the metric used to
///     define how different two pixels are based on their colors. Common examples may include simple
///     absolute difference for greyscale pixels or cartesian distance in the CIE-Lab color space
///     \[1\].
///
/// This filter averages pixels based on their spatial distance as well as their color
/// distance. Spatial distance is measured by the Gaussian function of the Euclidean distance
/// between two pixels with the user-specified standard deviation (`spatial_sigma`).
///
/// # References
///
///   \[1\] C. Tomasi and R. Manduchi. "Bilateral Filtering for Gray and Color
///        Images." IEEE International Conference on Computer Vision (1998)
///        839-846. DOI: 10.1109/ICCV.1998.710815
///
/// # Panics
///
/// 1. If `image.width() > i32::MAX as u32`
/// 2. If `image.height() > i32::MAX as u32`.
/// 3. If `image.width() == 0`
/// 4. If `image.height() == 0`
///
/// # Examples
///
/// ```
/// use imageproc::filter::bilateral::{bilateral_filter, GaussianEuclideanColorDistance};
/// use imageproc::utils::gray_bench_image;
///
/// let image = gray_bench_image(50, 50);
///
/// let filtered = bilateral_filter(&image, 2, 3., GaussianEuclideanColorDistance::new(10.0));
/// ```
#[must_use = "the function does not modify the original image"]
#[allow(clippy::doc_overindented_list_items)]
pub fn bilateral_filter<I, P, C>(
    image: &I,
    radius: u8,
    spatial_sigma: f64,
    color_distance: C,
) -> Image<P>
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    C: ColorDistance<P>,
    <P as image::Pixel>::Subpixel: 'static,
    f64: From<P::Subpixel> + AsPrimitive<P::Subpixel>,
{
    assert!(!image.width() > i32::MAX as u32);
    assert!(!image.height() > i32::MAX as u32);
    assert_ne!(image.width(), 0);
    assert_ne!(image.height(), 0);

    let radius = i16::from(radius);

    let window_range = -radius..=radius;
    let spatial_distance_lookup = window_range
        .clone()
        .cartesian_product(window_range.clone())
        .map(|(w_y, w_x)| {
            //The gaussian of the euclidean spatial distance
            gaussian_weight(
                <f64 as From<i16>>::from(w_x).powi(2) + <f64 as From<i16>>::from(w_y).powi(2),
                spatial_sigma.powi(2),
            )
        })
        .collect_vec();

    let (width, height) = image.dimensions();
    let bilateral_pixel_filter = |x, y| {
        debug_assert!(image.in_bounds(x, y));
        // Safety: `Image::from_fn` yields `col` in [0, width) and `row` in [0, height).
        let center_pixel = unsafe { image.unsafe_get_pixel(x, y) };

        let window_len = 2 * radius + 1;
        let weights_and_values = window_range
            .clone()
            .cartesian_product(window_range.clone())
            .map(|(w_y, w_x)| {
                // these casts will always be correct due to asserts made at the beginning of the
                // function about the image width/height
                //
                // the subtraction will also never overflow due to the `is_empty()` assert
                let window_y = (i32::from(w_y) + (y as i32)).clamp(0, (height as i32) - 1);
                let window_x = (i32::from(w_x) + (x as i32)).clamp(0, (width as i32) - 1);

                let (window_y, window_x) = (window_y as u32, window_x as u32);

                debug_assert!(image.in_bounds(window_x, window_y));
                let window_pixel = unsafe { image.unsafe_get_pixel(window_x, window_y) };

                let spatial_distance = spatial_distance_lookup
                    [(window_len * (w_y + radius) + (w_x + radius)) as usize];

                let color_distance_val =
                    color_distance.color_distance(&center_pixel, &window_pixel);
                let weight = spatial_distance * color_distance_val;

                (weight, window_pixel)
            });

        weighted_average(weights_and_values)
    };

    Image::from_fn(width, height, bilateral_pixel_filter)
}

fn weighted_average<P>(weights_and_values: impl Iterator<Item = (f64, P)>) -> P
where
    P: Pixel,
    <P as image::Pixel>::Subpixel: 'static,
    f64: From<P::Subpixel> + AsPrimitive<P::Subpixel>,
{
    let (weights_sum, weighted_channel_sums) = weights_and_values
        .map(|(w, v)| {
            (
                w,
                v.channels().iter().map(|s| w * f64::from(*s)).collect_vec(),
            )
        })
        .reduce(|(w1, channels1), (w2, channels2)| {
            (
                w1 + w2,
                channels1
                    .into_iter()
                    .zip_eq(channels2)
                    .map(|(c1, c2)| c1 + c2)
                    .collect_vec(),
            )
        })
        .expect("cannot find a weighted average given no weights and values");

    let channel_averages = weighted_channel_sums.iter().map(|x| x / weights_sum);

    *P::from_slice(
        &channel_averages
            .map(<f64 as AsPrimitive<P::Subpixel>>::as_)
            .collect_vec(),
    )
}

/// Un-normalized Gaussian Weight
fn gaussian_weight(x_squared: f64, sigma_squared: f64) -> f64 {
    (-0.5 * x_squared / sigma_squared).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilateral_filter_greyscale() {
        let image = gray_image!(
            1, 2, 3;
            4, 5, 6;
            7, 8, 9);
        let actual = bilateral_filter(&image, 1, 3.0, GaussianEuclideanColorDistance::new(10.0));

        let expect = gray_image!(
            2, 2, 3;
            4, 5, 5;
            6, 7, 7);

        assert_pixels_eq!(actual, expect);
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::proptest_utils::arbitrary_image;
    use image::Luma;
    use image::Rgb;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_bilateral_filter_greyscale(
            img in arbitrary_image::<Luma<u8>>(1..40, 1..40),
            radius in 0..5u8,
            color_sigma in any::<f64>(),
            spatial_sigma in any::<f64>(),
        ) {
            let out = bilateral_filter(&img, radius, spatial_sigma, GaussianEuclideanColorDistance::new(color_sigma));
            assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_bilateral_filter_rgb(
            img in arbitrary_image::<Rgb<u8>>(1..40, 1..40),
            radius in 0..5u8,
            color_sigma in any::<f64>(),
            spatial_sigma in any::<f64>(),
        ) {
            let out = bilateral_filter(&img, radius, spatial_sigma, GaussianEuclideanColorDistance::new(color_sigma));
            assert_eq!(out.dimensions(), img.dimensions());
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::{gray_bench_image, rgb_bench_image};
    use test::{black_box, Bencher};

    #[bench]
    fn bench_bilateral_filter_greyscale(b: &mut Bencher) {
        let image = gray_bench_image(100, 100);
        b.iter(|| {
            let filtered =
                bilateral_filter(&image, 5, 3., GaussianEuclideanColorDistance::new(10.0));
            black_box(filtered);
        });
    }

    #[bench]
    fn bench_bilateral_filter_rgb(b: &mut Bencher) {
        let image = rgb_bench_image(100, 100);
        b.iter(|| {
            let filtered =
                bilateral_filter(&image, 5, 3., GaussianEuclideanColorDistance::new(10.0));
            black_box(filtered);
        });
    }
}
