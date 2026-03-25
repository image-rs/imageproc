//! Bilateral Filter and associated items.

use image::{GenericImage, Pixel};
use num::cast::AsPrimitive;

use crate::definitions::Image;

/// A trait which provides a distance metric between two pixels based on their colors.
///
/// This trait is used with the `bilateral_filter()` function.
pub trait ColorDistance<P> {
    /// Returns a distance measure between the two pixels based on their colors
    fn color_distance(&self, pixel1: &P, pixel2: &P) -> f32;
}

/// A gaussian function of the euclidean distance between two pixel's colors.
///
/// This implements [`ColorDistance`].
pub struct GaussianEuclideanColorDistance {
    sigma_squared: f32,
}
impl GaussianEuclideanColorDistance {
    /// Creates a new [`GaussianEuclideanColorDistance`] using a given sigma value.
    ///
    /// Internally, this is stored as sigma squared for performance.
    pub fn new(sigma: f32) -> Self {
        GaussianEuclideanColorDistance {
            sigma_squared: sigma.powi(2),
        }
    }
}

impl<P> ColorDistance<P> for GaussianEuclideanColorDistance
where
    P: Pixel,
    f32: From<P::Subpixel>,
{
    fn color_distance(&self, pixel1: &P, pixel2: &P) -> f32 {
        let euclidean_distance_squared = pixel1
            .channels()
            .iter()
            .zip(pixel2.channels().iter())
            .map(|(c1, c2)| (f32::from(*c1) - f32::from(*c2)).powi(2))
            .sum::<f32>();

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
    spatial_sigma: f32,
    color_distance: C,
) -> Image<P>
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
    C: ColorDistance<P>,
    <P as image::Pixel>::Subpixel: 'static,
    f32: From<P::Subpixel> + AsPrimitive<P::Subpixel>,
{
    assert!(!image.width() > i32::MAX as u32);
    assert!(!image.height() > i32::MAX as u32);
    assert_ne!(image.width(), 0);
    assert_ne!(image.height(), 0);

    let radius = i16::from(radius);

    let spatial_sigma_squared = spatial_sigma.powi(2);
    let mut spatial_distance_lookup =
        Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
    for w_y in -radius..=radius {
        for w_x in -radius..=radius {
            spatial_distance_lookup.push(gaussian_weight(
                (w_x as f32).powi(2) + (w_y as f32).powi(2),
                spatial_sigma_squared,
            ));
        }
    }

    let (width, height) = image.dimensions();
    let window_len = 2 * radius + 1;

    let bilateral_pixel_filter = |x, y| {
        debug_assert!(image.in_bounds(x, y));
        // Safety: `Image::from_fn` yields `col` in [0, width) and `row` in [0, height).
        let center_pixel = unsafe { image.unsafe_get_pixel(x, y) };

        let mut channel_sums = [0f32; 4];
        let mut weight_sum = 0f32;

        for w_y in -radius..=radius {
            for w_x in -radius..=radius {
                // these casts will always be correct due to asserts made at the beginning of the
                // function about the image width/height
                //
                // the subtraction will also never overflow due to the `is_empty()` assert
                let window_y = (i32::from(w_y) + (y as i32)).clamp(0, (height as i32) - 1);
                let window_x = (i32::from(w_x) + (x as i32)).clamp(0, (width as i32) - 1);

                let (window_y, window_x) = (window_y as u32, window_x as u32);

                debug_assert!(image.in_bounds(window_x, window_y));
                // Safety: we clamped `window_x` and `window_y` to be in bounds.
                let window_pixel = unsafe { image.unsafe_get_pixel(window_x, window_y) };

                let spatial_weight = spatial_distance_lookup
                    [(window_len * (w_y + radius) + (w_x + radius)) as usize];
                let color_weight = color_distance.color_distance(&center_pixel, &window_pixel);
                let weight = spatial_weight * color_weight;

                weight_sum += weight;
                for (i, c) in window_pixel.channels().iter().enumerate() {
                    channel_sums[i] += weight * f32::from(*c);
                }
            }
        }

        let mut out_pixel = center_pixel;
        let num_channels = P::CHANNEL_COUNT as usize;
        let out_channels = out_pixel.channels_mut();
        for i in 0..num_channels {
            out_channels[i] = (channel_sums[i] / weight_sum).as_();
        }
        out_pixel
    };

    Image::from_fn(width, height, bilateral_pixel_filter)
}

/// Un-normalized Gaussian Weight
fn gaussian_weight(x_squared: f32, sigma_squared: f32) -> f32 {
    (-0.5 * x_squared / sigma_squared).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(miri, ignore = "assert_pixels_eq fails")]
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
            color_sigma in any::<f32>(),
            spatial_sigma in any::<f32>(),
        ) {
            let out = bilateral_filter(&img, radius, spatial_sigma, GaussianEuclideanColorDistance::new(color_sigma));
            prop_assert_eq!(out.dimensions(), img.dimensions());
        }

        #[test]
        fn proptest_bilateral_filter_rgb(
            img in arbitrary_image::<Rgb<u8>>(1..40, 1..40),
            radius in 0..5u8,
            color_sigma in any::<f32>(),
            spatial_sigma in any::<f32>(),
        ) {
            let out = bilateral_filter(&img, radius, spatial_sigma, GaussianEuclideanColorDistance::new(color_sigma));
            prop_assert_eq!(out.dimensions(), img.dimensions());
        }
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use crate::utils::{gray_bench_image, rgb_bench_image};
    use test::{Bencher, black_box};

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
