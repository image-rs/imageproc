//! Functions for adding synthetic noise to images.

use crate::definitions::{Clamp, HasBlack, HasWhite, Image};
use image::Pixel;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Uniform};

/// Adds independent additive Gaussian noise to all channels
/// of an image, with the given mean and standard deviation.
pub fn gaussian_noise<P>(image: &Image<P>, mean: f64, stddev: f64, seed: u64) -> Image<P>
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut out = image.clone();
    gaussian_noise_mut(&mut out, mean, stddev, seed);
    out
}
#[doc=generate_mut_doc_comment!("gaussian_noise")]
pub fn gaussian_noise_mut<P>(image: &mut Image<P>, mean: f64, stddev: f64, seed: u64)
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let normal = Normal::new(mean, stddev).unwrap();

    for p in image.pixels_mut() {
        for c in p.channels_mut() {
            let noise = normal.sample(&mut rng);
            *c = P::Subpixel::clamp((*c).into() + noise);
        }
    }
}

/// Adds multiplicative speckle noise to an image with the given mean and standard deviation.
/// Noise is added per pixel for realistic sensor noise simulation.
pub fn speckle_noise<P>(image: &Image<P>, mean: f64, stddev: f64, seed: u64) -> Image<P>
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut out = image.clone();
    speckle_noise_mut(&mut out, mean, stddev, seed);
    out
}

#[doc=generate_mut_doc_comment!("speckle_noise")]
pub fn speckle_noise_mut<P>(image: &mut Image<P>, mean: f64, stddev: f64, seed: u64)
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let normal = Normal::new(mean, stddev).unwrap();

    // Use the same noise pattern for each channel
    for p in image.pixels_mut() {
        let noise = normal.sample(&mut rng);
        p.apply(|c| {
            let original = c.into();
            P::Subpixel::clamp(original + original * noise)
        });
    }
}

/// Adds multiplicative speckle noise to an image with the given mean and standard deviation.
/// Noise is added independently per channel for data augmentation.
pub fn speckle_noise_per_channel<P>(image: &Image<P>, mean: f64, stddev: f64, seed: u64) -> Image<P>
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut out = image.clone();
    speckle_noise_per_channel_mut(&mut out, mean, stddev, seed);
    out
}

#[doc=generate_mut_doc_comment!("speckle_noise_per_channel")]
pub fn speckle_noise_per_channel_mut<P>(image: &mut Image<P>, mean: f64, stddev: f64, seed: u64)
where
    P: Pixel,
    P::Subpixel: Into<f64> + Clamp<f64>,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let normal = Normal::new(mean, stddev).unwrap();

    // Add sampled noise per channel in each pixel
    for p in image.pixels_mut() {
        p.apply(|c| {
            let noise = normal.sample(&mut rng);
            let original = c.into();
            P::Subpixel::clamp(original + original * noise)
        });
    }
}

/// Converts pixels to black or white at the given `rate` (between 0.0 and 1.0).
/// Black and white occur with equal probability.
pub fn salt_and_pepper_noise<P>(image: &Image<P>, rate: f64, seed: u64) -> Image<P>
where
    P: Pixel + HasBlack + HasWhite,
{
    let mut out = image.clone();
    salt_and_pepper_noise_mut(&mut out, rate, seed);
    out
}
#[doc=generate_mut_doc_comment!("salt_and_pepper_noise")]
pub fn salt_and_pepper_noise_mut<P>(image: &mut Image<P>, rate: f64, seed: u64)
where
    P: Pixel + HasBlack + HasWhite,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, 1.0).unwrap();

    for p in image.pixels_mut() {
        if uniform.sample(&mut rng) > rate {
            continue;
        }
        let r = uniform.sample(&mut rng);
        *p = if r >= 0.5 { P::white() } else { P::black() };
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use super::*;
    use image::GrayImage;
    use test::{Bencher, black_box};

    #[bench]
    fn bench_gaussian_noise_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| {
            gaussian_noise_mut(&mut image, 30.0, 40.0, 1);
        });
        black_box(image);
    }

    #[bench]
    fn bench_salt_and_pepper_noise_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| {
            salt_and_pepper_noise_mut(&mut image, 0.3, 1);
        });
        black_box(image);
    }

    #[bench]
    fn bench_speckle_noise_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| {
            speckle_noise_mut(&mut image, 0.0, 0.4, 1);
        });
        black_box(image);
    }

    #[bench]
    fn bench_speckle_noise_per_channel_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| {
            speckle_noise_per_channel_mut(&mut image, 0.0, 0.4, 1);
        });
        black_box(image);
    }
}
