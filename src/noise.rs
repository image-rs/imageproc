//! Functions for adding synthetic noise to images.

use crate::definitions::{Clamp, HasBlack, HasWhite, Image};
use crate::math::cast;
use conv::ValueInto;
use image::Pixel;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

/// Adds independent additive Gaussian noise to all channels
/// of an image, with the given mean and standard deviation.
pub fn gaussian_noise<P>(image: &Image<P>, mean: f64, stddev: f64, seed: u64) -> Image<P>
where
    P: Pixel + 'static,
    P::Subpixel: ValueInto<f64> + Clamp<f64>,
{
    let mut out = image.clone();
    gaussian_noise_mut(&mut out, mean, stddev, seed);
    out
}

/// Adds independent additive Gaussian noise to all channels
/// of an image in place, with the given mean and standard deviation.
pub fn gaussian_noise_mut<P>(image: &mut Image<P>, mean: f64, stddev: f64, seed: u64)
where
    P: Pixel + 'static,
    P::Subpixel: ValueInto<f64> + Clamp<f64>,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let normal = Normal::new(mean, stddev).unwrap();

    for p in image.pixels_mut() {
        for c in p.channels_mut() {
            let noise = normal.sample(&mut rng);
            *c = P::Subpixel::clamp(cast(*c) + noise);
        }
    }
}

/// Converts pixels to black or white at the given `rate` (between 0.0 and 1.0).
/// Black and white occur with equal probability.
pub fn salt_and_pepper_noise<P>(image: &Image<P>, rate: f64, seed: u64) -> Image<P>
where
    P: Pixel + HasBlack + HasWhite + 'static,
{
    let mut out = image.clone();
    salt_and_pepper_noise_mut(&mut out, rate, seed);
    out
}

/// Converts pixels to black or white in place at the given `rate` (between 0.0 and 1.0).
/// Black and white occur with equal probability.
pub fn salt_and_pepper_noise_mut<P>(image: &mut Image<P>, rate: f64, seed: u64)
where
    P: Pixel + HasBlack + HasWhite + 'static,
{
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let uniform = Uniform::new(0.0, 1.0);

    for p in image.pixels_mut() {
        if uniform.sample(&mut rng) > rate {
            continue;
        }
        let r = uniform.sample(&mut rng);
        *p = if r >= 0.5 { P::white() } else { P::black() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;
    use test::{black_box, Bencher};

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
}
