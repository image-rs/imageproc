//! Functions for adding synthetic noise to images.

use image::{GenericImage, ImageBuffer, Pixel};
use rand::{SeedableRng, StdRng};
use rand::distributions::{IndependentSample, Normal, Range};
use crate::definitions::{Clamp, HasBlack, HasWhite, Image};
use conv::ValueInto;
use crate::math::cast;

/// Adds independent additive Gaussian noise to all channels
/// of an image, with the given mean and standard deviation.
pub fn gaussian_noise<P>(image: &Image<P>, mean: f64, stddev: f64, seed: usize) -> Image<P>
where
    P: Pixel + 'static,
    P::Subpixel: ValueInto<f64> + Clamp<f64>,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    gaussian_noise_mut(&mut out, mean, stddev, seed);
    out
}

/// Adds independent additive Gaussian noise to all channels
/// of an image in place, with the given mean and standard deviation.
pub fn gaussian_noise_mut<P>(image: &mut Image<P>, mean: f64, stddev: f64, seed: usize)
where
    P: Pixel + 'static,
    P::Subpixel: ValueInto<f64> + Clamp<f64>,
{
    let seed_array: &[_] = &[seed];
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);
    let normal = Normal::new(mean, stddev);
    let num_channels = P::channel_count() as usize;

    for p in image.pixels_mut() {
        for c in 0..num_channels {
            let noise = normal.ind_sample(&mut rng);
            let channel: f64 = cast(p.channels()[c]);
            p.channels_mut()[c] = P::Subpixel::clamp(channel + noise);
        }
    }
}

/// Converts pixels to black or white at the given `rate` (between 0.0 and 1.0).
/// Black and white occur with equal probability.
pub fn salt_and_pepper_noise<P>(image: &Image<P>, rate: f64, seed: usize) -> Image<P>
where
    P: Pixel + HasBlack + HasWhite + 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    salt_and_pepper_noise_mut(&mut out, rate, seed);
    out
}

/// Converts pixels to black or white in place at the given `rate` (between 0.0 and 1.0).
/// Black and white occur with equal probability.
pub fn salt_and_pepper_noise_mut<P>(image: &mut Image<P>, rate: f64, seed: usize)
where
    P: Pixel + HasBlack + HasWhite + 'static,
{
    let seed_array: &[_] = &[seed];
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);
    let uniform = Range::new(0.0, 1.0);

    for p in image.pixels_mut() {
        if uniform.ind_sample(&mut rng) > rate {
            continue;
        }
        let r = uniform.ind_sample(&mut rng);
        *p = if r >= 0.5 { P::white() } else { P::black() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GrayImage;
    use test::{Bencher, black_box};

    #[bench]
    fn bench_gaussian_noise_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| { gaussian_noise_mut(&mut image, 30.0, 40.0, 1usize); });
        black_box(image);
    }

    #[bench]
    fn bench_salt_and_pepper_noise_mut(b: &mut Bencher) {
        let mut image = GrayImage::new(100, 100);
        b.iter(|| { salt_and_pepper_noise_mut(&mut image, 0.3, 1usize); });
        black_box(image);
    }
}
