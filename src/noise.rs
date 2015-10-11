//! Functions for adding synthetic noise to images.

use image::{
    GenericImage,
    ImageBuffer,
    Pixel
};

use rand::{
    SeedableRng,
    StdRng
};

use rand::distributions::{
    IndependentSample,
    Normal,
    Range
};

use traits::{
    Clamp,
    HasBlack,
    HasWhite
};

use conv::{
    ValueInto
};

use utils::{
    cast,
    VecBuffer
};

/// Adds independent additive Gaussian noise to all channels
/// of an image, with the given mean and standard deviation.
pub fn gaussian_noise<I>(image: &I, mean: f64, stddev: f64, seed: usize)
        -> VecBuffer<I::Pixel>
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f64> + Clamp<f64> + 'static {

    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    gaussian_noise_mut(&mut out, mean, stddev, seed);
    out
}

/// Adds independent additive Gaussian noise to all channels
/// of an image in place, with the given mean and standard deviation.
pub fn gaussian_noise_mut<I>(image: &mut I, mean: f64, stddev: f64, seed: usize)
    where I: GenericImage + 'static,
          I::Pixel: 'static,
          <I::Pixel as Pixel>::Subpixel: ValueInto<f64> + Clamp<f64> + 'static {

    let seed_array: &[_] = &[seed];
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);

    let normal = Normal::new(mean, stddev);

    for y in 0..image.height() {
        for x in 0..image.width() {
            let mut pix = image.get_pixel(x, y);
            let num_channels = I::Pixel::channel_count() as usize;

            for c in 0..num_channels {
                let noise = normal.ind_sample(&mut rng);
                let channel: f64 = cast(pix.channels()[c]);
                pix.channels_mut()[c]
                    = <I::Pixel as Pixel>::Subpixel::clamp(channel + noise);
            }

            image.put_pixel(x, y, pix);
        }
    }
}

/// Converts pixels to black or white at the given rate. Black and
/// white occur with equal probability.
pub fn salt_and_pepper_noise<I>(image: &I, rate: f64, seed: usize)
        -> VecBuffer<I::Pixel>
    where I: GenericImage + 'static,
          I::Pixel: HasBlack + HasWhite + 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    salt_and_pepper_noise_mut(&mut out, rate, seed);
    out
}

/// Converts pixels to black or white in place at the given rate. Black and
/// white occur with equal probability.
pub fn salt_and_pepper_noise_mut<I>(image: &mut I, rate: f64, seed: usize)
    where I: GenericImage + 'static,
          I::Pixel: HasBlack + HasWhite + 'static,
          <I::Pixel as Pixel>::Subpixel: 'static {

    let seed_array: &[_] = &[seed];
    let mut rng: StdRng = SeedableRng::from_seed(seed_array);

    let uniform = Range::new(0.0, 1.0);

    for y in 0..image.height() {
        for x in 0..image.width() {

            if uniform.ind_sample(&mut rng) > rate {
                continue;
            }

            if uniform.ind_sample(&mut rng) >= 0.5 {
                image.put_pixel(x, y, I::Pixel::white());
            }
            else {
                image.put_pixel(x, y, I::Pixel::black());
            }
        }
    }
}
