//! Statistical properties of images.

use image::{
    GenericImage,
    Pixel,
    Primitive
};

use num::Bounded;

use math::cast;
use conv::ValueInto;

/// Returns the square root of the mean of the squares of differences
/// between all subpixels in left and right. If you do not want to consider
/// all image channels then you should first change image format to remove
/// the irrelevant channels.
pub fn root_mean_squared_error<I, J, P>(left: &I, right: &J) -> f64
    where I: GenericImage<Pixel=P>,
          J: GenericImage<Pixel=P>,
          P: Pixel,
          P::Subpixel: ValueInto<f64>
{
    mean_squared_error(left, right).sqrt()
}

/// Returns the peak signal to noise ratio for a clean image and its noisy
/// aproximation. All channels are considered equally. If you do not want this
/// (e.g. if using RGBA) then change image formats first.
/// https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio.
pub fn peak_signal_to_noise_ratio<I, J, P>(original: &I, noisy: &J) -> f64
    where I: GenericImage<Pixel=P>,
          J: GenericImage<Pixel=P>,
          P: Pixel,
          P::Subpixel: ValueInto<f64> + Primitive
{
    let max: f64 = cast(<P::Subpixel as Bounded>::max_value());
    let mse = mean_squared_error(original, noisy);
    20f64 * max.log(10f64) - 10f64 * mse.log(10f64)
}

fn mean_squared_error<I, J, P>(left: &I, right: &J) -> f64
    where I: GenericImage<Pixel=P>,
          J: GenericImage<Pixel=P>,
          P: Pixel,
          P::Subpixel: ValueInto<f64>
{
    assert_dimensions_match!(left, right);
    let mut sum_squared_diffs = 0f64;
    for (p, q) in left.pixels().zip(right.pixels()) {
        for (c, d) in p.2.channels().iter().zip(q.2.channels().iter()) {
            let fc: f64 = cast(*c);
            let fd: f64 = cast(*d);
            let diff = fc - fd;
            sum_squared_diffs += diff * diff;
        }
    }
    let count = (left.width() * left.height() * P::channel_count() as u32) as f64;
    sum_squared_diffs / count
}

#[cfg(test)]
mod test {
    use super::root_mean_squared_error;
    use image::{
        GrayImage,
        ImageBuffer,
        RgbImage
    };

    #[test]
    fn test_root_mean_squared_error_grayscale() {
        let left:  GrayImage = ImageBuffer::from_raw(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let right: GrayImage = ImageBuffer::from_raw(3, 2, vec![8, 4, 7, 6, 9, 1]).unwrap();
        let rms = root_mean_squared_error(&left, &right);
        let expected = (114f64 / 6f64).sqrt();
        assert_eq!(rms, expected);
    }

    #[test]
    fn test_root_mean_squared_error_rgb() {
        let left: RgbImage  = ImageBuffer::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let right: RgbImage = ImageBuffer::from_raw(2, 1, vec![8, 4, 7, 6, 9, 1]).unwrap();
        let rms = root_mean_squared_error(&left, &right);
        let expected = (114f64 / 6f64).sqrt();
        assert_eq!(rms, expected);
    }

    #[test]
    #[should_panic]
    fn test_root_mean_squares_rejects_mismatched_dimensions() {
        let left: GrayImage  = ImageBuffer::from_raw(2, 1, vec![1, 2]).unwrap();
        let right: GrayImage = ImageBuffer::from_raw(1, 2, vec![8, 4]).unwrap();
        let _ = root_mean_squared_error(&left, &right);
    }
}
