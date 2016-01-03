//! Utils for testing and debugging.

use image::{
    DynamicImage,
    GenericImage,
    GrayImage,
    Luma,
    open,
    Pixel,
    Rgb,
    RgbImage
};

use quickcheck::{
    Arbitrary,
    Gen
};

use std::fmt::Debug;
use std::path::Path;

/// Panics if any pixels differ between the two input images.
#[macro_export]
macro_rules! assert_pixels_eq {
    ($actual:expr, $expected:expr) => ({
        assert_dimensions_match!($actual, $expected);
        let diffs = $crate::utils::pixel_diffs(&$actual, &$expected, |p, q| p != q);
        if !diffs.is_empty() {
            panic!($crate::utils::describe_pixel_diffs(diffs.into_iter()))
        }
     })
}

/// Panics if any pixels differ between the two images by more than the
/// given tolerance in a single channel.
#[macro_export]
macro_rules! assert_pixels_eq_within {
    ($actual:expr, $expected:expr, $channel_tolerance:expr) => ({

        assert_dimensions_match!($actual, $expected);
        let diffs = $crate::utils::pixel_diffs(&$actual, &$expected, |p, q| {

            use image::Pixel;
            let cp = p.2.channels();
            let cq = q.2.channels();
            if cp.len() != cq.len() {
                panic!("pixels have different channel counts. \
                    actual: {:?}, expected: {:?}", cp.len(), cq.len())
            }

            let mut large_diff = false;
            for i in 0..cp.len() {
                let sp = cp[i];
                let sq = cq[i];
                // Handle unsigned subpixels
                let diff = if sp > sq {sp - sq} else {sq - sp};
                if diff > $channel_tolerance {
                    large_diff = true;
                    break;
                }
            }

            large_diff
        });
        if !diffs.is_empty() {
            panic!($crate::utils::describe_pixel_diffs(diffs.into_iter()))
        }
    })
}

/// Panics if image dimensions do not match.
#[macro_export]
macro_rules! assert_dimensions_match {
    ($actual:expr, $expected:expr) => ({

        let actual_dim = $actual.dimensions();
        let expected_dim = $expected.dimensions();

        if actual_dim != expected_dim {
            panic!("dimensions do not match. \
                actual: {:?}, expected: {:?}", actual_dim, expected_dim)
        }
     })
}

/// Lists pixels that differ between left and right images.
pub fn pixel_diffs<I, F>(left: &I, right: &I, is_diff: F)
        -> Vec<((u32, u32, I::Pixel), (u32, u32, I::Pixel))>
    where I: GenericImage,
          I::Pixel: PartialEq,
          F: Fn((u32, u32, I::Pixel), (u32, u32, I::Pixel)) -> bool {

    // Can't just call $image.pixels(), as that needn't hit the
    // trait pixels method - ImageBuffer defines its own pixels
    // method with a different signature
    GenericImage::pixels(left)
        .zip(GenericImage::pixels(right))
        .filter(|&(p, q)| is_diff(p, q))//p != q)
        .collect::<Vec<_>>()
}

/// Gives a summary description of a list of pixel diffs for use in error messages.
pub fn describe_pixel_diffs<I, P>(diffs: I) -> String
    where I: Iterator<Item=(P, P)>,
          P: Debug {

    let mut err = "pixels do not match. ".to_string();
    err.push_str(&(diffs
        .take(5)
        .map(|d| format!("\nactual: {:?}, expected {:?} ", d.0, d.1))
        .collect::<Vec<_>>()
        .join("")));
    err
}

/// Loads image at given path, panicking on failure.
pub fn load_image_or_panic(path: &Path) -> DynamicImage {
     open(path)
         .ok()
         .expect(&format!("Could not load image at {:?}", path))
}

/// Gray image to use in benchmarks. This is neither noise nor
/// similar to natural images - it's just a convenience method
/// to produce an image that's not constant.
pub fn gray_bench_image(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let intensity = (x % 7 + y % 6) as u8;
            image.put_pixel(x, y, Luma([intensity]));
        }
    }
    image
}

/// RGB image to use in benchmarks. See comment on gray_bench_image.
pub fn rgb_bench_image(width: u32, height: u32) -> RgbImage {
    use std::cmp;
    let mut image = RgbImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = (x % 7 + y % 6) as u8;
            let g = 255u8 - r;
            let b = cmp::min(r, g);
            image.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    image
}

/// Wrapper for GrayImage to allow us to write an Arbitrary instance.
pub struct GrayTestImage(GrayImage);

impl Clone for GrayTestImage {
    fn clone(&self) -> Self {
        let mut out = GrayImage::new(self.0.width(), self.0.height());
        out.copy_from(&self.0, 0, 0);
        GrayTestImage(out)
    }
}

impl Arbitrary for GrayTestImage {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let (width, height) = small_image_dimensions(g);
        let mut image = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val: u8 = g.gen();
                image.put_pixel(x, y, Luma([val]));
            }
        }
        GrayTestImage(image)
    }
}

fn small_image_dimensions<G: Gen>(g: &mut G) -> (u32, u32) {
    let dims: (u8, u8) = Arbitrary::arbitrary(g);
    (dims.0 as u32, dims.1 as u32)
}

/// Wrapper for RgbImage to allow us to write an Arbitrary instance.
pub struct RgbTestImage(RgbImage);

impl Clone for RgbTestImage {
    fn clone(&self) -> Self {
        let mut out = RgbImage::new(self.0.width(), self.0.height());
        out.copy_from(&self.0, 0, 0);
        RgbTestImage(out)
    }
}

impl Arbitrary for RgbTestImage {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let (width, height) = small_image_dimensions(g);
        let mut image = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let red: u8 = g.gen();
                let green: u8 = g.gen();
                let blue: u8 = g.gen();
                image.put_pixel(x, y, Rgb([red, green, blue]));
            }
        }
        RgbTestImage(image)
    }
}

#[cfg(test)]
mod test {

    use image::{
        GrayImage,
        ImageBuffer
    };

    #[test]
    fn test_assert_pixels_eq_passes() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        assert_pixels_eq!(image, image);
    }

    #[test]
    #[should_panic]
    fn test_assert_pixels_eq_fails() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let diff: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 11, 02,
            10, 11, 12]).unwrap();

        assert_pixels_eq!(diff, image);
    }

    #[test]
    fn test_assert_pixels_eq_within_passes() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let diff: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 02, 02,
            10, 11, 12]).unwrap();

        assert_pixels_eq_within!(diff, image, 1);
    }

    #[test]
    #[should_panic]
    fn test_assert_pixels_eq_within_fails() {
        let image: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 01, 02,
            10, 11, 12]).unwrap();

        let diff: GrayImage = ImageBuffer::from_raw(3, 2, vec![
            00, 03, 02,
            10, 11, 12]).unwrap();

        assert_pixels_eq_within!(diff, image, 1);
    }
}
