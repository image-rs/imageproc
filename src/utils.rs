
extern crate image;

use conv::{
    ValueInto
};

use image::{
    ImageBuffer,
    Pixel
};

/// Panics if any pixels differ between the two input images.
#[macro_export]
macro_rules! assert_pixels_eq {
    ($actual:expr, $expected:expr) => ({
        let actual_dim = $actual.dimensions();
        let expected_dim = $expected.dimensions();

        if actual_dim != expected_dim {
            panic!("dimensions do not match. \
                actual: {:?}, expected: {:?}", actual_dim, expected_dim)
        }

        // Can't just call $image.pixels(), as that needn't hit the
        // trait pixels method - ImageBuffer defines its own pixels
        // method with a different signature
        let diffs = ::image::GenericImage::pixels(&$actual)
            .zip(::image::GenericImage::pixels(&$expected))
            .filter(|&(p, q)| p != q)
            .collect::<Vec<_>>();

        if !diffs.is_empty() {
            let mut err = "pixels do not match. ".to_string();

            let diff_messages = diffs
                .iter()
                .take(5)
                .map(|d| format!("\nactual: {:?}, expected {:?} ", d.0, d.1))
                .collect::<Vec<_>>()
                .join("");

            err.push_str(&diff_messages);
            panic!(err)
        }
     })
}

use std::path::Path;

/// Loads image at given path, panicking on failure.
pub fn load_image_or_panic(path: &Path) -> image::DynamicImage {
     image::open(path)
         .ok()
         .expect(&format!("Could not load image at {:?}", path))
}

/// Gray image to use in benchmarks. This is neither noise nor
/// similar to natural images - it's just a convenience method
/// to produce an image that's not constant.
pub fn gray_bench_image(width: u32, height: u32) -> image::GrayImage {
    let mut image = image::GrayImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let intensity = (x % 7 + y % 6) as u8;
            image.put_pixel(x, y, image::Luma([intensity]));
        }
    }
    image
}

/// RGB image to use in benchmarks. See comment on gray_bench_image.
pub fn rgb_bench_image(width: u32, height: u32) -> image::RgbImage {
    use std::cmp;
    let mut image = image::RgbImage::new(width, height);
    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = (x % 7 + y % 6) as u8;
            let g = 255u8 - r;
            let b = cmp::min(r, g);
            image.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    image
}

/// Helper for a conversion that we know can't fail.
pub fn cast<T, U>(x: T) -> U where T: ValueInto<U> {
    match x.value_into() {
        Ok(y) => y,
        Err(_) => panic!("Failed to convert"),
    }
}

/// An ImageBuffer containing Pixels of type P with storage
/// Vec<P::Subpixel>.
// TODO: This produces a compiler warning about trait bounds
// TODO: not being enforced in type definitions. In this case
// TODO: they are. Can we get rid of the warning?
pub type VecBuffer<P: Pixel> = ImageBuffer<P, Vec<P::Subpixel>>;
