//! Utilities to help with writing property-based tests
//! (e.g. [quickcheck] tests) for image processing functions.
//!
//! [quickcheck]: https://github.com/BurntSushi/quickcheck

use crate::definitions::Image;
use image::{GenericImage, ImageBuffer, Luma, Pixel, Primitive, Rgb};
use quickcheck::{Arbitrary, Gen};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use std::fmt;

/// Wrapper for image buffers to allow us to write an Arbitrary instance.
#[derive(Clone)]
pub struct TestBuffer<T: Pixel>(pub Image<T>);

/// 8bpp grayscale `TestBuffer`.
pub type GrayTestImage = TestBuffer<Luma<u8>>;

/// 24bpp RGB `TestBuffer`.
pub type RgbTestImage = TestBuffer<Rgb<u8>>;

impl<T: Pixel + ArbitraryPixel + Send + 'static> Arbitrary for TestBuffer<T>
where
    <T as Pixel>::Subpixel: Send,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let (width, height) = small_image_dimensions(g);
        let mut image = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pix: T = ArbitraryPixel::arbitrary(g);
                image.put_pixel(x, y, pix);
            }
        }
        TestBuffer(image)
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = TestBuffer<T>>> {
        Box::new(shrink(&self.0).map(TestBuffer))
    }
}

/// Workaround for not being able to define Arbitrary instances for pixel types
/// defines in other modules.
pub trait ArbitraryPixel {
    /// Generate an arbitrary instance of this pixel type.
    fn arbitrary<G: Gen>(g: &mut G) -> Self;
}

fn shrink<I>(image: &I) -> Box<dyn Iterator<Item = Image<I::Pixel>>>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut subs = vec![];

    let w = image.width();
    let h = image.height();

    if w > 0 {
        let left = copy_sub(image, 0, 0, w - 1, h);
        subs.push(left);
        let right = copy_sub(image, 1, 0, w - 1, h);
        subs.push(right);
    }
    if h > 0 {
        let top = copy_sub(image, 0, 0, w, h - 1);
        subs.push(top);
        let bottom = copy_sub(image, 0, 1, w, h - 1);
        subs.push(bottom);
    }

    Box::new(subs.into_iter())
}

fn copy_sub<I>(image: &I, x: u32, y: u32, width: u32, height: u32) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(width, height);
    for dy in 0..height {
        let oy = y + dy;
        for dx in 0..width {
            let ox = x + dx;
            out.put_pixel(dx, dy, image.get_pixel(ox, oy));
        }
    }
    out
}

impl<T: fmt::Debug + Pixel + 'static> fmt::Debug for TestBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "width: {}, height: {}, data: {:?}",
            self.0.width(),
            self.0.height(),
            self.0.enumerate_pixels().collect::<Vec<_>>()
        )
    }
}

fn small_image_dimensions<G: Gen>(g: &mut G) -> (u32, u32) {
    let dims: (u8, u8) = Arbitrary::arbitrary(g);
    ((dims.0 % 10) as u32, (dims.1 % 10) as u32)
}

impl<T: Send + Primitive> ArbitraryPixel for Rgb<T>
where
    Standard: Distribution<T>,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let red: T = g.gen();
        let green: T = g.gen();
        let blue: T = g.gen();
        Rgb([red, green, blue])
    }
}

impl<T: Send + Primitive> ArbitraryPixel for Luma<T>
where
    Standard: Distribution<T>,
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        let val: T = g.gen();
        Luma([val])
    }
}
