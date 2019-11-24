//! TODO: Doc
#![allow(missing_docs)]
use image::GenericImageView;
use image::{ImageBuffer, Pixel};
use std::ops::Deref;
pub struct Window<I, P> {
    pub image: I,
    pub xoffset: i32,
    pub yoffset: i32,
    pub xstride: u32,
    pub ystride: u32,
    pub pad_option: PadOption<P>,
}

type DerefPixel<I> = <<I as Deref>::Target as GenericImageView>::Pixel;

impl<I> Window<I, DerefPixel<I>>
where
    I: Deref,
    I::Target: GenericImageView + Sized,
{
    pub fn dimensions(&self) -> (u32, u32) {
        (self.xstride, self.ystride)
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> DerefPixel<I> {
        use PadOption::*;
        let (width, height) = self.image.dimensions();
        let (width, height) = (width as i32, height as i32);
        let (x, y) = (x as i32 + self.xoffset, y as i32 + self.yoffset);

        // TODO: check transformed `x`, `y` in bound
        match self.pad_option {
            Symmetric => {
                let x = clamp(x, -x, width * 2 - 2 - x) as u32;
                let y = clamp(y, -y, height * 2 - 2 - y) as u32;
                self.image.get_pixel(x, y)
            }
            Replicate => {
                let x = clamp(x, 0, width - 1) as u32;
                let y = clamp(y, 0, height - 1) as u32;
                self.image.get_pixel(x, y)
            }
            Circular => {
                let x = ((x + width) % width) as u32;
                let y = ((y + height) % height) as u32;
                self.image.get_pixel(x, y)
            }
            Constant(p) => {
                if x < 0 || x >= width || y < 0 || y >= height {
                    p
                } else {
                    self.image.get_pixel(x as u32, y as u32)
                }
            }
        }
    }
}

fn clamp<T: std::cmp::Ord>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

pub enum PadOption<P> {
    Symmetric,
    Replicate,
    Circular,
    Constant(P),
}

pub enum SizeOption {
    Same,
    Full,
    Valid,
}

pub trait ImageWindow {}
