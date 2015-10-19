
#![feature(test)]

#[cfg(test)]
extern crate test;
extern crate conv;
extern crate image;

#[macro_use]
extern crate nalgebra;
extern crate num;
extern crate rand;

#[macro_use]
pub mod utils;
pub mod affine;
pub mod contrast;
pub mod corners;
pub mod definitions;
pub mod drawing;
pub mod filter;
pub mod gradients;
pub mod hog;
pub mod integralimage;
pub mod localbinarypatterns;
pub mod math;
pub mod multiarray;
pub mod noise;
pub mod regionlabelling;
pub mod unionfind;
