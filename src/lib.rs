//! An image processing library, based on the
//! [image](https://github.com/PistonDevelopers/image) crate.
#![deny(missing_docs)]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
extern crate conv;
extern crate image;
extern crate itertools;
extern crate nalgebra;
extern crate num;
extern crate quickcheck;
extern crate rand;
extern crate rusttype;
extern crate rayon;

#[macro_use]
pub mod utils;
pub mod affine;
pub mod contrast;
pub mod corners;
pub mod definitions;
pub mod drawing;
pub mod edges;
pub mod filter;
pub mod gradients;
pub mod haar;
pub mod hog;
pub mod hough;
pub mod integral_image;
pub mod local_binary_patterns;
pub mod map;
pub mod math;
pub mod morphology;
pub mod noise;
pub mod pixelops;
pub mod property_testing;
pub mod rect;
pub mod region_labelling;
pub mod stats;
pub mod suppress;
pub mod union_find;
