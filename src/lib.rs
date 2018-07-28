//! An image processing library, based on the
//! [image](https://github.com/PistonDevelopers/image) crate.
#![deny(missing_docs)]
#![cfg_attr(test, feature(test))]

#![feature(rust_2018_preview)]
#![warn(rust_2018_idioms)]

#[cfg(test)]
extern crate test;

#[macro_use]
pub mod utils;
pub mod affine;
pub mod contrast;
pub mod corners;
pub mod definitions;
pub mod distance_transform;
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
pub mod seam_carving;
pub mod stats;
pub mod suppress;
pub mod template_matching;
pub mod union_find;
