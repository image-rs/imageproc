//! An image processing library based on the
//! [image] crate.
//!
//! Note that the image crate contains some image
//! processing functions (including image resizing) in its
//! `imageops` module, so check there if you cannot find
//! a standard image processing function in this crate.
//!
//! [image]: https://github.com/PistonDevelopers/image
#![deny(missing_docs)]
#![cfg_attr(test, feature(test))]

#[cfg(test)]
extern crate test;
extern crate conv;
extern crate image;
extern crate itertools;
extern crate num;
extern crate quickcheck;
extern crate rand;
extern crate rusttype;
extern crate rayon;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

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
