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
#![allow(
    clippy::cast_lossless, clippy::too_many_arguments, clippy::needless_range_loop,
    clippy::useless_let_if_seq, clippy::match_wild_err_arm, clippy::range_plus_one,
    clippy::trivially_copy_pass_by_ref, clippy::nonminimal_bool, clippy::expect_fun_call,
    clippy::many_single_char_names
)]
// This was added when bumping image from 0.21 to 0.22.1, as the new version
// deprecated unsafe_get_pixel on GenericImageView and we use the method
// throughout this library.
#![allow(deprecated)]

#[cfg(test)]
extern crate test;
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
pub mod proj;
#[cfg(any(feature = "property-testing", test))]
pub mod property_testing;
pub mod rect;
pub mod region_labelling;
pub mod seam_carving;
pub mod stats;
pub mod suppress;
pub mod template_matching;
pub mod union_find;
#[cfg(feature = "display-window")]
pub mod window;
