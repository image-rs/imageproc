#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![cfg_attr(test, feature(test))]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::zero_prefixed_literal,
    clippy::needless_range_loop
)]

#[cfg(test)]
extern crate test;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

#[cfg(test)]
mod proptest_utils;

#[macro_use]
pub mod utils;
#[macro_use]
pub mod doc_macros;
pub mod binary_descriptors;
pub mod compose;
pub mod contours;
pub mod contrast;
pub mod corners;
pub mod definitions;
pub mod distance_transform;
pub mod drawing;
pub mod edges;
pub mod filter;
pub mod geometric_transformations;
pub mod geometry;
pub mod gradients;
pub mod haar;
pub mod hog;
pub mod hough;
pub mod integral_image;
pub mod kernel;
pub mod local_binary_patterns;
pub mod map;
pub mod math;
pub mod morphology;
pub mod noise;
pub mod pixelops;
pub mod point;
pub mod rect;
pub mod region_labelling;
pub mod seam_carving;
pub mod stats;
pub mod suppress;
pub mod template_matching;
pub mod union_find;
#[cfg(feature = "display-window")]
pub mod window;

pub use image;
