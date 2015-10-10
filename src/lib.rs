
#![feature(test)]

extern crate conv;

extern crate image;

extern crate num;

extern crate rand;

#[cfg(test)]
extern crate test;

#[macro_use]
pub mod utils;

pub mod affine;

pub mod contrast;

pub mod integralimage;

pub mod filter;

pub mod unionfind;

pub mod regionlabelling;

pub mod corners;

pub mod drawing;

pub mod noise;

mod traits;
