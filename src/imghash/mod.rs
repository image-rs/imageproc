//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing

mod bits;
mod phash;
mod signals;

use bits::Bits64;

pub use phash::{phash, PHash};
