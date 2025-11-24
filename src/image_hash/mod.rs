//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing

mod average_hash;
mod bits;
mod phash;
mod signals;

use bits::Bits64;

pub use average_hash::{average_hash, AverageHash};
pub use phash::{phash, PHash};
