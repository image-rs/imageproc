//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing

mod bits;
mod phash;
mod signals;

use bits::Bits64;

#[cfg(feature = "fft")]
pub use phash::phash;
pub use phash::PHash;
