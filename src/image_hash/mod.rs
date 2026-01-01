//! [Perceptual hashing] algorithms for images.
//!
//! [Perceptual hashing]: https://en.wikipedia.org/wiki/Perceptual_hashing

mod average_hash;

mod bits;
#[cfg(feature = "fft")]
mod phash;
#[cfg(feature = "fft")]
mod signals;

#[cfg(feature = "fft")]
pub use phash::{PHash, phash};

use bits::Bits64;

pub use average_hash::{AverageHash, average_hash};
