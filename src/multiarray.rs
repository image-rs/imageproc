//! Helpers providing multi-dimensional views of 1d data.

use num::Zero;

/// A 3d array that owns its data.
pub struct Array3d<T> {
    pub data: Vec<T>,
    /// Lengths of the dimensions, from innermost (i.e. fastest-varying) to outermost.
    pub lengths: [usize; 3]
}

/// A view into a 3d array.
pub struct View3d<'a, T: 'a> {
    /// THe underlying data.
	pub data: &'a mut [T],
    /// Lengths of the dimensions, from innermost (i.e. fastest-varying) to outermost.
    pub lengths: [usize; 3]
}

impl<T: Zero + Clone> Array3d<T> {
    /// Allocates a new Array3d with the given dimensions.
	pub fn new(lengths: [usize; 3]) -> Array3d<T> {
		let data = vec![Zero::zero(); data_length(lengths)];
		Array3d { data: data, lengths: lengths }
	}

    /// Provides a 3d view of the data.
    pub fn view_mut(&mut self) -> View3d<T> {
        View3d::from_raw(&mut self.data, self.lengths)
    }
}

impl<'a, T> View3d<'a, T> {

	/// Constructs index from existing data and the lengths of the desired dimensions.
    pub fn from_raw(data: &'a mut [T], lengths: [usize; 3]) -> View3d<'a, T> {
        View3d { data: data, lengths: lengths }
    }

    /// Immutable access to the raw data.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable access to the raw data.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// An immutable reference from a 3d index.
    pub fn at(&self, indices: [usize; 3]) -> &T {
        &self.data[self.offset(indices)]
    }

    /// A mutable reference from a 3d index.
    pub fn at_mut(&mut self, indices: [usize; 3]) -> &mut T {
        &mut self.data[self.offset(indices)]
    }

    /// All entries with the given outer dimensions. As the first dimension
    /// is fastest varying, this is a contiguous slice.
    pub fn inner_slice(&self, x1: usize, x2: usize) -> &[T] {
        let offset = self.offset([0, x1, x2]);
        & self.data[offset..offset + self.lengths[0]]
    }

    /// All entries with the given outer dimensions. As the first dimension
    /// is fastest varying, this is a contiguous slice.
    pub fn inner_slice_mut(&mut self, x1: usize, x2: usize) -> &mut [T] {
        let offset = self.offset([0, x1, x2]);
        &mut self.data[offset..offset + self.lengths[0]]
    }

    fn offset(&self, indices: [usize; 3]) -> usize {
        indices[2] * self.lengths[1] * self.lengths[0] + indices[1] * self.lengths[0] + indices[0]
    }
}

/// Length of array needed for the given dimensions.
fn data_length(lengths: [usize; 3]) -> usize {
    lengths[0] * lengths[1] * lengths[2]
}
