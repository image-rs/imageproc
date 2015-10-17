//! Helpers providing multi-dimensional views of 1d data.

use num::Zero;

/// A mutable slice together with 3 dimensions.
pub trait Mut3d<T> {
    fn data_mut(&mut self) -> &mut [T];
    fn dimensions(&self) -> Dim3;
}

/// Dimensions of a 3d array.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Dim3 {
    /// Length of the innermost, i.e. fastest-varying, dimension.
    pub len_0: usize,
    /// Length of the middle dimension.
    pub len_1: usize,
    /// Length of the outermost dimension.
    pub len_2: usize
}

impl Dim3 {
    pub fn new(len_0: usize, len_1: usize, len_2: usize) -> Dim3 {
        Dim3 { len_0: len_0, len_1: len_1, len_2: len_2}
    }
}

pub fn element_at_mut<V, T>(view: &mut V, x0: usize, x1: usize, x2: usize) -> &mut T
    where V: Mut3d<T> {
    let d = view.dimensions();
    let idx = x2 * d.len_1 * d.len_0 + x1 * d.len_0 + x0;
    &mut view.data_mut()[idx]
}

pub fn inner_dimension_slice<V, T>(view: &mut V, x1: usize, x2: usize) -> &mut [T]
    where V: Mut3d<T> {
    let d = view.dimensions();
    let idx = x2 * d.len_1 * d.len_0 + x1 * d.len_0;
    &mut view.data_mut()[idx..idx + d.len_0]
}

/// A 3d array that owns its data.
pub struct Array3d<T> {
    pub data: Vec<T>,
    pub dim: Dim3
}

/// A view into a 3d array.
pub struct View3d<'a, T: 'a> {
	pub data: &'a mut [T],
    pub dim: Dim3
}

// TODO: the two impls are identical. This is stupid
impl<T> Mut3d<T> for Array3d<T> {
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn dimensions(&self) -> Dim3 {
        self.dim
    }
}

impl<'a, T> Mut3d<T> for View3d<'a, T> {
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn dimensions(&self) -> Dim3 {
        self.dim
    }
}

impl<T: Zero + Clone> Array3d<T> {
    /// Allocates a new Array3d with the given dimensions.
	pub fn new(len_0: usize, len_1: usize, len_2: usize) -> Array3d<T> {
		let data = vec![Zero::zero(); len_0 * len_1 * len_2];
		Array3d {
			data: data,
			dim: Dim3::new(len_0, len_1, len_2)
		}
	}
}

impl<'a, T: Clone> View3d<'a, T> {
	/// Constructs index from existing data and the lengths of the desired dimensions.
    pub fn from_raw(data: &'a mut [T], len_0: usize, len_1: usize, len_2: usize) -> View3d<'a, T> {
        View3d {
            data: data,
            dim: Dim3::new(len_0, len_1, len_2)
        }
    }
}
