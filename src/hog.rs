//! Histogram of oriented gradients and helpers for visualizing them.
//! http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

use image::{
	GenericImage,
	Luma
};
use gradients::{
	horizontal_sobel,
	vertical_sobel
};
use num::{
	Zero
};
use std::f32;

/// Parameters for HoG descriptors.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HogOptions {
	/// Number of gradient orientation bins.
	pub orientations: u32,

	/// Whether gradients in opposite directions
	/// are treated as equal.
    pub signed: bool,

	/// Width and height of cell in pixels.
	pub cell_side: u32,

	/// Width and height of block in cells.
	pub block_side: u32,

	/// Offset of the start of one block from the next in cells.
	pub block_stride: u32

	// TODO: choice of normalisation - for now we just scale
    // TODO: to unit L2 norm
}

impl HogOptions {
    pub fn new(orientations: u32, signed: bool, cell_side: u32,
        block_side: u32, block_stride: u32) -> HogOptions {

        HogOptions {
            orientations: orientations,
            signed: signed,
            cell_side: cell_side,
            block_side: block_side,
            block_stride: block_stride}
    }
}

/// Returns the total size in floats of the HoG descriptor for an
/// image of given width and height using these options.
pub fn hog_size(width: u32, height: u32, options: HogOptions) -> u32
{
	let cells_wide = width / options.cell_side;
	let cells_high = height / options.cell_side;
	let blocks_wide = num_blocks(cells_wide, options.block_side, options.block_stride);
	let blocks_high = num_blocks(cells_high, options.block_side, options.block_stride);
	options.orientations * blocks_high * blocks_wide * options.block_side.pow(2)
}

// One function to get the dim, another to produce size from a dim
struct HogDim {
	cells_wide: u32,
	cells_high: u32
}


/// Computes orientation histograms for each cell of an image, or none if the
/// image dimensions are incompatible with the provided options.
// TODO: produce a helpful error message if the options are invalid
fn cell_histograms<I>(image: &I, options: HogOptions) -> Option<HistGrid>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();
    if !is_valid_size(width, height, options) {
        return None;
    }

	let cells_wide = width / options.cell_side;
	let cells_high = height / options.cell_side;
	let mut grid = Array3d::new(options.orientations, cells_wide, cells_high);

	let horizontal = horizontal_sobel(image);
	let vertical = vertical_sobel(image);
	let cell_area = (options.cell_side * options.cell_side) as f32;

	let dir_range = if options.signed {2f32 * f32::consts::PI} else {f32::consts::PI};
	let dir_bin_width = dir_range / (options.orientations as f32);

	for y in 0..height {
		let y_inter = Interpolation::from_position(y as f32 / options.cell_side as f32);

		for x in 0..width {
			let x_inter = Interpolation::from_position(x as f32 / options.cell_side as f32);

			let h = horizontal.get_pixel(x, y)[0] as f32;
			let v = vertical.get_pixel(x, y)[0] as f32;
			let m = (h.powi(2) + v.powi(2)).sqrt();
			let d = v.atan2(h);

			let o_inter = Interpolation::from_position_wrapping(
				d / dir_bin_width, options.orientations);

			for iy in 0..2usize {
				for ix in 0..2usize {
					for io in 0..2usize {
						if within_bounds(&grid, ix as u32, iy as u32) {
							let wy = y_inter.weights[iy];
							let wx = x_inter.weights[ix];
							let wo = o_inter.weights[io];
							let py = y_inter.indices[iy];
							let px = x_inter.indices[ix];
							let po = o_inter.indices[io];
							let up = wy * wx * wo * m / cell_area;
						 	*grid.at(po, px, py) = *grid.at(po, px, py) + up;
						}
					}
				}
			}
		}
	}

	Some(grid)
}

/// Indices and weights for an interpolated value.
struct Interpolation {
	indices: [u32; 2],
	weights: [f32; 2]
}

impl Interpolation {
	/// Interpolates between two indices, without wrapping.
	fn from_position(pos: f32) -> Interpolation {
		let fraction = pos - pos.floor();
		Interpolation {
			indices: [pos as u32, pos as u32 + 1],
			weights: [1f32 - fraction, fraction]
		}
	}

	/// Interpolates between two indices, wrapping the right index.
	/// Assumes that the left index is within bounds.
	fn from_position_wrapping(pos: f32, length: u32) -> Interpolation {
		let mut right = (pos as u32) + 1;
		if right >= length {
			right = 0;
		}
		let fraction = pos - pos.floor();
		Interpolation {
			indices: [pos as u32, right],
			weights: [1f32 - fraction, fraction]
		}
	}
}

type HistGrid = Array3d<f32>;

fn within_bounds(grid: &HistGrid, x: u32, y: u32) -> bool {
	x < grid.y_len && y < grid.z_len
}

/// A dense 3d array. x is the fastest varying coordinate, y
/// the next fastest.
pub struct Array3d<T> {
    data: Vec<T>,
	pub x_len: u32,
    pub y_len: u32,
    pub z_len: u32
}

impl<T: Zero + Clone> Array3d<T> {
    pub fn new(x_len: u32, y_len: u32, z_len: u32) -> Array3d<T> {
        Array3d {
            data: vec![Zero::zero(); (x_len * y_len * z_len) as usize],
            x_len: x_len,
            y_len: y_len,
			z_len: z_len
        }
    }

	pub fn at(&mut self, x: u32, y: u32, z: u32) -> &mut T {
		let idx = z * self.y_len * self.x_len + y * self.x_len + x;
		&mut self.data[idx as usize]
	}
}

/// Number of blocks required to cover num_cells cells when each block is
/// block_side long and blocks are staggered by block_stride. Assumes that
/// options are compatible (see is_valid_size function).
fn num_blocks(num_cells: u32, block_side: u32, block_stride: u32) -> u32
{
	(num_cells + block_stride - block_side) / block_stride
}

/// Returns false if an image of the given size can't be evenly covered by
/// cells and blocks of the size and stride specified in options.
fn is_valid_size(width: u32, height: u32, options: HogOptions) -> bool {

    if width % options.cell_side != 0 {
        return false;
    }
    if height % options.cell_side != 0 {
        return false;
    }
    if (width - options.block_side) % options.block_stride != 0 {
        return false;
    }
    if (height - options.block_side) % options.block_stride != 0 {
        return false;
    }
    true
}

#[cfg(test)]
mod test {

    use super::{
        HogOptions,
        hog_size,
        num_blocks
    };

    #[test]
    fn test_num_blocks() {
        // -----
        // ***
        //   ***
        assert_eq!(num_blocks(5, 3, 2), 2);
        // -----
        // *****
        assert_eq!(num_blocks(5, 5, 2), 1);
        // ----
        // **
        //   **
        assert_eq!(num_blocks(4, 2, 2), 2);
        // ---
        // *
        //  *
        //   *
        assert_eq!(num_blocks(3, 1, 1), 3);
    }

    #[test]
    fn test_hog_size() {
        assert_eq!(hog_size(40, 40, HogOptions::new(8, true, 5, 2, 1)), 1568);
        assert_eq!(hog_size(40, 40, HogOptions::new(9, true, 4, 2, 1)), 2916);
        assert_eq!(hog_size(40, 40, HogOptions::new(8, true, 4, 2, 1)), 2592);
    }
}
