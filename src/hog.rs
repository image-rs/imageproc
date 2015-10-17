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
use multiarray::{
	Array3d,
	element_at_mut,
	inner_dimension_slice,
	Mut3d,
	View3d
};
use math::l2_norm;
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

/// Dimensions of a HoG descriptor. The descriptor for the whole
/// image is the concatenation of a blocks_wide * blocks_high grid
/// of block descriptors, each of size block_descriptor_size.
pub struct HogDimensions {
	/// Size of the descriptor for a single block.
	pub block_descriptor_size: u32,
	/// Number of (possibly overlapping) blocks used to tile the image width.
	pub blocks_wide: u32,
	/// Number of (possibly overlapping) blocks used to tile the image height.
	pub blocks_high: u32
}

impl HogDimensions {
	/// Size of a block descriptor using the given options, and number of blocks required
	/// to tile an image of given width and height.
	pub fn from_options(width: u32, height: u32, options: HogOptions) -> HogDimensions {
		let cells_wide = width / options.cell_side;
		let cells_high = height / options.cell_side;
		HogDimensions {
			block_descriptor_size: options.orientations * options.block_side.pow(2),
			blocks_wide: num_blocks(cells_wide, options.block_side, options.block_stride),
			blocks_high: num_blocks(cells_high, options.block_side, options.block_stride)
		}
	}

	/// The total size in floats of the HoG descriptor with these dimensions.
	pub fn descriptor_length(&self) -> u32 {
		self.block_descriptor_size * self.blocks_wide * self.blocks_high
	}
}

/// Computes the HoG descriptor of an image, or None if the provided
/// options are incompatible with the image size.
// TODO: produce a helpful error message if the options are invalid
// TODO: support color images by taking the channel with maximum gradient at each point
pub fn hog<I>(image: &I, options: HogOptions) -> Option<Vec<f32>>
	where I: GenericImage<Pixel=Luma<u8>> + 'static {

	let mut grid: Array3d<f32>;
	match cell_histograms(image, options) {
		None => return None,
		Some(array) => grid = array
	};

	let (width, height) = image.dimensions();
	let dimensions = HogDimensions::from_options(width, height, options);
	let cells_wide = width / options.cell_side;
	let cells_high = height / options.cell_side;
	let mut descriptor = Array3d::new(
		dimensions.block_descriptor_size as usize,
		dimensions.blocks_wide as usize,
		dimensions.blocks_high as usize);

	for by in 0..dimensions.blocks_high {
		for bx in 0..dimensions.blocks_wide {
			let mut block_data = inner_dimension_slice(&mut descriptor, bx as usize, by as usize);
			let mut block = View3d::from_raw(
				&mut block_data,
				options.orientations as usize,
				cells_wide as usize,
				cells_high as usize);

			for iy in 0..options.block_side {
				let cy = by * options.block_stride + iy;

				for ix in 0..options.block_side {
					let cx = bx * options.block_stride + ix;
					let mut slice = inner_dimension_slice(&mut block, ix as usize, iy as usize);
					let hist = inner_dimension_slice(&mut grid, cx as usize, cy as usize);
					for dir in 0..options.orientations {
						slice[dir as usize] = hist[dir as usize];
					}
					let norm = l2_norm(slice);
					for dir in 0..options.orientations {
						slice[dir as usize] /= norm;
					}
				}
			}
		}
	}

	Some(descriptor.data)
}

/// Computes orientation histograms for each cell of an image, or none if the
/// image dimensions are incompatible with the provided options.
// TODO: produce a helpful error message if the options are invalid
fn cell_histograms<I>(image: &I, options: HogOptions) -> Option<Array3d<f32>>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();
    if !is_valid_size(width, height, options) {
        return None;
    }

	let cells_wide = width / options.cell_side;
	let cells_high = height / options.cell_side;
	let mut grid = Array3d::new(
		options.orientations as usize, cells_wide as usize, cells_high as usize);
	let grid_dim = grid.dimensions();

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
						if ix >= grid_dim.len_1 || iy >= grid_dim.len_2 {
							continue;
						}

						let wy = y_inter.weights[iy];
						let wx = x_inter.weights[ix];
						let wo = o_inter.weights[io];
						let py = y_inter.indices[iy] as usize;
						let px = x_inter.indices[ix] as usize;
						let po = o_inter.indices[io] as usize;
						let up = wy * wx * wo * m / cell_area;
						let current = *element_at_mut(&mut grid, po, px, py);
					 	*element_at_mut(&mut grid, po, px, py) = current + up;
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
        HogDimensions,
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
    fn test_hog_dimensions() {
        assert_eq!(
			HogDimensions::from_options(40, 40, HogOptions::new(8, true, 5, 2, 1))
				.descriptor_length(), 1568);
        assert_eq!(
			HogDimensions::from_options(40, 40, HogOptions::new(9, true, 4, 2, 1))
				.descriptor_length(), 2916);
        assert_eq!(
			HogDimensions::from_options(40, 40, HogOptions::new(8, true, 4, 2, 1))
				.descriptor_length(), 2592);
    }
}
