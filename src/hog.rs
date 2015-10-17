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
	pub orientations: usize,

	/// Whether gradients in opposite directions
	/// are treated as equal.
    pub signed: bool,

	/// Width and height of cell in pixels.
	pub cell_side: usize,

	/// Width and height of block in cells.
	pub block_side: usize,

	/// Offset of the start of one block from the next in cells.
	pub block_stride: usize

	// TODO: choice of normalisation - for now we just scale
    // TODO: to unit L2 norm
}

impl HogOptions {
    pub fn new(orientations: usize, signed: bool, cell_side: usize,
        block_side: usize, block_stride: usize) -> HogOptions {

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
	pub block_descriptor_size: usize,
	/// Number of (possibly overlapping) blocks required to tile the image width.
	pub blocks_wide: usize,
	/// Number of (possibly overlapping) blocks required to tile the image height.
	pub blocks_high: usize,
	/// Number of non-overlapping cells required to tile the image width.
	pub cells_wide: usize,
	/// Number of non-overlapping cells required to tile the image height.
	pub cells_high: usize
}

impl HogDimensions {
	/// Size of a block descriptor using the given options, and number of blocks required
	/// to tile an image of given width and height.
	pub fn from_options(width: u32, height: u32, options: HogOptions) -> HogDimensions {
		let cells_wide = width as usize/ options.cell_side;
		let cells_high = height as usize/ options.cell_side;
		HogDimensions {
			block_descriptor_size: options.orientations * options.block_side.pow(2),
			blocks_wide: num_blocks(cells_wide, options.block_side, options.block_stride),
			blocks_high: num_blocks(cells_high, options.block_side, options.block_stride),
			cells_wide: cells_wide,
			cells_high: cells_high
		}
	}

	/// The total size in floats of the HoG descriptor with these dimensions.
	pub fn descriptor_length(&self) -> usize {
		self.blocks_wide * self.blocks_high * self.block_descriptor_size
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
	let hog_dim = HogDimensions::from_options(width, height, options);
	let mut descriptor = Array3d::new(
		hog_dim.block_descriptor_size,
		hog_dim.blocks_wide,
		hog_dim.blocks_high);

	for by in 0..hog_dim.blocks_high {
		for bx in 0..hog_dim.blocks_wide {
			let mut block_data = inner_dimension_slice(&mut descriptor, bx, by);
			let mut block = View3d::from_raw(&mut block_data, options.orientations,
				hog_dim.cells_wide, hog_dim.cells_high);

			for iy in 0..options.block_side {
				let cy = by * options.block_stride + iy;

				for ix in 0..options.block_side {
					let cx = bx * options.block_stride + ix;
					let mut slice = inner_dimension_slice(&mut block, ix, iy);
					let hist = inner_dimension_slice(&mut grid, cx, cy);
					for dir in 0..options.orientations {
						slice[dir] = hist[dir];
					}
					let norm = l2_norm(slice);
					for dir in 0..options.orientations {
						slice[dir] /= norm;
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

	let dimensions = HogDimensions::from_options(width, height, options);
	let mut grid = Array3d::new(options.orientations, dimensions.cells_wide, dimensions.cells_high);
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
						let py = y_inter.indices[iy];
						let px = x_inter.indices[ix];
						let po = o_inter.indices[io];
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
	indices: [usize; 2],
	weights: [f32; 2]
}

impl Interpolation {
	/// Interpolates between two indices, without wrapping.
	fn from_position(pos: f32) -> Interpolation {
		let fraction = pos - pos.floor();
		Interpolation {
			indices: [pos as usize, pos as usize + 1],
			weights: [1f32 - fraction, fraction]
		}
	}

	/// Interpolates between two indices, wrapping the right index.
	/// Assumes that the left index is within bounds.
	fn from_position_wrapping(pos: f32, length: usize) -> Interpolation {
		let mut right = (pos as usize) + 1;
		if right >= length {
			right = 0;
		}
		let fraction = pos - pos.floor();
		Interpolation {
			indices: [pos as usize, right],
			weights: [1f32 - fraction, fraction]
		}
	}
}

/// Number of blocks required to cover num_cells cells when each block is
/// block_side long and blocks are staggered by block_stride. Assumes that
/// options are compatible (see is_valid_size function).
fn num_blocks(num_cells: usize, block_side: usize, block_stride: usize) -> usize
{
	(num_cells + block_stride - block_side) / block_stride
}

/// Returns false if an image of the given size can't be evenly covered by
/// cells and blocks of the size and stride specified in options.
fn is_valid_size(width: u32, height: u32, options: HogOptions) -> bool {

    if width as usize % options.cell_side != 0 {
        return false;
    }
    if height as usize % options.cell_side != 0 {
        return false;
    }
    if (width as usize - options.block_side) % options.block_stride != 0 {
        return false;
    }
    if (height as usize - options.block_side) % options.block_stride != 0 {
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
