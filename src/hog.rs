//! Histogram of oriented gradients and helpers for visualizing them.
//! http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

use image::{
	GenericImage,
	ImageBuffer,
	Luma
};
use gradients::{
	horizontal_sobel,
	vertical_sobel
};
use multiarray::{
	Array3d,
	View3d
};
use definitions::{
	Clamp,
	VecBuffer
};
use math::l2_norm;
use std::f32;

/// Parameters for HoG descriptors.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HogOptions {
	/// Number of gradient orientation bins.
	pub orientations: usize,
	/// Whether gradients in opposite directions are treated as equal.
    pub signed: bool,
	/// Width and height of cell in pixels.
	pub cell_side: usize,
	/// Width and height of block in cells.
	pub block_side: usize,
	/// Offset of the start of one block from the next in cells.
	pub block_stride: usize
	// TODO: choice of normalisation - for now we just scale to unit L2 norm
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

/// HoG options plus values calculated from these options and the desired
/// image dimensions. Validation must occur when instances of this struct
/// are created - functions receiving a spec will assume that it is valid.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HogSpec {
	/// Original options.
	options: HogOptions,
	/// Number of non-overlapping cells required to cover the image's width.
	cells_wide: usize,
	/// Number of non-overlapping cells required to cover the image height.
	cells_high: usize,
	/// Number of (possibly overlapping) blocks required to cover the image's width.
	blocks_wide: usize,
	/// Number of (possibly overlapping) blocks required to cover the image's height.
	blocks_high: usize
}

impl HogSpec {
	/// Returns None if image dimensions aren't compatible with the provided options.
	pub fn from_options(width: u32, height: u32, options: HogOptions) -> Option<HogSpec> {
		let w = width as usize;
		let h = height as usize;
		let divides_evenly = |n, q| n % q == 0;

		if !divides_evenly(w, options.cell_side) ||
		   !divides_evenly(h, options.cell_side) ||
		   !divides_evenly(w - options.block_side, options.block_stride) ||
		   !divides_evenly(h - options.block_side, options.block_stride) {
            return None;
	   	}

		let cells_wide = w / options.cell_side;
		let cells_high = h / options.cell_side;
		Some(HogSpec {
			options: options,
			cells_wide: cells_wide,
			cells_high: cells_high,
			blocks_wide: num_blocks(cells_wide, options.block_side, options.block_stride),
			blocks_high: num_blocks(cells_high, options.block_side, options.block_stride)
		})
	}

	/// The total size in floats of the HoG descriptor with these dimensions.
	pub fn descriptor_length(&self) -> usize {
		self.blocks_wide * self.blocks_high * self.block_descriptor_length()
	}

	/// The size in floats of the descriptor for a single block.
	fn block_descriptor_length(&self) -> usize {
		self.options.orientations * self.options.block_side.pow(2)
	}

	/// Dimensions of a grid of cell histograms, viewed as a 3d array.
	/// Innermost dimension is orientation bin, then horizontal cell location,
	/// then vertical cell location.
	fn cell_grid_lengths(&self) -> [usize; 3] {
		[self.options.orientations, self.cells_wide, self.cells_high]
	}

	/// Dimensions of a grid of block descriptors, viewed as a 3d array.
	/// Innermost dimension is block descriptor position, then horizontal block location,
	/// then vertical block location.
	fn block_grid_lengths(&self) -> [usize; 3] {
		[self.block_descriptor_length(), self.blocks_wide, self.blocks_high]
	}

	/// Area of an image cell in pixels.
	fn cell_area(&self) -> usize {
		self.options.cell_side * self.options.cell_side
	}
}

/// Number of blocks required to cover num_cells cells when each block is
/// block_side long and blocks are staggered by block_stride. Assumes that
/// options are compatible.
fn num_blocks(num_cells: usize, block_side: usize, block_stride: usize) -> usize
{
	(num_cells + block_stride - block_side) / block_stride
}

/// Computes the HoG descriptor of an image, or None if the provided
/// options are incompatible with the image size.
// TODO: produce a helpful error message if the options are invalid
// TODO: support color images by taking the channel with maximum gradient at each point
pub fn hog<I>(image: &I, options: HogOptions) -> Option<Vec<f32>>
	where I: GenericImage<Pixel=Luma<u8>> + 'static {
	let hog_spec: HogSpec;
	match HogSpec::from_options(image.width(), image.height(), options) {
		None => return None,
		Some(spec) => hog_spec = spec
	}
	Some(hog_impl(image, hog_spec))
}

/// Computes the HoG descriptor of an image. Assumes that the provided dimensions are valid.
fn hog_impl<I>(image: &I, spec: HogSpec) -> Vec<f32>
	where I: GenericImage<Pixel=Luma<u8>> + 'static {

	let mut grid: Array3d<f32> = cell_histograms(image, spec);
	let mut descriptor = Array3d::new(spec.block_grid_lengths());

	for by in 0..spec.blocks_high {
		for bx in 0..spec.blocks_wide {
			let mut block_view = descriptor.view_mut();
			let mut block_data = block_view.inner_slice_mut(bx, by);
			let mut block = View3d::from_raw(&mut block_data, spec.cell_grid_lengths());

			for iy in 0..spec.options.block_side {
				let cy = by * spec.options.block_stride + iy;

				for ix in 0..spec.options.block_side {
					let cx = bx * spec.options.block_stride + ix;
					let mut slice = block.inner_slice_mut(ix, iy);
					let hist_view = grid.view_mut();
					let hist = hist_view.inner_slice(cx, cy);

					// TODO: do this faster
					for dir in 0..spec.options.orientations {
						slice[dir] = hist[dir];
					}
					let norm = l2_norm(slice);
					for dir in 0..spec.options.orientations {
						slice[dir] /= norm;
					}
				}
			}
		}
	}

	descriptor.data
}

/// Computes orientation histograms for each cell of an image. Assumes that
/// the provided dimensions are valid.
pub fn cell_histograms<I>(image: &I, spec: HogSpec) -> Array3d<f32>
    where I: GenericImage<Pixel=Luma<u8>> + 'static {

    let (width, height) = image.dimensions();
	let mut grid = Array3d::new(spec.cell_grid_lengths());
	let cell_area = spec.cell_area() as f32;
	let cell_side = spec.options.cell_side as f32;
	let horizontal = horizontal_sobel(image);
	let vertical = vertical_sobel(image);
	let interval = orientation_bin_width(spec.options);

	for y in 0..height {
		let mut grid_view = grid.view_mut();
		let y_inter = Interpolation::from_position(y as f32 / cell_side);

		for x in 0..width {
			let x_inter = Interpolation::from_position(x as f32 / cell_side);

			let h = horizontal.get_pixel(x, y)[0] as f32;
			let v = vertical.get_pixel(x, y)[0] as f32;
			let m = (h.powi(2) + v.powi(2)).sqrt();
			let mut d = v.atan2(h);
			if d < 0f32 {
				d = d + 2f32 * f32::consts::PI;
			}

			let o_inter
				= Interpolation::from_position_wrapping(d / interval, spec.options.orientations);

			for iy in 0..2usize {
				let py = y_inter.indices[iy];

				for ix in 0..2usize {
					let px = x_inter.indices[ix];

					for io in 0..2usize {
						let po = o_inter.indices[io];
						if contains_outer(&grid_view, px, py) {
							let wy = y_inter.weights[iy];
							let wx = x_inter.weights[ix];
							let wo = o_inter.weights[io];
							let up = wy * wx * wo * m / cell_area;
							let current = *grid_view.at_mut([po, px, py]);
						 	*grid_view.at_mut([po, px, py]) = current + up;
						}
					}
				}
			}
		}
	}

	grid
}

/// True if the given outer two indices into a view are within bounds.
fn contains_outer<T>(view: &View3d<T>, u: usize, v: usize) -> bool {
	u < view.lengths[1] && v < view.lengths[2]
}

/// Width of an orientation histogram bin in radians.
fn orientation_bin_width(options: HogOptions) -> f32 {
	let dir_range = if options.signed {2f32 * f32::consts::PI} else {f32::consts::PI};
	dir_range / (options.orientations as f32)
}

/// Indices and weights for an interpolated value.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Interpolation {
	indices: [usize; 2],
	weights: [f32; 2]
}

impl Interpolation {

	/// Creates new interpolation with provided indices and weights.
	fn new (indices: [usize; 2], weights: [f32; 2]) -> Interpolation {
		Interpolation { indices: indices, weights: weights }
	}

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

/// Draws a ray from the center of an image, in a direction theta radians
/// clockwise from the y axis (recall that image coordinates increase from
/// top left to bottom right).
pub fn draw_ray<I>(image: &I, theta: f32, color: I::Pixel)
		-> VecBuffer<I::Pixel>
	where I: GenericImage, I::Pixel: 'static {
	let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0);
    draw_ray_mut(&mut out, theta, color);
    out
}

/// Draws a ray from the center of an image in place, in a direction theta radians
/// clockwise from the y axis (recall that image coordinates increase from
/// top left to bottom right).
pub fn draw_ray_mut<I>(image: &mut I, theta: f32, color: I::Pixel)
    where I: GenericImage, I::Pixel: 'static {

	use std::cmp;
	use drawing::draw_line_segment_mut;

	let (width, height) = image.dimensions();
	let scale = cmp::max(width, height) as f32 / 2f32;
	let start_x = (width / 2) as f32;
	let start_y = (height / 2) as f32;
	let start = (start_x, start_y);
	let x_step = -scale * theta.sin();
	let y_step = scale * theta.cos();
	let end = (start_x + x_step, start_y + y_step);

	draw_line_segment_mut(image, start, end, color);
}

/// Visualises a histogram of edge orientation strengths as a collection of rays
/// emanating from the centre of a square image. The intensity of each ray is
/// proportional to the value of the bucket centred on its direction.
pub fn star(side_length: u32, hist: &[f32], signed: bool) -> VecBuffer<Luma<u8>> {
	let mut out = ImageBuffer::new(side_length, side_length);
	draw_star_mut(&mut out, hist, signed);
	out
}

/// Draws a visualisation of a histogram of edge orientation strengths as a collection of rays
/// emanating from the centre of a square image. The intensity of each ray is
/// proportional to the value of the bucket centred on its direction.
pub fn draw_star_mut<I>(image: &mut I, hist: &[f32], signed: bool)
	where I: GenericImage<Pixel=Luma<u8>> {
	let orientations = hist.len() as f32;
	for bucket in 0..hist.len() {
		if signed {
			let dir = (2f32 * f32::consts::PI * bucket as f32) / orientations;
			let intensity = u8::clamp(hist[bucket]);
			draw_ray_mut(image, dir, Luma([intensity]));
		}
		else {
			let dir = (f32::consts::PI * bucket as f32) / orientations;
			let intensity = u8::clamp(hist[bucket]);
			draw_ray_mut(image, dir, Luma([intensity]));
			draw_ray_mut(image, dir + f32::consts::PI, Luma([intensity]));
		}
	}
}

/// Visualises an array of orientation histograms.
/// The dimensions of the provided Array3d are orientation bucket,
/// horizontal location of the cell, then vertical location of the cell.
/// Note that we ignore block-level aggregation or normalisation here.
/// Each rendered star has side length star_side, so the image will have
/// width grid.lengths[1] * star_side and height grid.lengths[2] * star_side.
pub fn render_hist_grid(star_side: u32, grid: &View3d<f32>, signed: bool) -> VecBuffer<Luma<u8>> {
	let width = grid.lengths[1] as u32 * star_side;
	let height = grid.lengths[2] as u32 * star_side;
	let mut out = ImageBuffer::new(width, height);

	for y in 0..grid.lengths[2] {
		let y_window = y as u32 * star_side;
		for x in 0..grid.lengths[1] {
			let x_window = x as u32 * star_side;
			let mut window = out.sub_image(x_window, y_window, star_side, star_side);
			let hist = grid.inner_slice(x, y);
			draw_star_mut(&mut window, hist, signed);
		}
	}

	out
}

#[cfg(test)]
mod test {

    use super::{
        HogOptions,
        HogSpec,
		Interpolation,
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
    fn test_hog_spec() {
        assert_eq!(
			HogSpec::from_options(40, 40, HogOptions::new(8, true, 5, 2, 1))
				.unwrap().descriptor_length(), 1568);
        assert_eq!(
			HogSpec::from_options(40, 40, HogOptions::new(9, true, 4, 2, 1))
				.unwrap().descriptor_length(), 2916);
        assert_eq!(
			HogSpec::from_options(40, 40, HogOptions::new(8, true, 4, 2, 1))
				.unwrap().descriptor_length(), 2592);
    }

	#[test]
	fn test_interpolation_from_position() {
		assert_eq!(Interpolation::from_position(10f32),
			Interpolation::new([10, 11], [1f32, 0f32]));
		assert_eq!(Interpolation::from_position(10.25f32),
			Interpolation::new([10, 11], [0.75f32, 0.25f32]));
	}

	#[test]
	fn test_interpolation_from_position_wrapping() {
		assert_eq!(Interpolation::from_position_wrapping(10f32, 11),
			Interpolation::new([10, 0], [1f32, 0f32]));
		assert_eq!(Interpolation::from_position_wrapping(10.25f32, 11),
			Interpolation::new([10, 0], [0.75f32, 0.25f32]));
		assert_eq!(Interpolation::from_position_wrapping(10f32, 12),
			Interpolation::new([10, 11], [1f32, 0f32]));
		assert_eq!(Interpolation::from_position_wrapping(10.25f32, 12),
			Interpolation::new([10, 11], [0.75f32, 0.25f32]));
	}
}
