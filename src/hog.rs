//! [HoG features](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
//! and helpers for visualizing them.

use crate::definitions::{Clamp, Image};
use crate::gradients::{horizontal_sobel, vertical_sobel};
use crate::math::l2_norm;
use image::{GenericImage, GrayImage, ImageBuffer, Luma};
use num::Zero;
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
    pub block_stride: usize, // TODO: choice of normalisation - for now we just scale to unit L2 norm
}

impl HogOptions {
    /// User-provided options, prior to validation.
    pub fn new(
        orientations: usize,
        signed: bool,
        cell_side: usize,
        block_side: usize,
        block_stride: usize,
    ) -> HogOptions {
        HogOptions {
            orientations,
            signed,
            cell_side,
            block_side,
            block_stride,
        }
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
    blocks_high: usize,
}

impl HogSpec {
    /// Returns an error message if image dimensions aren't compatible with the provided options.
    pub fn from_options(width: u32, height: u32, options: HogOptions) -> Result<HogSpec, String> {
        let (cells_wide, cells_high) =
            Self::checked_cell_dimensions(width as usize, height as usize, options)?;
        let (blocks_wide, blocks_high) =
            Self::checked_block_dimensions(cells_wide, cells_high, options)?;
        Ok(HogSpec {
            options,
            cells_wide,
            cells_high,
            blocks_wide,
            blocks_high,
        })
    }

    fn invalid_options_message(errors: &[String]) -> String {
        format!("Invalid HoG options: {0}", errors.join(", "))
    }

    /// Returns (cells wide, cells high), or an error message if cell side doesn't evenly divide width and height.
    fn checked_cell_dimensions(
        width: usize,
        height: usize,
        options: HogOptions,
    ) -> Result<(usize, usize), String> {
        let mut errors: Vec<String> = vec![];
        if width % options.cell_side != 0 {
            errors.push(format!(
                "cell side {} does not evenly divide width {}",
                options.cell_side, width
            ));
        }
        if height % options.cell_side != 0 {
            errors.push(format!(
                "cell side {} does not evenly divide height {}",
                options.cell_side, height
            ));
        }
        if !errors.is_empty() {
            return Err(Self::invalid_options_message(&errors));
        }
        Ok((width / options.cell_side, height / options.cell_side))
    }

    /// Returns (blocks wide, blocks high), or an error message if the block size and stride don't evenly cover
    /// the grid of cells.
    fn checked_block_dimensions(
        cells_wide: usize,
        cells_high: usize,
        options: HogOptions,
    ) -> Result<(usize, usize), String> {
        let mut errors: Vec<String> = vec![];
        if (cells_wide - options.block_side) % options.block_stride != 0 {
            errors.push(format!(
                "block stride {} does not evenly divide (cells wide {} - block side {})",
                options.block_stride, cells_wide, options.block_side
            ));
        }
        if (cells_high - options.block_side) % options.block_stride != 0 {
            errors.push(format!(
                "block stride {} does not evenly divide (cells high {} - block side {})",
                options.block_stride, cells_high, options.block_side
            ));
        }
        if !errors.is_empty() {
            return Err(Self::invalid_options_message(&errors));
        }
        Ok((
            num_blocks(cells_wide, options.block_side, options.block_stride),
            num_blocks(cells_high, options.block_side, options.block_stride),
        ))
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
        [
            self.block_descriptor_length(),
            self.blocks_wide,
            self.blocks_high,
        ]
    }

    /// Dimensions of a single block descriptor, viewed as a 3d array.
    /// Innermost dimension is histogram bin, then horizontal cell location, then
    /// vertical cell location.
    fn block_internal_lengths(&self) -> [usize; 3] {
        [
            self.options.orientations,
            self.options.block_side,
            self.options.block_side,
        ]
    }

    /// Area of an image cell in pixels.
    fn cell_area(&self) -> usize {
        self.options.cell_side * self.options.cell_side
    }
}

/// Number of blocks required to cover `num_cells` cells when each block is
/// `block_side` long and blocks are staggered by `block_stride`. Assumes that
/// options are compatible.
fn num_blocks(num_cells: usize, block_side: usize, block_stride: usize) -> usize {
    (num_cells + block_stride - block_side) / block_stride
}

/// Computes the HoG descriptor of an image, or None if the provided
/// options are incompatible with the image size.
// TODO: support color images by taking the channel with maximum gradient at each point
pub fn hog(image: &GrayImage, options: HogOptions) -> Result<Vec<f32>, String> {
    match HogSpec::from_options(image.width(), image.height(), options) {
        Err(e) => Err(e),
        Ok(spec) => {
            let mut grid: Array3d<f32> = cell_histograms(image, spec);
            let grid_view = grid.view_mut();
            let descriptor = hog_descriptor_from_hist_grid(grid_view, spec);
            Ok(descriptor)
        }
    }
}

/// Computes the HoG descriptor of an image. Assumes that the spec and grid
/// dimensions are consistent.
fn hog_descriptor_from_hist_grid(grid: View3d<'_, f32>, spec: HogSpec) -> Vec<f32> {
    let mut descriptor = Array3d::new(spec.block_grid_lengths());
    {
        let mut block_view = descriptor.view_mut();

        for by in 0..spec.blocks_high {
            for bx in 0..spec.blocks_wide {
                let mut block_data = block_view.inner_slice_mut(bx, by);
                let mut block = View3d::from_raw(&mut block_data, spec.block_internal_lengths());

                for iy in 0..spec.options.block_side {
                    let cy = by * spec.options.block_stride + iy;
                    for ix in 0..spec.options.block_side {
                        let cx = bx * spec.options.block_stride + ix;
                        let slice = block.inner_slice_mut(ix, iy);
                        let hist = grid.inner_slice(cx, cy);
                        copy(hist, slice);
                    }
                }
            }
        }

        for by in 0..spec.blocks_high {
            for bx in 0..spec.blocks_wide {
                let norm = block_norm(&block_view, bx, by);
                if norm > 0f32 {
                    let block_mut = block_view.inner_slice_mut(bx, by);
                    for i in 0..block_mut.len() {
                        block_mut[i] /= norm;
                    }
                }
            }
        }
    }

    descriptor.data
}

/// L2 norm of the block descriptor at given location within an image descriptor.
fn block_norm(view: &View3d<'_, f32>, bx: usize, by: usize) -> f32 {
    let block_data = view.inner_slice(bx, by);
    l2_norm(block_data)
}

fn copy<T: Copy>(from: &[T], to: &mut [T]) {
    to.clone_from_slice(&from[..to.len()]);
}

/// Computes orientation histograms for each cell of an image. Assumes that
/// the provided dimensions are valid.
pub fn cell_histograms(image: &GrayImage, spec: HogSpec) -> Array3d<f32> {
    let (width, height) = image.dimensions();
    let mut grid = Array3d::new(spec.cell_grid_lengths());
    let cell_area = spec.cell_area() as f32;
    let cell_side = spec.options.cell_side as f32;
    let horizontal = horizontal_sobel(image);
    let vertical = vertical_sobel(image);
    let interval = orientation_bin_width(spec.options);
    let range = direction_range(spec.options);

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
                d += range;
            }
            if !spec.options.signed && d >= f32::consts::PI {
                d -= f32::consts::PI;
            }

            let o_inter =
                Interpolation::from_position_wrapping(d / interval, spec.options.orientations);

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
fn contains_outer<T>(view: &View3d<'_, T>, u: usize, v: usize) -> bool {
    u < view.lengths[1] && v < view.lengths[2]
}

/// Width of an orientation histogram bin in radians.
fn orientation_bin_width(options: HogOptions) -> f32 {
    direction_range(options) / (options.orientations as f32)
}

/// Length of the range of possible directions in radians.
fn direction_range(options: HogOptions) -> f32 {
    if options.signed {
        2f32 * f32::consts::PI
    } else {
        f32::consts::PI
    }
}

/// Indices and weights for an interpolated value.
#[derive(Debug, Copy, Clone, PartialEq)]
struct Interpolation {
    indices: [usize; 2],
    weights: [f32; 2],
}

impl Interpolation {
    /// Creates new interpolation with provided indices and weights.
    fn new(indices: [usize; 2], weights: [f32; 2]) -> Interpolation {
        Interpolation { indices, weights }
    }

    /// Interpolates between two indices, without wrapping.
    fn from_position(pos: f32) -> Interpolation {
        let fraction = pos - pos.floor();
        Self::new(
            [pos as usize, pos as usize + 1],
            [1f32 - fraction, fraction],
        )
    }

    /// Interpolates between two indices, wrapping the right index.
    /// Assumes that the left index is within bounds.
    fn from_position_wrapping(pos: f32, length: usize) -> Interpolation {
        let mut right = (pos as usize) + 1;
        if right >= length {
            right = 0;
        }
        let fraction = pos - pos.floor();
        Self::new([pos as usize, right], [1f32 - fraction, fraction])
    }
}

/// Visualises an array of orientation histograms.
/// The dimensions of the provided Array3d are orientation bucket,
/// horizontal location of the cell, then vertical location of the cell.
/// Note that we ignore block-level aggregation or normalisation here.
/// Each rendered star has side length `star_side`, so the image will have
/// width grid.lengths[1] * `star_side` and height grid.lengths[2] * `star_side`.
pub fn render_hist_grid(star_side: u32, grid: &View3d<'_, f32>, signed: bool) -> Image<Luma<u8>> {
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

/// Draws a ray from the center of an image in place, in a direction theta radians
/// clockwise from the y axis (recall that image coordinates increase from
/// top left to bottom right).
fn draw_ray_mut<I>(image: &mut I, theta: f32, color: I::Pixel)
where
    I: GenericImage,
    I::Pixel: 'static,
{
    use crate::drawing::draw_line_segment_mut;
    use std::cmp;

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

/// Draws a visualisation of a histogram of edge orientation strengths as a collection of rays
/// emanating from the centre of a square image. The intensity of each ray is
/// proportional to the value of the bucket centred on its direction.
fn draw_star_mut<I>(image: &mut I, hist: &[f32], signed: bool)
where
    I: GenericImage<Pixel = Luma<u8>>,
{
    let orientations = hist.len() as f32;
    for bucket in 0..hist.len() {
        if signed {
            let dir = (2f32 * f32::consts::PI * bucket as f32) / orientations;
            let intensity = Clamp::clamp(hist[bucket]);
            draw_ray_mut(image, dir, Luma([intensity]));
        } else {
            let dir = (f32::consts::PI * bucket as f32) / orientations;
            let intensity = Clamp::clamp(hist[bucket]);
            draw_ray_mut(image, dir, Luma([intensity]));
            draw_ray_mut(image, dir + f32::consts::PI, Luma([intensity]));
        }
    }
}

/// A 3d array that owns its data.
pub struct Array3d<T> {
    /// The owned data.
    data: Vec<T>,
    /// Lengths of the dimensions, from innermost (i.e. fastest-varying) to outermost.
    lengths: [usize; 3],
}

/// A view into a 3d array.
pub struct View3d<'a, T> {
    /// The underlying data.
    data: &'a mut [T],
    /// Lengths of the dimensions, from innermost (i.e. fastest-varying) to outermost.
    lengths: [usize; 3],
}

impl<T: Zero + Clone> Array3d<T> {
    /// Allocates a new Array3d with the given dimensions.
    fn new(lengths: [usize; 3]) -> Array3d<T> {
        let data = vec![Zero::zero(); data_length(lengths)];
        Array3d { data, lengths }
    }

    /// Provides a 3d view of the data.
    pub fn view_mut(&mut self) -> View3d<'_, T> {
        View3d::from_raw(&mut self.data, self.lengths)
    }
}

impl<'a, T> View3d<'a, T> {
    /// Constructs index from existing data and the lengths of the desired dimensions.
    fn from_raw(data: &'a mut [T], lengths: [usize; 3]) -> View3d<'a, T> {
        View3d { data, lengths }
    }

    /// A mutable reference from a 3d index.
    fn at_mut(&mut self, indices: [usize; 3]) -> &mut T {
        &mut self.data[self.offset(indices)]
    }

    /// All entries with the given outer dimensions. As the first dimension
    /// is fastest varying, this is a contiguous slice.
    fn inner_slice(&self, x1: usize, x2: usize) -> &[T] {
        let offset = self.offset([0, x1, x2]);
        &self.data[offset..offset + self.lengths[0]]
    }

    /// All entries with the given outer dimensions. As the first dimension
    /// is fastest varying, this is a contiguous slice.
    fn inner_slice_mut(&mut self, x1: usize, x2: usize) -> &mut [T] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gray_bench_image;
    use ::test;

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
    fn test_hog_spec_valid_options() {
        assert_eq!(
            HogSpec::from_options(40, 40, HogOptions::new(8, true, 5, 2, 1))
                .unwrap()
                .descriptor_length(),
            1568
        );
        assert_eq!(
            HogSpec::from_options(40, 40, HogOptions::new(9, true, 4, 2, 1))
                .unwrap()
                .descriptor_length(),
            2916
        );
        assert_eq!(
            HogSpec::from_options(40, 40, HogOptions::new(8, true, 4, 2, 1))
                .unwrap()
                .descriptor_length(),
            2592
        );
    }

    #[test]
    fn test_hog_spec_invalid_options() {
        let opts = HogOptions {
            orientations: 8,
            signed: true,
            cell_side: 3,
            block_side: 4,
            block_stride: 2,
        };
        let expected = "Invalid HoG options: block stride 2 does not evenly divide (cells wide 7 - block side 4), \
			block stride 2 does not evenly divide (cells high 7 - block side 4)";
        assert_eq!(
            HogSpec::from_options(21, 21, opts),
            Err(expected.to_owned())
        );
    }

    #[test]
    fn test_interpolation_from_position() {
        assert_eq!(
            Interpolation::from_position(10f32),
            Interpolation::new([10, 11], [1f32, 0f32])
        );
        assert_eq!(
            Interpolation::from_position(10.25f32),
            Interpolation::new([10, 11], [0.75f32, 0.25f32])
        );
    }

    #[test]
    fn test_interpolation_from_position_wrapping() {
        assert_eq!(
            Interpolation::from_position_wrapping(10f32, 11),
            Interpolation::new([10, 0], [1f32, 0f32])
        );
        assert_eq!(
            Interpolation::from_position_wrapping(10.25f32, 11),
            Interpolation::new([10, 0], [0.75f32, 0.25f32])
        );
        assert_eq!(
            Interpolation::from_position_wrapping(10f32, 12),
            Interpolation::new([10, 11], [1f32, 0f32])
        );
        assert_eq!(
            Interpolation::from_position_wrapping(10.25f32, 12),
            Interpolation::new([10, 11], [0.75f32, 0.25f32])
        );
    }

    #[test]
    fn test_hog_descriptor_from_hist_grid() {
        // A grid of cells 3 wide and 2 high. Each cell contains a histogram of 2 items.
        // There are two blocks, the left covering the leftmost 2x2 region, and the
        // right covering the rightmost 2x2 region. These regions overlap by one cell column.
        // There's no significance to the contents of the histograms used here, we're
        // just checking that the values are binned and normalised correctly.
        let opts = HogOptions {
            orientations: 2,
            signed: true,
            cell_side: 5,
            block_side: 2,
            block_stride: 1,
        };

        let spec = HogSpec::from_options(15, 10, opts).unwrap();

        let mut grid = Array3d::<f32>::new([2, 3, 2]);
        let mut view = grid.view_mut();

        {
            let tl = view.inner_slice_mut(0, 0);
            copy(&[1f32, 3f32, 2f32], tl);
        }
        {
            let tm = view.inner_slice_mut(1, 0);
            copy(&[2f32, 3f32, 5f32], tm);
        }
        {
            let tr = view.inner_slice_mut(2, 0);
            copy(&[0f32, 1f32, 0f32], tr);
        }
        {
            let bl = view.inner_slice_mut(0, 1);
            copy(&[5f32, 0f32, 7f32], bl);
        }
        {
            let bm = view.inner_slice_mut(1, 1);
            copy(&[3f32, 7f32, 9f32], bm);
        }
        {
            let br = view.inner_slice_mut(2, 1);
            copy(&[6f32, 1f32, 4f32], br);
        }

        let descriptor = hog_descriptor_from_hist_grid(view, spec);
        assert_eq!(descriptor.len(), 16);

        let counts = [1, 3, 2, 3, 5, 0, 3, 7, 2, 3, 0, 1, 3, 7, 6, 1];
        let mut expected = [0f32; 16];

        let left_norm = 106f32.sqrt();
        let right_norm = 109f32.sqrt();

        for i in 0..8 {
            expected[i] = counts[i] as f32 / left_norm;
        }
        for i in 8..16 {
            expected[i] = counts[i] as f32 / right_norm;
        }

        assert_eq!(descriptor, expected);
    }

    #[test]
    fn test_direction_interpolation_within_bounds() {
        let image = gray_image!(
			2, 1, 0;
			2, 1, 0;
			2, 1, 0);

        let opts_signed = HogOptions {
            orientations: 8,
            signed: true,
            cell_side: 3,
            block_side: 1,
            block_stride: 1,
        };

        let desc_signed = hog(&image, opts_signed);
        test::black_box(desc_signed.unwrap());

        let opts_unsigned = HogOptions {
            orientations: 8,
            signed: false,
            cell_side: 3,
            block_side: 1,
            block_stride: 1,
        };

        let desc_unsigned = hog(&image, opts_unsigned);
        test::black_box(desc_unsigned.unwrap());
    }

    #[bench]
    fn bench_hog(b: &mut test::Bencher) {
        let image = gray_bench_image(88, 88);
        let opts = HogOptions {
            orientations: 8,
            signed: true,
            cell_side: 8,
            block_side: 3,
            block_stride: 2,
        };
        b.iter(|| {
            let desc = hog(&image, opts);
            test::black_box(desc.unwrap());
        });
    }
}
