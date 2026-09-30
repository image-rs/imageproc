//! An implementation of [seam carving].
//!
//! Uses the w2 algorithm from
//! Huang, H., Fu, T., Rosin, P.L. et al.
//! Real-time content-aware image resizing. Sci. China Ser. F-Inf. Sci. 52, 172–182 (2009).
//! <https://doi.org/10.1007/s11432-009-0041-9>
//!
//! Archived copy: <https://web.archive.org/web/20120404113111/https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/seam-carving-ChinaF.pdf>
//!
//!
//! # Examples
//!
//! Shrink an image to a given width using [`shrink_width`]:
//!
//! ```
//! use std::path::PathBuf;
//! use imageproc::seam_carving::shrink_width;
//!
//! let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//!     .join("tests/data/elephant.png");
//! let image = image::open(path).unwrap().into_rgb8();
//!
//! let result = shrink_width(&image, 250);
//! assert_eq!(result.width(), 250);
//! assert_eq!(result.height(), image.height());
//! ```
//!
//! Shrink both dimensions by using [`shrink_width`], rotating the image,
//! [`shrink_width`] again and rotate back:
//!
//! ```
//! use std::path::PathBuf;
//! use imageproc::{seam_carving::shrink_width, geometric_transformations::{rotate90, rotate270}};
//!
//! let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//!     .join("tests/data/elephant.png");
//! let image = image::open(path).unwrap().into_rgb8();
//!
//! let target_width = 250;
//! let target_height = 150;
//!
//! let shrunk_once = shrink_width(&image, target_width);
//! let rotated = rotate90(&shrunk_once);
//! let carved_height = shrink_width(&rotated, target_height);
//! let result = rotate270(&carved_height);
//!
//! assert_eq!(result.width(), target_width);
//! assert_eq!(result.height(), target_height);
//! ```
//!
//! See `examples/seam_carving.rs` for more usage examples.
//!
//! [seam carving]: https://en.wikipedia.org/wiki/Seam_carving

use crate::{definitions::Image, map::map_pixels};

use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};

/// An image seam connecting the bottom of an image to its top (in that order).
#[derive(Debug, Clone)]
pub struct VerticalSeam(Vec<u32>);

/// Find a single vertical seam.
/// See [`find_vertical_seams`] for finding multiple seams at once.
#[deprecated(note = "prefer `find_vertical_seams` instead, it finds multiple seams at once")]
pub fn find_vertical_seam<P>(image: &Image<P>) -> VerticalSeam
where
    P: Pixel<Subpixel = u8>,
{
    let seams = find_vertical_seams(image);

    seams.to_vec_lowest(1).swap_remove(0)
}

/// Returns the result of removing `seam` from `image`.
///
/// See [`remove_vertical_seams`] for removing multiple seams at once
// This should just mutate an image in place. The problem is that we don't have a
// way of talking about views of ImageBuffer without devolving into supporting
// arbitrary GenericImages. And a lot of other functions don't support those because
// it would make them a lot slower.
#[deprecated(note = "prefer `remove_vertical_seams` instead, it removes multiple seams at once")]
pub fn remove_vertical_seam<P>(image: &Image<P>, seam: &VerticalSeam) -> Image<P>
where
    P: Pixel,
{
    assert!(
        seam.0.len() as u32 == image.height(),
        "seam length does not match image height"
    );

    let (width, height) = image.dimensions();
    let mut out = image.buffer_with_dimensions(width - 1, height);

    for y in 0..height {
        let x_seam = seam.0[(height - y - 1) as usize];
        for x in 0..x_seam {
            out.put_pixel(x, y, *image.get_pixel(x, y));
        }
        for x in (x_seam + 1)..width {
            out.put_pixel(x - 1, y, *image.get_pixel(x, y));
        }
    }

    out
}

/// Draws a series of `seams` on `image` in red. Assumes that the provided seams were
/// removed in the given order from the input image.
///
/// Incompatible with VerticalSeam array returned by
/// [`VerticalSeams::to_vec`] or [`VerticalSeams::to_vec_lowest`] as they are in
/// the origninal coordinate space of the image.
/// In which case [`draw_vertical_seams_by_energy`] should be used instead.
pub fn draw_vertical_seams(image: &GrayImage, seams: &[VerticalSeam]) -> Image<Rgb<u8>> {
    let height = image.height();

    let mut out = map_pixels(image, |p| p.to_rgb());
    let mut offsets = vec![vec![]; height as usize];

    for seam in seams {
        assert_eq!(
            seam.0.len() as u32,
            height,
            "seam length does not match image height"
        );

        for (y, x) in (0..height).rev().zip(&seam.0) {
            let mut x_original = *x;

            for o in &offsets[y as usize] {
                if *o < *x {
                    x_original += 1;
                }
            }

            out.put_pixel(x_original, y, Rgb([255, 0, 0]));
            offsets[y as usize].push(x_original);
        }
    }

    out
}

/// Draws the `count` lowest-energy vertical seams on `image`, colouring them by
/// total seam energy.
///
/// Lower-energy seams are drawn in green, mid-range seams in yellow, and
/// higher-energy seams in red.
pub fn draw_vertical_seams_by_energy<P>(
    image: &Image<P>,
    vertical_seams: &VerticalSeams,
    count: usize,
) -> Image<Rgba<u8>>
where
    P: Pixel<Subpixel = u8>,
{
    let (width, height) = image.dimensions();
    let width_usize = width as usize;
    let height_usize = height as usize;

    assert_eq!(
        vertical_seams.seam_energies.len(),
        width_usize,
        "invalid vertical_seams struct, vertical_seams.seam_energies must contain the full length of the image"
    );
    assert_eq!(
        vertical_seams.deltas.len(),
        width_usize.saturating_mul(height_usize.saturating_sub(1)),
        "invalid vertical_seams, vertical_seams.deltas must contain all pixels"
    );

    let mut out = map_pixels(image, |p| p.to_rgba());
    let seam_ids = vertical_seams.lowest_energy_seams(count);

    if seam_ids.is_empty() {
        return out;
    }

    let (min_energy, max_energy) = seam_ids
        .iter()
        .map(|&start_x| vertical_seams.seam_energies[start_x])
        .fold((u64::MAX, u64::MIN), |(min_energy, max_energy), energy| {
            (min_energy.min(energy), max_energy.max(energy))
        });

    let range = max_energy - min_energy;

    let mut seams: Vec<(usize, Rgba<u8>)> = seam_ids
        .into_iter()
        .map(|start_x| {
            let color = if min_energy == max_energy {
                Rgba([0, 255, 0, 255])
            } else {
                let energy = vertical_seams.seam_energies[start_x] - min_energy;
                let position = (energy * 510) / range;

                if position <= 255 {
                    Rgba([position as u8, 255, 0, 255])
                } else {
                    Rgba([255, (510 - position) as u8, 0, 255])
                }
            };

            (start_x, color)
        })
        .collect();

    let mut deltas = vertical_seams.deltas.chunks_exact(width_usize);

    for y in 0..height_usize {
        for &(x, color) in &seams {
            assert!(x < width_usize, "seam x-position is outside image bounds");

            out.put_pixel(x as u32, y as u32, color);
        }

        if let Some(deltas_row) = deltas.next() {
            seams.iter_mut().for_each(|(x, _)| {
                *x = ((*x as isize) + (deltas_row[*x] as isize)) as usize;
            });

            for i in 1..seams.len() {
                if seams[i].0 < seams[i - 1].0 {
                    seams.swap(i, i - 1);
                }
            }
        }
    }

    out
}

/// The complete set of vertical seams computed for an image, stored compactly for use
/// with [`remove_vertical_seams`].
///
/// For individual [`VerticalSeam`]s see [`Self::to_vec`] (useful for debugging and
/// for passing to [`draw_vertical_seams`]).
#[derive(Debug, Clone)]
pub struct VerticalSeams {
    /// Row-major `(height-1) * width` array of per-step deltas (+1, -1, or 0)
    /// for each pixel in the image.
    // According to benchmarks this seems to be the cheapest/fastest way to store this.
    // It defers computation of the X-position until the seam is actually used.
    deltas: Vec<i8>,
    /// Total accumulated energy of each seam, indexed by the seam's starting x position.
    /// There is exactly one seam per column.
    seam_energies: Vec<u64>,
    /// Width of the image the vertical seams are from.
    width: u32,
    /// Height of the image the vertical seams are from.
    height: u32,
}

impl VerticalSeams {
    /// Get the VerticalSeam that starts at X
    pub fn get_seam(&self, start_x: usize) -> VerticalSeam {
        let (width, height) = (self.width as usize, self.height as usize);
        let mut x = start_x;
        let mut seam = Vec::with_capacity(height);
        seam.push(x as u32);

        for y in 0..height.saturating_sub(1) {
            let delta = self.deltas[(y * width) + x];

            x = ((x as isize) + (delta as isize)) as usize;
            seam.push(x as u32);
        }

        seam.reverse();

        VerticalSeam(seam)
    }

    /// The total accumulated energy of each seam, indexed by starting x-position.
    ///
    /// `seam_energies()[i]` is the total pixel energy of the seam that begins at column `i`
    /// in the top row. Useful for visualisation and debugging.
    pub fn seam_energies(&self) -> &[u64] {
        &self.seam_energies
    }

    /// A vector containing all [`VerticalSeam`].
    /// All VerticalSeam are relative to the original coordinate space of the image.
    pub fn to_vec(&self) -> Vec<VerticalSeam> {
        (0..self.width as usize).map(|x| self.get_seam(x)).collect()
    }

    /// A vector containing the `count` lowest energy [`VerticalSeam`] in order of their starting X position.
    /// All VerticalSeam are relative to the original coordinate space of the image.
    pub fn to_vec_lowest(&self, count: usize) -> Vec<VerticalSeam> {
        self.lowest_energy_seams(count)
            .into_iter()
            .map(|x| self.get_seam(x))
            .collect()
    }

    /// Get the starting X positions of the `count` lowest energy seams in order of their starting X position.
    pub fn lowest_energy_seams(&self, count: usize) -> Vec<usize> {
        let len = self.seam_energies.len();
        let mut ids: Vec<usize> = (0..len).collect();

        if count >= len {
            return ids;
        }

        let (lowest, _, _) = ids.select_nth_unstable_by(count, |a, b| {
            self.seam_energies[*a].cmp(&self.seam_energies[*b])
        });

        lowest.sort_unstable();
        lowest.to_vec()
    }
}

/// Reduces the width of an image using seam carving.
///
/// Uses the w2 algorithm from Huang et.al.
/// <https://web.archive.org/web/20120404113111/https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/seam-carving-ChinaF.pdf>.
///
/// This removes all required seams in a single pass.
///
/// # Panics
///
/// Panics if:
/// - `target_width > img.width()`
/// - `target_width == 0`
/// - `img.width() < 2` ([`find_vertical_seams`])
/// - `img.height() == 0` ([`find_vertical_seams`])
///
/// Unless `target_width == img.width()` in which case the original image is returned.
///
/// # Examples
/// ```
/// use std::path::PathBuf;
/// use imageproc::seam_carving::shrink_width;
///
/// let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
///     .join("tests/data/elephant.png");
/// let image = image::open(path).unwrap().into_rgb8();
///
/// let result = shrink_width(&image, 250);
/// assert_eq!(result.width(), 250);
/// assert_eq!(result.height(), image.height());
/// ```
#[must_use = "the function does not modify the original image"]
pub fn shrink_width<P>(img: &Image<P>, target_width: u32) -> Image<P>
where
    P: Pixel<Subpixel = u8>,
{
    let width = img.width();

    if width == target_width {
        return img.clone();
    }

    assert!(
        target_width <= width,
        "target_width must be <= input image width"
    );

    assert!(target_width > 0, "target_width must be above 0");

    let vertical_seams = find_vertical_seams(img);

    let remove_count = width.saturating_sub(target_width) as usize;
    remove_vertical_seams(img, &vertical_seams, remove_count)
}

/// Compute the intensity/luma of each pixel in the image.
/// map_pixels() too slow, this is ~7x faster.
/// ```text
/// test seam_carving::benches::elephant::bench_compute_intensity_map_pixels ... bench: 515,088.75 ns/iter (+/- 4,437.63)
/// test seam_carving::benches::elephant::bench_compute_intensity            ... bench:  74,287.87 ns/iter (+/- 2,848.04)
/// ```
#[must_use]
fn compute_intensity<P>(img: &Image<P>) -> Image<Luma<u8>>
where
    P: Pixel<Subpixel = u8>,
{
    let (width, height) = img.dimensions();

    let mut output = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);

    for (pixel, intensity) in img.pixels().zip(output.iter_mut()) {
        *intensity = pixel.to_luma()[0];
    }

    output
}

/// Computes the energy for each pixel
/// Using eq. 1, might be cutting some corners for speed but testing (and the paper) has so
/// far shown result similar to using sobel gradient, but with better performance.
#[must_use]
fn compute_energy(intensity: &Image<Luma<u8>>) -> Image<Luma<u16>> {
    let width = intensity.width();
    let height = intensity.height();
    let intensity = intensity.as_raw();
    let w = width as usize;
    let h = height as usize;

    let mut output = ImageBuffer::<Luma<u16>, Vec<u16>>::new(width, height);
    let energy = output.as_mut();

    if w == 0 || h == 0 {
        return output;
    }

    // Degenerate cases
    if w == 1 {
        energy.iter_mut().enumerate().for_each(|(y, e)| {
            let y_up = y.saturating_sub(1);
            let y_down = (y + 1).min(h - 1);
            *e = intensity[y_down].abs_diff(intensity[y_up]) as u16;
        });

        return output;
    }

    if h == 1 {
        energy.iter_mut().enumerate().for_each(|(x, e)| {
            let x_left = x.saturating_sub(1);
            let x_right = (x + 1).min(w - 1);
            *e = intensity[x_right].abs_diff(intensity[x_left]) as u16;
        });

        return output;
    }

    // Top row
    {
        let row0 = &intensity[0..w];
        let row1 = &intensity[w..2 * w];
        let out = &mut energy[0..w];

        out.iter_mut().enumerate().for_each(|(x, dst)| {
            let x_left = x.saturating_sub(1);
            let x_right = (x + 1).min(w - 1);

            let dx = row0[x_right].abs_diff(row0[x_left]) as u16;
            // Symmetric clamp: treat the row above as identical to row 0,
            // so dy = |row1[x] - row0[x]|.
            let dy = row1[x].abs_diff(row0[x]) as u16;
            *dst = dx + dy;
        });
    }

    // Bottom row
    {
        let y = h - 1;
        let row = &intensity[y * w..(y + 1) * w];
        let row_up = &intensity[(y - 1) * w..y * w];

        let out = &mut energy[y * w..(y + 1) * w];

        out.iter_mut().enumerate().for_each(|(x, dst)| {
            let x_left = x.saturating_sub(1);
            let x_right = (x + 1).min(w - 1);

            let dx = row[x_right].abs_diff(row[x_left]) as u16;
            // Symmetric clamp: treat the row below as identical to the bottom row,
            // so dy = |row[x] - row_up[x]|.
            let dy = row[x].abs_diff(row_up[x]) as u16;
            *dst = dx + dy;
        });
    }

    const INTERIOR_X_DIST: u16 = 2;

    energy[w..(h - 1) * w]
        .chunks_mut(w)
        .enumerate()
        .for_each(|(i, row_energy)| {
            let y = i + 1;

            let row = &intensity[y * w..(y + 1) * w];
            let row_up = &intensity[(y - 1) * w..y * w];
            let row_down = &intensity[(y + 1) * w..(y + 2) * w];

            // Edge pixels use clamped central differences.
            let left_dy = row_down[0].abs_diff(row_up[0]) as u16;
            let left_dx = row[1].abs_diff(row[0]) as u16;
            row_energy[0] = left_dx + left_dy;

            for x in 1..w - 1 {
                let dx = row[x + 1].abs_diff(row[x - 1]) as u16;
                let dy = row_down[x].abs_diff(row_up[x]) as u16;
                row_energy[x] = (dx + dy) / INTERIOR_X_DIST;
            }

            let dy = row_down[w - 1].abs_diff(row_up[w - 1]) as u16;
            let dx = row[w - 1].abs_diff(row[w - 2]) as u16;
            row_energy[w - 1] = dx + dy;
        });

    output
}

// Helper traits and functions to choose between u32/u64 and u64/u128 for
// bottom-up cumulative energy (M) and DP in compute_vertical_seams to
// allow for up to u32::MAX image sizes using u64 and u128 on larger images
// while keeping u32 and u64 for images < 65k x 65k to keep it faster.
//
// Quite overkill but keeps precision that we wouldn't get from f64 at sizes above ~5.1k x 5.1k.
// Although realistically no one is gonna run this on an image that is 65k x 65k, so it might
// make sense to just panic/error at those image sizes instead.
trait CumulativeCost:
    Primitive + Copy + Ord + Default + From<u16> + Into<u64> + core::ops::Add<Output = Self>
{
}
impl CumulativeCost for u32 {}
impl CumulativeCost for u64 {}

trait DpCost:
    Copy + Ord + Default + From<u64> + core::ops::Add<Output = Self> + core::ops::Mul<Output = Self>
{
}

impl DpCost for u64 {}
impl DpCost for u128 {}

/// Calculate max height where the bottom up calculation can fit within u32
fn m_fits_u32(height: usize) -> bool {
    height as u128 <= (u32::MAX as u128) / (u8::MAX as u128 * 2)
}

/// Calculate max dimensions where the dp calculation can fit within u64
fn dp_fits_u64(width: usize, height: usize) -> bool {
    let h0 = (height / 2) as u128;
    let h1 = (height - height / 2) as u128;
    let bound = ((u8::MAX as u32 * 2).pow(2)) as u128 * (width as u128) * h0 * h1;

    bound <= u64::MAX as u128
}

/// Computes the cumulative energy for each pixel based on the pixels below it
///
/// Eq. 9 from the paper. Fills `m[y][x] = energy[y][x] + min(m[y+1][x-1..=x+1])`, bottom-up.
/// `m[0][x]` is then the cost of the cheapest seam through column x in the top row.
///
/// u32 is enough for ~8.4 million pixels tall images.
/// To support images u32::MAX tall output needs to use u64,
/// that is 3x slower on a 350x188 image and 2x on a 3000x3000 image.
/// Uses generics to allow to easily swap between u32 and u64 in the same impl
/// to keep performance high for smaller images.
#[must_use]
fn compute_bottom_up_min_cumulative_energy<C>(energy: &Image<Luma<u16>>) -> Image<Luma<C>>
where
    C: CumulativeCost,
{
    let width = energy.width() as usize;
    let height = energy.height() as usize;
    debug_assert!(width > 1);
    let mut output = ImageBuffer::<Luma<C>, Vec<C>>::new(energy.width(), energy.height());
    let m = output.as_mut();
    let energy = energy.as_raw();

    {
        let y = height - 1;
        let row = y * width;

        for x in 0..width {
            let idx = row + x;
            m[idx] = <C as From<u16>>::from(energy[idx]);
        }
    }

    for y in (0..height - 1).rev() {
        let below = (y + 1) * width;

        {
            let x = 0;
            let idx = y * width + x;
            let best = m[below + x].min(m[below + x + 1]);
            m[idx] = <C as From<u16>>::from(energy[idx]) + best;
        }

        for x in 1..width - 1 {
            let idx = y * width + x;

            let best = m[below + x].min(m[below + x - 1]).min(m[below + x + 1]);

            m[idx] = <C as From<u16>>::from(energy[idx]) + best;
        }

        {
            let x = width - 1;
            let idx = y * width + x;
            let best = m[below + x].min(m[below + x - 1]);
            m[idx] = <C as From<u16>>::from(energy[idx]) + best;
        }
    }

    output
}

/// Computes the full set of vertical seams using the paper's row-to-row matching.
///
/// Eq. 5, 7, and 10 from the paper.
///
/// Assigns seams to columns in the next row by finding the
/// adjacent transpositions that maximises sum of products `a[x] * m_next[x]`.
///
/// u64 for dp is enough for images up to ~65kx65k before it overflows.
/// To support images >= u32::MAX in width/height dp needs to be u128,
/// that is 1.3x slower on a 350x188 image and 2x on a 3000x3000 image.
/// Uses traits to allow to easily swap between u64 and u128 in the same impl
/// to keep performance high for smaller images.
///
/// # Panics
/// Panics if:
/// - dimensions of `energy` and `m` don't match.
#[must_use]
fn compute_vertical_seams<M, D>(energy: &Image<Luma<u16>>, m: &Image<Luma<M>>) -> VerticalSeams
where
    M: CumulativeCost,
    D: DpCost,
{
    assert_eq!(
        energy.dimensions(),
        m.dimensions(),
        "energy and m must be of same width and height"
    );
    let width = energy.width() as usize;
    let height = energy.height() as usize;

    let energy = energy.as_raw();
    let m = m.as_raw();

    // A(i, j) represents the cumulative energy along the seam which passes through pixel I(i, j) from row 1 to j.
    // We only keep track of the active row as that's what relevant for our actual computations.
    let mut a: Vec<u64> = energy[..width].iter().map(|v| u64::from(*v)).collect();

    // origin[x] is the seam id (start_x) of the seam currently at position `x`.
    // origin and deltas are a representation of the matching function m(i, k) from the paper.
    let mut origin: Vec<_> = (0u32..width as u32).collect();
    let mut deltas = vec![0i8; (height - 1) * width];

    // should_stay[x] is true if the optimal transition for position x was stay
    // (seam x maps to column x in the next row), false if "swap" (seams x-1 and x cross).
    // For eq. 5.
    let mut should_stay = vec![false; width];

    for ((m_next, energy_next), delta_row) in m[width..]
        .chunks_exact(width)
        .zip(energy[width..].chunks_exact(width))
        .zip(deltas.chunks_exact_mut(width))
    {
        // dp_im1 and dp_im2 represent F(i-1) and F(i-2) from eq. 7
        // which are used to determine if we should swap or stay on x
        let mut dp_im2 = D::from(0u64);
        let mut dp_im1 = D::from(a[0]) * D::from(m_next[0].into());

        // Position x=0 can only stay as there is no -1 pixel.
        should_stay[0] = true;

        let mut previous_a = a[0];
        let mut previous_m_next = m_next[0].into();

        let row_data = a[1..]
            .iter()
            .copied()
            .zip(m_next[1..].iter().copied().map(Into::into))
            .zip(should_stay[1..].iter_mut());

        for ((current_a, current_m_next), stay) in row_data {
            let stay_dp = dp_im1 + D::from(current_a * current_m_next);
            let swap_dp = dp_im2
                + D::from(current_a * previous_m_next)
                + D::from(previous_a * current_m_next);

            dp_im2 = dp_im1;

            if swap_dp > stay_dp {
                *stay = false;
                dp_im1 = swap_dp;
            } else {
                *stay = true;
                dp_im1 = stay_dp;
            }

            previous_a = current_a;
            previous_m_next = current_m_next;
        }

        // Backtrack right-to-left: resolve each choice into a delta and propagate
        // the updated cumulative energies and origin ids to the next row.
        // Explained further in section 3.1 of the paper.

        // This zip-chain turns the iterator output quite ugly...
        // But it gives a good performance boost (~5%) on larger images
        let mut backtrack_iter = should_stay
            .iter()
            .zip(energy_next)
            .zip(a.iter_mut())
            .zip(origin.iter_mut())
            .zip(delta_row.iter_mut())
            .rev();

        while let Some(((((stay, energy), a), origin), delta)) = backtrack_iter.next() {
            if *stay {
                // `delta` and `origin` are already initizialized correctly for the stay condition.
                // Only `a` needs to be adjusted.
                *a += *energy as u64;
            } else {
                let (
                    (
                        (
                            (_next_stay, next_energy),
                            next_a
                        ),
                        next_origin
                    ),
                    next_delta
                ) = backtrack_iter.next()
                    .expect("there will always be a previous position because x=0 is forced to stay=true");

                *delta = -1;
                *next_delta = 1;

                let a_x = *a;
                let a_xm1 = *next_a;

                let origin_x = *origin;
                let origin_xm1 = *next_origin;

                *a = a_xm1 + (*energy as u64);
                *origin = origin_xm1;

                *next_a = a_x + (*next_energy as u64);
                *next_origin = origin_x;
            }
        }
    }

    // Convert from "energy by current position" to "energy by start_x".
    let mut seam_energies = vec![0u64; width];
    for x in 0..width {
        seam_energies[origin[x] as usize] = a[x];
    }

    VerticalSeams {
        deltas,
        seam_energies,
        width: width as u32,
        height: height as u32,
    }
}

/// Computes a set of vertical seams (one per top-row x) and their total energies
/// using the w2 algorithm from Huang et al (2009).
///
/// The returned [`VerticalSeams`] is a compact representation intended to be fed
/// into [`remove_vertical_seams`]; use [`VerticalSeams::to_vec`] to reconstruct
/// individual [`VerticalSeam`]s.
///
/// # Panics
///
/// Panics if `img.width() < 2` or `img.height() == 0`.
///
/// # Examples
/// ```
/// use std::path::PathBuf;
/// use imageproc::seam_carving::find_vertical_seams;
///
/// let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
///     .join("tests/data/elephant.png");
/// let image = image::open(path).unwrap().into_rgb8();
///
/// let seams = find_vertical_seams(&image);
///
/// // One seam per column in the top row.
/// assert_eq!(seams.to_vec().len(), image.width() as usize);
///
/// // The compact representation stores one total energy per seam.
/// assert_eq!(seams.seam_energies().len(), image.width() as usize);
/// assert_eq!(seams.to_vec_lowest(3).len(), 3);
/// ```
#[must_use]
pub fn find_vertical_seams<P>(img: &Image<P>) -> VerticalSeams
where
    P: Pixel<Subpixel = u8>,
{
    let width = img.width() as usize;
    let height = img.height() as usize;

    assert!(
        width > 1,
        "image must be at least 2 pixels wide to have any seams"
    );
    assert!(
        height > 0,
        "image must be at least 1 pixel tall to have any seams"
    );

    let intensity = compute_intensity(img);
    let energy = compute_energy(&intensity);

    match (m_fits_u32(height), dp_fits_u64(width, height)) {
        (true, true) => {
            let m = compute_bottom_up_min_cumulative_energy::<u32>(&energy);
            compute_vertical_seams::<u32, u64>(&energy, &m)
        }
        (false, true) => {
            let m = compute_bottom_up_min_cumulative_energy::<u64>(&energy);
            compute_vertical_seams::<u64, u64>(&energy, &m)
        }
        (true, false) => {
            let m = compute_bottom_up_min_cumulative_energy::<u32>(&energy);
            compute_vertical_seams::<u32, u128>(&energy, &m)
        }
        (false, false) => {
            let m = compute_bottom_up_min_cumulative_energy::<u64>(&energy);
            compute_vertical_seams::<u64, u128>(&energy, &m)
        }
    }
}

/// Removes `remove_count` of `vertical_seams` from `img`.
///
/// # Panics
///
/// Panics if `remove_count >= img.width()`, or if `vertical_seams` does not
/// correspond to the dimensions of `img`.
///
/// # Examples
/// ```
/// use std::path::PathBuf;
/// use imageproc::seam_carving::{find_vertical_seams, remove_vertical_seams};
///
/// let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
///     .join("tests/data/elephant.png");
/// let image = image::open(path).unwrap().into_rgb8();
///
/// let seams = find_vertical_seams(&image);
/// let result = remove_vertical_seams(&image, &seams, 25);
///
/// assert_eq!(result.width(), image.width() - 25);
/// assert_eq!(result.height(), image.height());
/// ```
#[must_use = "the function does not modify the original image"]
pub fn remove_vertical_seams<P>(
    img: &Image<P>,
    vertical_seams: &VerticalSeams,
    remove_count: usize,
) -> Image<P>
where
    P: Pixel,
{
    let (width, height) = img.dimensions();
    let width_usize = width as usize;
    let height_usize = height as usize;

    if remove_count == 0 {
        return img.clone();
    }

    assert!(
        remove_count < width_usize,
        "remove_count must be less than `img.width()`"
    );
    assert_eq!(
        vertical_seams.seam_energies.len(),
        width_usize,
        "invalid vertical_seams struct, vertical_seams.seam_energies must contain the full length of the image"
    );
    assert_eq!(
        vertical_seams.deltas.len(),
        (height_usize.saturating_sub(1)) * width_usize,
        "invalid vertical_seams, vertical_seams.deltas must contain all pixels"
    );

    let bytes_per_pixel = P::CHANNEL_COUNT as usize;
    assert!(
        bytes_per_pixel > 0,
        "the image has zero bytes per pixels, this shouldn't be possible...?"
    );

    let src = img.as_raw();
    let src_stride = width_usize * bytes_per_pixel;

    let out_width_usize = width_usize - remove_count;
    let out_stride = out_width_usize * bytes_per_pixel;

    debug_assert!(
        out_width_usize > 0,
        "resulting image must be at least 1 pixel wide."
    );

    let mut output = img.buffer_with_dimensions(out_width_usize as u32, height);
    let out_buf = output.as_mut();

    // Start with the selected seam ids (which are x positions in row 0).
    let mut seams = vertical_seams.lowest_energy_seams(remove_count);

    let mut deltas = vertical_seams.deltas.chunks_exact(width_usize);

    let rows = src
        .chunks_exact(src_stride)
        .zip(out_buf.chunks_exact_mut(out_stride));

    for (src_row, dst_row) in rows {
        // Copy row in spans, skipping seam pixels.
        let mut src_off = 0usize;
        let mut dst_off = 0usize;

        for &seam_x in &seams {
            debug_assert!(seam_x < width_usize);
            let seam_off = seam_x * bytes_per_pixel;

            let span = &src_row[src_off..seam_off];
            dst_row[dst_off..dst_off + span.len()].copy_from_slice(span);

            dst_off += span.len();
            src_off = seam_off + bytes_per_pixel;
        }

        // The rest of the row.
        let span = &src_row[src_off..src_stride];
        dst_row[dst_off..out_stride].copy_from_slice(span);
        debug_assert_eq!(dst_off + span.len(), out_stride);

        // Propagate seam positions to the next row.
        if let Some(deltas_row) = deltas.next() {
            seams.iter_mut().for_each(|x| {
                *x = ((*x as isize) + (deltas_row[*x] as isize)) as usize;
            });

            // The algorithm only uses disjoint swaps; one bubble-sort pass
            // fixes all inversions without creating new ones.
            for i in 1..seams.len() {
                if seams[i] < seams[i - 1] {
                    seams.swap(i, i - 1);
                }
            }

            debug_assert!(
                seams.windows(2).all(|w| w[0] < w[1]),
                "seams sorted after delta propagation"
            );
        }
    }

    output
}

#[cfg(not(miri))]
#[cfg(test)]
mod benches {
    use std::path::PathBuf;

    use crate::utils::{gray_bench_image, load_image_or_panic};

    pub fn load_bench_img(img_name: &str) -> image::RgbImage {
        let input_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join(img_name);

        load_image_or_panic(input_path).into_rgb8()
    }

    macro_rules! bench_suite {
        (
            $name:ident,
            shrink_by: [$($suffix:ident => $m:expr),+ $(,)?],
            image: $i:expr
        ) => {
            mod $name {
                use super::super::*;
                use super::*;
                use test::{Bencher, black_box};

                #[bench]
                fn bench_compute_intensity(b: &mut Bencher) {
                    let img = $i;

                    b.iter(|| {
                        let r = compute_intensity(&img);
                        black_box(r);
                    })
                }

                #[bench]
                fn bench_compute_energy(b: &mut Bencher) {
                    let img = $i;
                    let luma = compute_intensity(&img);

                    b.iter(|| {
                        let r = compute_energy(&luma);
                        black_box(r);
                    })
                }

                #[bench]
                fn bench_compute_bottom_up(b: &mut Bencher) {
                    let img = $i;
                    let luma = compute_intensity(&img);
                    let energies = compute_energy(&luma);

                    b.iter(|| {
                        let r = compute_bottom_up_min_cumulative_energy::<u32>(&energies);
                        black_box(r);
                    })
                }

                #[bench]
                fn bench_compute_vertical_seams(b: &mut Bencher) {
                    let img = $i;
                    let luma = compute_intensity(&img);
                    let energies = compute_energy(&luma);
                    let m = compute_bottom_up_min_cumulative_energy::<u32>(&energies);

                    b.iter(|| {
                        let r = compute_vertical_seams::<u32, u64>(&energies, &m);
                        black_box(r);
                    })
                }

                $(
                    mod $suffix {
                        use super::super::super::*;
                        use super::super::*;
                        use test::{Bencher, black_box};

                        #[bench]
                        fn bench_remove_seams(b: &mut Bencher) {
                            let img = $i;
                            let vertical_seams = find_vertical_seams(&img);

                            b.iter(|| {
                                let r = remove_vertical_seams(&img, &vertical_seams, $m as usize);
                                black_box(r);
                            })
                        }

                        #[bench]
                        fn bench_shrink_width(b: &mut Bencher) {
                            let img = $i;
                            let w = img.width();

                            b.iter(|| {
                                let filtered = super::shrink_width(&img, w - $m);
                                black_box(filtered);
                            })
                        }
                    }
                )+
            }
        };
    }

    bench_suite!(elephant, shrink_by: [shrink_width_r100 => 100], image: load_bench_img("elephant.png"));
    bench_suite!(zebra, shrink_by: [shrink_width_r100 => 100], image: load_bench_img("zebra.png"));
    bench_suite!(robin, shrink_by: [shrink_width_r100 => 100], image: load_bench_img("robin.png"));
    bench_suite!(grey_100x100, shrink_by: [
        shrink_width_r1 => 1,
        shrink_width_r4 => 4,
        shrink_width_r8 => 8,
        shrink_width_r25 => 25
    ], image: gray_bench_image(100, 100));
    bench_suite!(grey_500x500, shrink_by: [shrink_width_r50 => 50], image: gray_bench_image(500, 500));
    bench_suite!(grey_1000x1000, shrink_by: [shrink_width_r100 => 100], image: gray_bench_image(1000, 1000));
    //bench_suite!(grey_3000x3000, shrink_by: [shrink_width_r300 => 300], image: gray_bench_image(3000, 3000));
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, RgbImage};

    // -------------------------------------------------------------------------
    // compute_energy
    // -------------------------------------------------------------------------

    #[test]
    fn energy_uniform_image_is_zero() {
        // A completely uniform image has no gradient anywhere.
        let width = 4u32;
        let height = 4u32;
        let buf = vec![100u8; (width * height) as usize];
        let intensity = GrayImage::from_vec(width, height, buf).expect("valid image");
        let energy = compute_energy(&intensity);
        assert!(
            &energy.as_raw().iter().all(|&e| e == 0),
            "uniform image must have zero energy everywhere"
        );
    }

    #[test]
    fn energy_single_column() {
        // w=1 degenerate path: only dy, no dx.
        // pixels top-to-bottom: 0, 100, 200
        // dy at y=0: |100 - 0|   = 100  (clamped: y_up = 0, y_down = 1)
        // dy at y=1: |200 - 0|   = 200  (y_up=0, y_down=2)
        // dy at y=2: |200 - 100| = 100  (y_up=1, y_down=2 clamped)
        let intensity = gray_image!(0; 100; 200);
        let energy = compute_energy(&intensity);
        assert_eq!(energy.as_raw(), &vec![100u16, 200u16, 100u16]);
    }

    #[test]
    fn energy_single_row() {
        // h=1 degenerate path: only dx, no dy.
        // pixels left-to-right: 0, 100, 200
        // dx at x=0: |100 -   0| = 100  (clamped left)
        // dx at x=1: |200 -   0| = 200  (central)
        // dx at x=2: |200 - 100| = 100  (clamped right)
        let intensity = gray_image!(0, 100, 200);
        let energy = compute_energy(&intensity);
        assert_eq!(energy.as_raw(), &vec![100u16, 200u16, 100u16]);
    }

    #[test]
    fn energy_horizontal_gradient_3x3() {
        // Grid (row-major, top-to-bottom):
        //   0  64 128
        //   0   0   0
        //   0  64 128
        //
        // Row 0 (top):   dy = |row1[x] - row0[x]|, dx = clamped central on row0
        //   x=0: dx=|64-0|=64,   dy=|0-0|=0     => 64
        //   x=1: dx=|128-0|=128, dy=|0-64|=64   => 192
        //   x=2: dx=|128-64|=64, dy=|0-128|=128 => 192
        //
        // Row 1 (middle): row1 = [0,0,0], full central diff
        //   x=0: dx=|row1[1]-row1[0]|=0, dy=|row2[0]-row0[0]|=0 => 0
        //   x=1: dx=|row1[2]-row1[0]|=0, dy=|row2[1]-row0[1]|=0 => 0
        //   x=2: dx=|row1[2]-row1[1]|=0, dy=|row2[2]-row0[2]|=0 => 0
        //
        // Row 2 (bottom): dy = |row2[x] - row1[x]|, dx = clamped central on row2
        //   x=0: dx=|64-0|=64,   dy=|0-0|=0    => 64
        //   x=1: dx=|128-0|=128, dy=|64-0|=64  => 192
        //   x=2: dx=|128-64|=64, dy=|128-0|=128 => 192
        let intensity = gray_image!(
            0, 64, 128;
            0, 0, 0;
            0, 64, 128
        );

        let energy = compute_energy(&intensity);

        #[rustfmt::skip]
        let expected: Vec<u16> = vec![
             64, 192, 192,
              0,   0,   0,
             64, 192, 192,
        ];
        assert_eq!(energy.as_raw(), &expected);
    }

    #[test]
    fn energy_vertical_gradient_3x3() {
        // Grid (row-major):
        //   0   0   0
        //  64  64  64
        // 128 128 128
        //
        // Row 0 (top):    dy=|64-0|=64,   dx=0 everywhere => 64
        // Row 1 (middle): dy=|128-0|=128, dx=0             => 128
        // Row 2 (bottom): dy=|128-64|=64, dx=0             => 64
        let intensity = gray_image!(
            0, 0, 0;
            64, 64, 64;
            128, 128, 128
        );

        let energy = compute_energy(&intensity);

        #[rustfmt::skip]
        let expected: Vec<u16> = vec![
             64,  64,  64,
            128,  64, 128,
             64,  64,  64,
        ];
        assert_eq!(energy.as_raw(), &expected);
    }

    // -------------------------------------------------------------------------
    // compute_bottom_up_min_cumulative_energy
    // -------------------------------------------------------------------------

    #[test]
    fn cumulative_energy_single_row() {
        // With height=1 there is no DP to do; output equals the energy row.
        let energy: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(3, 1, vec![10, 20, 30]).expect("valid image");
        let m = compute_bottom_up_min_cumulative_energy::<u32>(&energy);
        assert_eq!(m.as_raw(), &vec![10u32, 20, 30]);
    }

    #[test]
    fn cumulative_energy_two_rows() {
        // energy (row-major, top first):
        //   10  20  30   (row 0)
        //    5  15  25   (row 1, bottom - copied as-is)
        //
        // row 0 (filled last in DP):
        //   x=0: 10 + min(5,15)       = 15
        //   x=1: 20 + min(5,15,25)    = 25
        //   x=2: 30 + min(15,25)      = 45
        let energy: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(3, 2, vec![10, 20, 30, 5, 15, 25]).expect("valid image");
        let m = compute_bottom_up_min_cumulative_energy::<u32>(&energy);
        assert_eq!(m.as_raw(), &vec![15u32, 25, 45, 5, 15, 25]);
    }

    #[test]
    fn cumulative_energy_steers_toward_minimum() {
        // energy:
        //  100 100 100   (row 0)
        //  100   1 100   (row 1 - only the centre is cheap)
        //
        // Bottom row: [100, 1, 100]
        // Row 0:
        //   x=0: 100 + min(100,1)     = 101
        //   x=1: 100 + min(100,1,100) = 101
        //   x=2: 100 + min(1,100)     = 101
        let energy: ImageBuffer<Luma<u16>, Vec<u16>> =
            ImageBuffer::from_raw(3, 2, vec![100, 100, 100, 100, 1, 100]).expect("valid image");
        let m = compute_bottom_up_min_cumulative_energy::<u32>(&energy);
        assert_eq!(m.as_raw(), &vec![101u32, 101, 101, 100, 1, 100]);
    }

    // -------------------------------------------------------------------------
    // find_vertical_seams / VerticalSeams::to_vec
    // -------------------------------------------------------------------------

    #[test]
    fn find_seams_produces_correct_count() {
        // There must be exactly `width` seams, one starting at each x in row 0.
        let width = 5u32;
        let height = 4u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);

        assert_eq!(seams.to_vec().len(), width as usize);
    }

    #[test]
    fn find_seams_each_seam_has_correct_length() {
        let width = 6u32;
        let height = 8u32;
        let img = GrayImage::from_vec(width, height, vec![0u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);

        for seam in seams.to_vec() {
            assert_eq!(
                seam.0.len(),
                height as usize,
                "each seam must have one entry per row"
            );
        }
    }

    #[test]
    fn find_seams_adjacency_constraint() {
        // Eq. 2 from the paper: consecutive entries in a seam must differ by at most 1.
        // seam.0 is stored bottom-to-top, so we check all consecutive pairs.
        let width = 8u32;
        let height = 6u32;
        let img = GrayImage::from_vec(width, height, vec![0u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);

        for seam in seams.to_vec() {
            for w in seam.0.windows(2) {
                assert!(
                    w[0].abs_diff(w[1]) <= 1,
                    "seam adjacency violated: {} and {} differ by more than 1",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn lowest_energy_seam_has_minimum_energy() {
        // Column 2 is a constant 128; all other columns alternate 0/255 rows,
        // producing high vertical gradient energy. Column 2 has zero dy everywhere
        // (same value in every row), so the seam that runs through it should have
        // strictly lower total energy than any seam through the high-gradient columns.
        let img = gray_image!(
              0, 255, 128,   0, 255;
            255,   0, 128, 255,   0;
              0, 255, 128,   0, 255;
            255,   0, 128, 255,   0
        );
        let seams = find_vertical_seams(&img);

        // The minimum energy across all seams should be strictly less than
        // the average, since the high-gradient columns drive the average up.
        let energies = seams.seam_energies();
        let min_energy = *energies.iter().min().unwrap();
        let max_energy = *energies.iter().max().unwrap();
        assert!(
            min_energy < max_energy,
            "expected seam energies to differ, got all equal: {min_energy}"
        );

        assert_eq!(
            min_energy, energies[2],
            "third column should have lowest energy"
        );

        // The single lowest-energy seam should return exactly one result.
        let lowest = seams.to_vec_lowest(1);
        assert_eq!(lowest.len(), 1);
    }

    #[test]
    fn lowest_energy_seams_count_boundary() {
        let width = 4u32;
        let height = 3u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);

        assert_eq!(seams.lowest_energy_seams(0).len(), 0);
        assert_eq!(seams.lowest_energy_seams(1).len(), 1);
        assert_eq!(seams.lowest_energy_seams(4).len(), 4); // == width
        assert_eq!(seams.lowest_energy_seams(10).len(), 4); // > width, clamped
    }

    // -------------------------------------------------------------------------
    // remove_vertical_seams
    // -------------------------------------------------------------------------

    #[test]
    fn remove_zero_seams_returns_clone() {
        let img = gray_image!(
            1, 2, 3, 4;
            5, 6, 7, 8;
            9, 10, 11, 12
        );
        let seams = find_vertical_seams(&img);
        let result = remove_vertical_seams(&img, &seams, 0);

        assert_eq!(result.dimensions(), img.dimensions());
        assert_eq!(result.as_raw(), img.as_raw());
    }

    #[test]
    fn remove_one_seam_reduces_width_by_one() {
        let width = 5u32;
        let height = 4u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);
        let result = remove_vertical_seams(&img, &seams, 1);

        assert_eq!(result.width(), width - 1);
        assert_eq!(result.height(), height);
    }

    #[test]
    fn remove_all_but_one_column() {
        let width = 5u32;
        let height = 3u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);
        let result = remove_vertical_seams(&img, &seams, (width - 1) as usize);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), height);
    }

    #[test]
    fn remove_seam_pixel_correctness() {
        // 3×2 image with distinct pixel values so we can check exactly which
        // pixels survive.
        //
        //  row 0 (top):    10  20  30
        //  row 1 (bottom): 40  50  60
        //
        // energy (computed via compute_intensity then compute_energy):
        //   intensities are the same as pixel values for Luma<u8>.
        //   x=0: dx=|20-10|=10, dy=|40-10|=30 => 40
        //   x=1: dx=|30-10|=20, dy=|50-20|=30 => 50
        //   x=2: dx=|30-20|=10, dy=|60-30|=30 => 40
        //
        // cumulative (bottom row = energy, row 0 adds min from below):
        //   bottom: [40, 50, 60]
        //   top:
        //     x=0: 40 + min(40,50)    = 80
        //     x=1: 50 + min(40,50,60) = 90
        //     x=2: 40 + min(50,60)    = 90
        //
        // The lowest-energy seam starts at x=0 (cumulative=80).
        // After removing it the remaining image is:
        //   row 0: 20 30
        //   row 1: 50 60
        #[rustfmt::skip]
        let img = gray_image!(
            10, 20, 30;
            40, 50, 60
        );
        let seams = find_vertical_seams(&img);
        let result = remove_vertical_seams(&img, &seams, 1);

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 2);

        let raw = result.as_raw();
        assert_eq!(raw[0], 20, "top-left should be 20");
        assert_eq!(raw[1], 30, "top-right should be 30");
        assert_eq!(raw[2], 50, "bottom-left should be 50");
        assert_eq!(raw[3], 60, "bottom-right should be 60");
    }

    // -------------------------------------------------------------------------
    // shrink_width
    // -------------------------------------------------------------------------

    #[test]
    fn shrink_width_noop() {
        // target_width == current width: remove_count is 0, returns clone.
        let width = 4u32;
        let height = 3u32;
        let pixels: Vec<u8> = (0u8..(width * height) as u8).collect();
        let img = GrayImage::from_vec(width, height, pixels).expect("valid image");
        let result = shrink_width(&img, width);

        assert_eq!(result.dimensions(), img.dimensions());
        assert_eq!(result.as_raw(), img.as_raw());
    }

    #[test]
    fn shrink_width_correct_dimensions() {
        let width = 10u32;
        let height = 6u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let target_width = 7u32;
        let result = shrink_width(&img, target_width);

        assert_eq!(result.width(), target_width);
        assert_eq!(result.height(), height);
    }

    // -------------------------------------------------------------------------
    // VerticalSeams internal consistency
    // -------------------------------------------------------------------------

    #[test]
    fn to_vec_and_to_vec_lowest_full_agree() {
        // On a non-uniform image, seams diverge; to_vec_lowest(width) must still
        // return the same seams as to_vec(), ordered by starting x position.
        #[rustfmt::skip]
        let img = gray_image!(
             10, 200,  50, 180,  20, 160;
            190,  30, 170,  40, 150,  60;
            100, 120,  80, 140,  90, 110;
             70, 200,  30, 180,  50, 210
        );
        let seams = find_vertical_seams(&img);

        let all = seams.to_vec();
        let lowest_all = seams.to_vec_lowest(6);

        assert_eq!(all.len(), lowest_all.len());
        for (a, b) in all.iter().zip(lowest_all.iter()) {
            assert_eq!(
                a.0, b.0,
                "seam paths from to_vec and to_vec_lowest(width) must match on varied image"
            );
        }
    }

    #[test]
    fn seam_energies_equal_for_uniform_image() {
        // For a uniform image every seam has the same total energy.
        let width = 4u32;
        let height = 3u32;
        let img = GrayImage::from_vec(width, height, vec![100u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);
        let energies = seams.seam_energies();
        let first = energies[0];

        assert!(
            energies.iter().all(|&e| e == first),
            "all seams in a uniform image must have equal energy, got: {:?}",
            energies
        );
    }

    #[test]
    fn rgb_image_shrinks_correctly() {
        // Verify the pipeline works end-to-end with an RGB image (3 bytes/pixel).
        let width = 6u32;
        let height = 4u32;
        let pixels: Vec<u8> = (0u8..=254)
            .cycle()
            .take((width * height * 3) as usize)
            .collect();

        let img = RgbImage::from_vec(width, height, pixels).unwrap();
        let target_width = 4u32;
        let result = shrink_width(&img, target_width);

        assert_eq!(result.width(), target_width);
        assert_eq!(result.height(), height);
    }

    // -------------------------------------------------------------------------
    // draw_vertical_seams
    // -------------------------------------------------------------------------
    #[test]
    fn draw_vertical_seams_removed_individually_matches_original_coordinates() {
        #[rustfmt::skip]
        let img = gray_image!(
            10, 20, 30, 40;
            50, 60, 70, 80;
            90, 100, 110, 120
        );

        // First seam removes the leftmost column in every row.
        // Second seam then removes x=1 in the once-shrunken image, which corresponds
        // to original x=2 in every row.
        let individually_removed = vec![VerticalSeam(vec![0, 0, 0]), VerticalSeam(vec![1, 1, 1])];

        let drawn_individually = draw_vertical_seams(&img, &individually_removed);

        let red = Rgb([255, 0, 0]);

        for y in 0..3 {
            assert_eq!(*drawn_individually.get_pixel(0, y), red);
            assert_eq!(*drawn_individually.get_pixel(2, y), red);

            // Unchanged columns should still be grayscale converted to RGBA.
            let expected_col_1 = img.get_pixel(1, y)[0];
            let expected_col_3 = img.get_pixel(3, y)[0];

            assert_eq!(
                *drawn_individually.get_pixel(1, y),
                Rgb([expected_col_1, expected_col_1, expected_col_1])
            );
            assert_eq!(
                *drawn_individually.get_pixel(3, y),
                Rgb([expected_col_3, expected_col_3, expected_col_3])
            );
        }
    }

    // -------------------------------------------------------------------------
    // Edge case tests for minimum dimension images
    // -------------------------------------------------------------------------

    #[test]
    fn energy_width_two_height_one() {
        // w=2 is the minimum valid width; only dx, no dy (h=1)
        // pixels: 128, 255
        // dx at x=0: |255-128|=127 (clamped left)
        // dx at x=1: |255-128|=127 (central, clamped right)
        let intensity = gray_image!(128, 255);
        let energy = compute_energy(&intensity);
        assert_eq!(energy.as_raw(), &vec![127u16, 127]);
    }

    #[test]
    fn shrink_two_width_to_one() {
        // 2x3 image shrunk from width 2 to width 1 (minimum possible output)
        let img = gray_image!(1, 2; 3, 4; 5, 6);
        let result = shrink_width(&img, 1);

        assert_eq!(result.width(), 1u32);
        assert_eq!(result.height(), 3u32);
    }

    #[test]
    fn find_seams_on_two_pixel_width() {
        // Verify seams are found correctly on minimum width image
        let img = GrayImage::from_vec(2, 3, vec![1u8, 2, 3, 4, 5, 6]).expect("valid image");
        let seams = find_vertical_seams(&img);
        assert_eq!(seams.to_vec().len(), 2u32 as usize);
        for seam in seams.to_vec() {
            assert_eq!(seam.0.len(), 3u32 as usize);
            for &x in &seam.0 {
                assert!(x < 2, "seam x={x} out of bounds for 2-width image");
            }
        }
    }

    #[test]
    fn energy_two_by_two() {
        // With a 2×2 image every pixel is simultaneously a corner: it is in the
        // top/bottom row AND the leftmost/rightmost column.  Both dx and dy use
        // clamped one-sided differences for every pixel.
        //
        // Pixel layout:
        //    (0,0)=10  (1,0)=50
        //    (0,1)=30  (1,1)=90
        //
        // Top row (y=0), using dy = |row1[x] - row0[x]|:
        //   x=0: dx=|50-10|=40, dy=|30-10|=20 => 60
        //   x=1: dx=|50-10|=40, dy=|90-50|=40 => 80
        //
        // Bottom row (y=1), using dy = |row1[x] - row0[x]| (row1 == row below clamped):
        //   x=0: dx=|90-30|=60, dy=|30-10|=20 => 80
        //   x=1: dx=|90-30|=60, dy=|90-50|=40 => 100
        let intensity = gray_image!(10, 50; 30, 90);
        let energy = compute_energy(&intensity);
        assert_eq!(energy.as_raw(), &vec![60u16, 80, 80, 100]);
    }

    // -------------------------------------------------------------------------
    // Pixel-correctness: removed seam is an interior column
    // -------------------------------------------------------------------------

    #[test]
    fn remove_seam_interior_column_pixel_correctness() {
        // Design a 5×2 image where the cheapest seam passes through column 2,
        // not the leftmost column.
        //
        // Row 0: 200,  10,  10,  10, 200
        // Row 1: 200,  10,  10,  10, 200
        //
        // All pixels are u8 gray values.  Intensities equal pixel values.
        //
        // energy (using top/bottom-row formulas, h=2):
        //   Top row (y=0): dy = |row1[x] - row0[x]|
        //     x=0: dx=|10-200|=190, dy=|200-200|=0  => 190
        //     x=1: dx=|10-200|=190, dy=|10-10|=0    => 190
        //     x=2: dx=|10-10|=0,   dy=|10-10|=0     => 0
        //     x=3: dx=|200-10|=190, dy=|10-10|=0    => 190
        //     x=4: dx=|200-10|=190, dy=|200-200|=0  => 190
        //   Bottom row (y=1): symmetric by construction => same values
        //
        // Cumulative (bottom row initialised to its own energy; top row adds min
        // from the three neighbours below):
        //   bottom: [190, 190, 0, 190, 190]
        //   top:
        //     x=0: 190 + min(190,190)   = 380
        //     x=1: 190 + min(190,0,190) = 190   <- cheapest at top
        //     x=2:   0 + min(190,0,190) = 0     <- cheapest at top
        //     x=3: 190 + min(0,190,190) = 190
        //     x=4: 190 + min(190,190)   = 380
        //
        // Lowest-energy seam: starts at x=2 (cumulative 0) and stays at x=2
        // in row 1 (delta 0, since energy at x=2 bottom is the minimum neighbour
        // of x=2 top).
        //
        // After removing that seam:
        //   row 0: 200  10  10 200
        //   row 1: 200  10  10 200
        #[rustfmt::skip]
        let img = gray_image!(
            200, 10, 10, 10, 200;
            200, 10, 10, 10, 200
        );
        let seams = find_vertical_seams(&img);
        let result = remove_vertical_seams(&img, &seams, 1);

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 2);

        let raw = result.as_raw();
        assert_eq!(raw[0], 200, "row0 col0");
        assert_eq!(raw[1], 10, "row0 col1");
        assert_eq!(raw[2], 10, "row0 col2");
        assert_eq!(raw[3], 200, "row0 col3");
        assert_eq!(raw[4], 200, "row1 col0");
        assert_eq!(raw[5], 10, "row1 col1");
        assert_eq!(raw[6], 10, "row1 col2");
        assert_eq!(raw[7], 200, "row1 col3");
    }

    // -------------------------------------------------------------------------
    // seam_energy_matches_sum_along_path
    // -------------------------------------------------------------------------

    #[test]
    fn seam_energy_matches_sum_along_reconstructed_path() {
        // Verify that seam_energies()[start_x] equals the sum of compute_energy
        // values along the seam reconstructed by to_vec().
        //
        // We use a small varied image so the seams are non-trivial.
        #[rustfmt::skip]
        let img = gray_image!(
              0, 100,  50, 200;
            150,  25, 175,  75;
             80, 220,  10, 130
        );

        let intensity = compute_intensity(&img);
        let energy_img = compute_energy(&intensity);
        let energy_raw = energy_img.as_raw();
        let width = img.width() as usize;

        let seams = find_vertical_seams(&img);
        let seam_energies = seams.seam_energies();

        for (start_x, seam) in seams.to_vec().iter().enumerate() {
            // seam.0 is bottom-to-top; index 0 = bottom row, last = top row.
            let path_sum: u64 = seam
                .0
                .iter()
                .rev()
                .enumerate()
                .map(|(y, &x)| energy_raw[y * width + x as usize] as u64)
                .sum();

            assert_eq!(
                seam_energies[start_x], path_sum,
                "seam energy mismatch for start_x={start_x}: stored={} computed={}",
                seam_energies[start_x], path_sum
            );
        }
    }

    // -------------------------------------------------------------------------
    // #[should_panic] contract tests
    // -------------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "image must be at least 2 pixels")]
    fn find_vertical_seams_panics_on_width_one() {
        let img = gray_image!(10; 20; 30);
        let _ = find_vertical_seams(&img);
    }

    #[test]
    #[should_panic(expected = "image must be at least 1 pixel tall")]
    fn find_vertical_seams_panics_on_zero_height() {
        let img = GrayImage::new(4, 0);
        let _ = find_vertical_seams(&img);
    }

    #[test]
    #[should_panic(expected = "target_width must be <= input image width")]
    fn shrink_width_panics_when_target_exceeds_width() {
        let img = GrayImage::from_vec(4, 3, vec![0u8; 12]).expect("valid image");
        let _ = shrink_width(&img, 5);
    }

    #[test]
    #[should_panic(expected = "target_width must be above 0")]
    fn shrink_width_panics_on_zero_target() {
        let img = GrayImage::from_vec(4, 3, vec![0u8; 12]).expect("valid image");
        let _ = shrink_width(&img, 0);
    }

    #[test]
    #[should_panic(expected = "remove_count must be less than")]
    fn remove_vertical_seams_panics_when_remove_count_equals_width() {
        let width = 4u32;
        let height = 3u32;
        let img = GrayImage::from_vec(width, height, vec![128u8; (width * height) as usize])
            .expect("valid image");
        let seams = find_vertical_seams(&img);
        // remove_count == width is forbidden (result would be 0-width image)
        let _ = remove_vertical_seams(&img, &seams, width as usize);
    }

    #[test]
    #[should_panic(expected = "seam length does not match image height")]
    fn draw_vertical_seams_panics_on_wrong_seam_length() {
        let img = GrayImage::from_vec(4, 3, vec![128u8; 12]).expect("valid image");
        // Build a seam with the wrong length (2 instead of 3)
        let bad_seam = VerticalSeam(vec![0, 1]);
        draw_vertical_seams(&img, &[bad_seam]);
    }

    // -------------------------------------------------------------------------
    // remove_vertical_seam
    // -------------------------------------------------------------------------
    #[test]
    #[allow(deprecated)]
    fn test_remove_vertical_seam_left_edge() {
        #[rustfmt::skip]
        let image = gray_image!(
            00, 01, 02, 03;
            10, 11, 12, 13;
            20, 21, 22, 23);

        #[rustfmt::skip]
        let expected = gray_image!(
            01, 02, 03;
            11, 12, 13;
            21, 22, 23);

        let seam = VerticalSeam(vec![0, 0, 0]);

        assert_pixels_eq!(remove_vertical_seam(&image, &seam), expected);
    }

    #[test]
    #[allow(deprecated)]
    fn test_remove_vertical_seam_right_edge() {
        #[rustfmt::skip]
        let image = gray_image!(
            00, 01, 02, 03;
            10, 11, 12, 13;
            20, 21, 22, 23);

        #[rustfmt::skip]
        let expected = gray_image!(
            00, 01, 02;
            10, 11, 12;
            20, 21, 22);

        let seam = VerticalSeam(vec![3, 3, 3]);

        assert_pixels_eq!(remove_vertical_seam(&image, &seam), expected);
    }

    #[test]
    #[allow(deprecated)]
    fn test_remove_vertical_seam_zig_zag() {
        #[rustfmt::skip]
        let image = gray_image!(
            00, 01, 02, 03;
            10, 11, 12, 13;
            20, 21, 22, 23);

        #[rustfmt::skip]
        let expected = gray_image!(
            01, 02, 03;
            10, 12, 13;
            20, 21, 23);

        let seam = VerticalSeam(vec![2, 1, 0]);

        assert_pixels_eq!(remove_vertical_seam(&image, &seam), expected);
    }

    #[test]
    #[allow(deprecated)]
    fn test_remove_vertical_seam_single_column() {
        #[rustfmt::skip]
        let image = gray_image!(
            00;
            10;
            20);

        let seam = VerticalSeam(vec![0, 0, 0]);
        let carved = remove_vertical_seam(&image, &seam);

        assert_eq!(carved.dimensions(), (0, 3));
        assert!(carved.is_empty());
    }

    #[test]
    #[allow(deprecated)]
    #[should_panic(expected = "seam length does not match image height")]
    fn test_remove_vertical_seam_rejects_wrong_seam_length() {
        #[rustfmt::skip]
        let image = gray_image!(
            00, 01;
            10, 11;
            20, 21);

        let seam = VerticalSeam(vec![0, 0]);

        remove_vertical_seam(&image, &seam);
    }
}
