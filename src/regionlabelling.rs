//! Functions for finding and labelling connected components of an image.

use image::{
    GenericImage,
    ImageBuffer,
    Luma
};

use definitions::{
    VecBuffer
};

use unionfind::DisjointSetForest;

use std::cmp;
use std::collections::HashMap;

/// Whether we consider the NW, NE, SW, and SE neighbors of
/// a pixel to be connected to it, or just its N, S, E, and W
/// neighbors.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Connectivity { Four, Eight }

/// Returns an image of the same size as the input, where each pixel
/// is labelled by the connected foreground component it belongs to,
/// or 0 if it's in the background. Input pixels are treated as belonging
/// to the background if and only if they have value 0.
pub fn connected_components<I>(image: &I, conn: Connectivity)
        -> VecBuffer<Luma<u32>>
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    // TODO: add macro to abandon early if either dimension is zero
    if width == 0 || height == 0 {
        return out;
    }

    let background = Luma([0u8]);
    let mut forest = DisjointSetForest::new((width * height) as usize);
    let mut adj_labels = [0u32; 4];
    let mut next_label = 1;

    for y in 0..height {
        for x in 0..width {
            out.put_pixel(x, y, Luma([0]));
        }

        for x in 0..width {

            let current = image.get_pixel(x, y);
            if current == background {
                continue;
            }

            let mut num_adj = 0;

            if x > 0 {
                // West
                let pixel = image.get_pixel(x - 1, y);
                if pixel == current {
                    let label = out.get_pixel(x - 1, y)[0];
                    adj_labels[num_adj] = label;
                    num_adj += 1;
                }
            }

            if y > 0 {
                // North
                let pixel = image.get_pixel(x, y - 1);
                if pixel == current {
                    let label = out.get_pixel(x, y - 1)[0];
                    adj_labels[num_adj] = label;
                    num_adj += 1;
                }

                if conn == Connectivity::Eight {
                    if x > 0 {
                        // North West
                        let pixel = image.get_pixel(x - 1, y - 1);
                        if pixel == current {
                            let label = out.get_pixel(x - 1, y - 1)[0];
                            adj_labels[num_adj] = label;
                            num_adj += 1;
                        }
                    }
                    if x < width - 1 {
                        // North East
                        let pixel = image.get_pixel(x + 1, y - 1);
                        if pixel == current {
                            let label = out.get_pixel(x + 1, y - 1)[0];
                            adj_labels[num_adj] = label;
                            num_adj += 1;
                        }
                    }
                }
            }

            if num_adj == 0 {
                out.put_pixel(x, y, Luma([next_label]));
                next_label += 1;
            }
            else {
                let mut min_label = u32::max_value();
                for n in 0..num_adj {
                    min_label = cmp::min(min_label, adj_labels[n]);
                }
                out.put_pixel(x, y, Luma([min_label]));
                for n in 0..num_adj {
                    forest.union(min_label as usize, adj_labels[n] as usize);
                }
            }
        }
    }

    // Make components start at 1
    let mut root_name = 1;
    let mut root_idx: HashMap<usize, usize> = HashMap::new();

    for y in 0..height {
        for x in 0..width {
            if image.get_pixel(x, y) == background {
                continue;
            }

            let label = out.get_pixel(x, y)[0];
            let root = forest.root(label as usize);

            match root_idx.get(&root).map(|x| *x) {
                Some(idx) => {
                    out.put_pixel(x, y, Luma([idx as u32]));
                }
                None => {
                    let idx = root_name;
                    root_idx.insert(root, idx);
                    out.put_pixel(x, y, Luma([idx as u32]));
                    root_name += 1;
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod test {

    use super::{
        connected_components
    };
    use super::Connectivity::{
        Four,
        Eight
    };
    use image::{
        GrayImage,
        ImageBuffer,
        Luma
    };

    #[test]
    fn test_connected_components_four() {
        let image: GrayImage = ImageBuffer::from_raw(4, 4, vec![
            1, 0, 2, 1,
            0, 1, 1, 0,
            0, 0, 0, 0,
            0, 0, 0, 1]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(4, 4, vec![
                1, 0, 2, 3,
                0, 4, 4, 0,
                0, 0, 0, 0,
                0, 0, 0, 5]).unwrap();

        let labelled = connected_components(&image, Four);
        assert_pixels_eq!(labelled, expected);
    }

    #[test]
    fn test_connected_components_eight() {
        let image: GrayImage = ImageBuffer::from_raw(4, 4, vec![
            1, 0, 2, 1,
            0, 1, 1, 0,
            0, 0, 0, 0,
            0, 0, 0, 1]).unwrap();

        let expected: ImageBuffer<Luma<u32>, Vec<u32>>
            = ImageBuffer::from_raw(4, 4, vec![
                1, 0, 2, 1,
                0, 1, 1, 0,
                0, 0, 0, 0,
                0, 0, 0, 3]).unwrap();

        let labelled = connected_components(&image, Eight);
        assert_pixels_eq!(labelled, expected);
    }

    // TODO: add benchmark
}
