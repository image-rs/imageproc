//! Utilities for working with the connectivity of pixels and their neighbors

use image::Pixel;

use crate::{
    definitions::{Image,Position},
    point::Point,
};


/// Determines which neighbors of a pixel we consider
/// to be connected to it.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Connectivity {
    /// A pixel is connected to its N, S, E and W neighbors.
    Four,
    /// A pixel is connected to all of its neighbors.
    Eight,
    /// A pixel is connected to its NW, NE, SE, and SW neighbors.
    Diagonal,
    /// To be diagonally adjacent, pixels are either (Four-adjacent to each
    /// other) OR (they are diagonal to each other AND have no Four-adjacent neighbors in common).
    M,
}


/// Returns an [Iterator]<Item=[Point]<[u32]>> containing the indices of the nearest four/eight neighbors (based on the [Connectivity] value passed to the function) of the given point, performing boundary
/// checks. Note that (u32,u32) implements [From]<[Point]<[u32]>> so conversion is trivial.
/// 
/// ## Example
///
/// ```
/// /*
/// This simple image is used for the test below, where P is the point whose neighbors we want to
/// obtain. The 4s indicate the indices which will be yielded by [Connectivity::Four]; if
/// [Connectivity::Eight] is used instead, the indices of pixels labeled 8 will be included as
/// well. Note that this test covers an instance where the indices on the left side are out of
/// bounds -- they will not be included.
/// */
///
///
/// # extern crate imageproc;
/// use imageproc::{
///     gray_image,
///     definitions::Position,
///     point::Point,
///     connectivity::{Connectivity, neighbor_indices},
/// };
/// # fn main() {
///
/// let img = gray_image!(
///     0, 0, 0, 0, 0,
///     1, 1, 0, 0, 0,
///     1, 1, 0, 0, 0,
///     1, 1, 0, 0, 0,
///     0, 0, 0, 0, 0
/// );
/// let pos = (0,2); // X and Y indices of the point whose neighbors we're interested in
///
/// // Nearest 4 neighbors
/// let mut neighbors: Vec<(u32,u32)> = neighbor_indices(&img, &pos, &Connectivity::Four)
///     .map(|p| <(u32,u32)>::from(p))
///     .collect();
/// assert_eq!(neighbors.sort(), [(0,1), (1,2), (0,3)].to_vec().sort());
///
/// // Nearest 8 neighbors
/// neighbors = neighbor_indices(&img, &pos, &Connectivity::Eight)
///     .map(|p| <(u32,u32)>::from(p))
///     .collect();
/// assert_eq!(neighbors.sort(), [(0,1), (1,1), (1,2), (0,3), (1,3)].to_vec().sort());
///
///  let img2 = gray_image!(
///     0, 0, 0, 0, 0, 1,
///     1, 1, 0, 9, 1, 0,
///     2, 1, 1, 1, 0, 3, 
///     1, 1, 0, 1, 1, 5,
///     0, 0, 0, 0, 0, 1
///  ); 
///
///  neighbors = neighbor_indices(&img2, &(3,2), &Connectivity::M)
///     .map(|p| <(u32,u32)>::from(p))
///     .collect();
///  assert_eq!(neighbors.sort(), [(2,2), (4,1), (4,3)].to_vec().sort());
///
/// # }
/// ```
///
///
pub fn neighbor_indices<T, P>(image: &Image<P>, point: &T, connectivity: &Connectivity) -> impl Iterator<Item=Point<u32>>
where
    T: Position,
    P: Pixel+PartialEq,
{
    let (x,y) = (point.x(), point.y());
    let (width,height) = image.dimensions();
    if y >= height || x >= width { // no need to bounds check for negative inputs as the Position trait guarantees u32 indices
        Vec::new().into_iter() // no allocation needed
    } else { let mut out = match connectivity {
            Connectivity::Four|Connectivity::Diagonal => Vec::with_capacity(4),
            Connectivity::Eight|Connectivity::M => Vec::with_capacity(8),
        };
        match connectivity {
            Connectivity::Diagonal => {},
            _ => {
                if y >= 1 {
                    out.push(Point::new(x, y-1)); // North
                }
                if x >= 1 {
                    out.push(Point::new(x-1, y)); // West
                }
                if x+1 < width {
                    out.push(Point::new(x+1, y)); // East
                }
                if y+1 < height {
                    out.push(Point::new(x, y+1)); // South
                }
            }
        }
        match connectivity {
            Connectivity::Four => {},
            Connectivity::Eight | Connectivity::Diagonal => {
                if x >= 1 && y >= 1 {
                    out.push(Point::new(x-1, y-1)); // Northwest
                }
                if x+1 < width && y >= 1 {
                    out.push(Point::new(x+1, y-1)); // Northeast
                }
                if x+1 < width && y+1 < height {
                    out.push(Point::new(x+1, y+1)); // Southeast
                }
                if x >= 1 && y+1 < height {
                    out.push(Point::new(x-1, y+1)); // Southwest
                }
            },
            Connectivity::M => {
                let mut my_n4s = neighbors(image, point, &Connectivity::Four);
                if neighbors(image, point, &Connectivity::Diagonal)
                .any(move |n| {
                        my_n4s.any(|my_n| my_n == n)
                    }
                ) {
                    if x >= 1 && y >= 1 {
                        out.push(Point::new(x-1, y-1)); // Northwest
                    }
                    if x+1 < width && y >= 1 {
                        out.push(Point::new(x+1, y-1)); // Northeast
                    }
                    if x+1 < width && y+1 < height {
                        out.push(Point::new(x+1, y+1)); // Southeast
                    }
                    if x >= 1 && y+1 < height {
                        out.push(Point::new(x-1, y+1)); // Southwest
                    }
                }
            },
        }
        out.into_iter()
    }
}

/// Given a reference to an image, the index of a certain pixel, and a [Connectivity] value,
/// returns an iterator containing references to the neighboring pixels
/// 
/// ## Example
///
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// use imageproc::{
///     gray_image,
///     definitions::Position,
///     point::Point,
///     connectivity::*,
/// };
///
/// # fn main() {
///
/// use image::Luma;
///
///
/// let image = gray_image!(
///     4, 0, 1, 2, 1;
///     0, 3, 255, 4, 0;
///     0, 5, 6, 7, 1;
///     1, 0, 0, 1, 4);
///
/// let point: Point<u32> = Point::new(2, 1);
/// let n4: Vec<&Luma<u8>> = neighbors(&image, &point, &Connectivity::Four)
///     .collect();
/// assert_eq!(n4, [&image[(2,0)], &image[(1,1)], &image[(3,1)], &image[(2,2)]].to_vec());
/// 
///
/// # }
///
/// ```
pub fn neighbors<'a, T,P>(image: &'a Image<P>, point: &T, connectivity: &Connectivity) -> impl Iterator<Item= &'a P>
where
P: Pixel+PartialEq,
T: Position,
{
    neighbor_indices(image, point, connectivity)
        .map(move |n| {
            let tuple = <(u32,u32)>::from(n);
            &image[tuple]
        }
    )
}

/*
/// Returns an iterator containing mutable references to the neighboring pixels according to the
/// connectivity parameter
pub fn neighbors_mut<'a, T,P, S>(image: &'a mut Image<P>, point: &T, connectivity: &Connectivity) -> impl Iterator<Item= &'a mut P>
where
P: Pixel<Subpixel=S>,
T: Position,
{
    let (width,height) = image.dimensions();
    neighbor_indices(point, height, width, connectivity)
        .map(move |n| {
        }
            //&mut image[(n.x, n.y)];

            //image.get_mut((n.x, n.y)).unwrap(); // we did bounds checking in neighbor_indices. get_unchecked_mut would work here but is unsafe
            //P::from_slice_mut(&mut image[(n.x, n.y)].get_mut([..]))
        }
    )
}

*/
