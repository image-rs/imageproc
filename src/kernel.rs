//! Traits and Structs for the abstraction of kernels used in gradient methods

use itertools::Itertools;

use crate::point::Point;

/// A kernel object, in column major layout
///
/// Used in methods such as [`gradents()`]
pub trait Kernel<K> {
    /// The width of the kernel
    fn width(&self) -> u32;
    /// The height of the kernel
    fn height(&self) -> u32;

    /// Access an element of the kernel
    fn get(&self, x: u32, y: u32) -> &K;
    /// Enumerate all elements of the kernel
    fn enumerate<'a>(&'a self) -> impl Iterator<Item = (Point<u32>, &'a K)>
    where
        K: 'a;
}
impl<T, K> Kernel<K> for &T
where
    T: Kernel<K>,
{
    fn width(&self) -> u32 {
        (*self).width()
    }
    fn height(&self) -> u32 {
        (*self).height()
    }
    fn get(&self, x: u32, y: u32) -> &K {
        (*self).get(x, y)
    }
    fn enumerate<'a>(&'a self) -> impl Iterator<Item = (Point<u32>, &'a K)>
    where
        K: 'a,
    {
        (*self).enumerate()
    }
}

/// An owned 2D kernel, used to filter images via convolution.
#[derive(Debug, Clone)]
pub struct OwnedKernel<K> {
    data: Vec<K>,
    width: u32,
    height: u32,
}
impl<K> OwnedKernel<K> {
    /// Construct a kernel from a slice and its dimensions. The input slice is
    /// in row-major form.
    pub fn new(data: Vec<K>, width: u32, height: u32) -> OwnedKernel<K> {
        assert!(width > 0 && height > 0, "width and height must be non-zero");
        assert!(
            width * height == data.len() as u32,
            "Invalid kernel len: expecting {}, found {}",
            width * height,
            data.len()
        );
        OwnedKernel {
            data,
            width,
            height,
        }
    }
    /// Maps the kernel from type `K` to `Q` via the closure `f`.
    pub fn map<Q>(self, f: &impl Fn(K) -> Q) -> OwnedKernel<Q> {
        OwnedKernel {
            data: self.data.into_iter().map(f).collect(),
            width: self.width,
            height: self.height,
        }
    }
}

impl<K> Kernel<K> for OwnedKernel<K> {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn get(&self, x: u32, y: u32) -> &K {
        &self.data[(y * self.width + x) as usize]
    }
    fn enumerate<'a>(&'a self) -> impl Iterator<Item = (Point<u32>, &'a K)>
    where
        K: 'a,
    {
        let points = (0..self.height)
            .cartesian_product(0..self.width)
            .map(|(y, x)| Point { x, y });

        points.zip(self.data.iter())
    }
}

/// A type for combining two kernels into one struct
pub struct TwoKernels<T> {
    /// The first of the two kernels
    pub kernel1: T,
    /// The second of the two kernels
    pub kernel2: T,
}
