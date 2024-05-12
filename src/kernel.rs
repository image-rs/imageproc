//! Traits and Structs for the abstraction of kernels used in gradient methods

use itertools::Itertools;

use crate::point::Point;

/// A 2D kernel
///
/// Used in methods such as [`gradients()`](crate::gradients::gradients)
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

    /// Returns the sobel horizontal 3x3 kernel.
    pub fn sobel_horizontal_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-1, 0, 1, -2, 0, 2, -1, 0, 1].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }
    /// Returns the sobel vertical 3x3 kernel.
    pub fn sobel_vertical_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-1, -2, -1, 0, 0, 0, 1, 2, 1].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }

    /// Returns the scharr horizontal 3x3 kernel.
    pub fn scharr_horizontal_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-3, 0, 3, -10, 0, 10, -3, 0, 3].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }
    /// Returns the scharr vertical 3x3 kernel.
    pub fn scharr_vertical_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-3, -10, -3, 0, 0, 0, 3, 10, 3].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }

    /// Returns the prewitt horizontal 3x3 kernel.
    pub fn prewitt_horizontal_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-1, 0, 1, -1, 0, 1, -1, 0, 1].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }
    /// Returns the prewitt vertical 3x3 kernel.
    pub fn prewitt_vertical_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [-1, -1, -1, 0, 0, 0, 1, 1, 1].map(K::from).to_vec(),
            width: 3,
            height: 3,
        }
    }

    /// Returns the roberts horizontal 3x3 kernel.
    pub fn roberts_horizontal_2x2() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [1, 0, 0, -1].map(K::from).to_vec(),
            width: 2,
            height: 2,
        }
    }
    /// Returns the roberts vertical 3x3 kernel.
    pub fn roberts_vertical_2x2() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [0, 1, -1, -0].map(K::from).to_vec(),
            width: 2,
            height: 2,
        }
    }

    /// Returns the laplacian 3x3 kernel.
    pub fn laplacian_3x3() -> Self
    where
        K: From<i8> + Clone,
    {
        Self {
            data: [0, 1, 0, 1, -4, 1, 0, 1, 0].map(K::from).to_vec(),
            width: 3,
            height: 3,
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

#[derive(Debug)]
/// An borrowed 2D kernel, used to filter images via convolution.
pub struct BorrowedKernel<'a, K> {
    data: &'a [K],
    width: u32,
    height: u32,
}
impl<'b, K> Kernel<K> for BorrowedKernel<'b, K> {
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
