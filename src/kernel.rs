//! The kernel type for filter operations

/// An borrowed 2D kernel, used to filter images via convolution.
#[derive(Debug, Copy, Clone)]
pub struct Kernel<'a, K> {
    pub(crate) data: &'a [K],
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl<'a, K> Kernel<'a, K> {
    /// Construct a kernel from a slice and its dimensions. The input slice is
    /// in row-major order.
    ///
    /// # Panics
    /// 1. If `width == 0 || height == 0`.
    /// 2. If `width * height != data.len() as u32`.
    pub const fn new(data: &'a [K], width: u32, height: u32) -> Kernel<'a, K> {
        assert!(width > 0 && height > 0, "width and height must be non-zero");
        assert!(width * height == data.len() as u32);
        Kernel {
            data,
            width,
            height,
        }
    }

    /// Get the value in the kernel at the given `x` and `y` position.
    ///
    /// # Panics
    ///
    /// If the `x` or `y` is outside of the width or height of the kernel.
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> &K {
        &self.data[(y * self.width + x) as usize]
    }
}

/// The sobel horizontal 3x3 kernel.
pub const SOBEL_HORIZONTAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-1, 0, 1, -2, 0, 2, -1, 0, 1], 3, 3);
/// The sobel vertical 3x3 kernel.
pub const SOBEL_VERTICAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-1, -2, -1, 0, 0, 0, 1, 2, 1], 3, 3);

/// The scharr horizontal 3x3 kernel.
pub const SCHARR_HORIZONTAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-3, 0, 3, -10, 0, 10, -3, 0, 3], 3, 3);
/// The scharr vertical 3x3 kernel.
pub const SCHARR_VERTICAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-3, -10, -3, 0, 0, 0, 3, 10, 3], 3, 3);

/// The prewitt horizontal 3x3 kernel.
pub const PREWITT_HORIZONTAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-1, 0, 1, -1, 0, 1, -1, 0, 1], 3, 3);
/// The prewitt vertical 3x3 kernel.
pub const PREWITT_VERTICAL_3X3: Kernel<'static, i32> =
    Kernel::new(&[-1, -1, -1, 0, 0, 0, 1, 1, 1], 3, 3);

/// The roberts horizontal 3x3 kernel.
pub const ROBERTS_HORIZONTAL_2X2: Kernel<'static, i32> = Kernel::new(&[1, 0, 0, -1], 2, 2);
/// The roberts vertical 3x3 kernel.
pub const ROBERTS_VERTICAL_2X2: Kernel<'static, i32> = Kernel::new(&[0, 1, -1, -0], 2, 2);

/// The 4-connectivity laplacian 3x3 kernel.
pub const FOUR_LAPLACIAN_3X3: Kernel<'static, i16> =
    Kernel::new(&[0, 1, 0, 1, -4, 1, 0, 1, 0], 3, 3);
/// The 8-connectivity laplacian 3x3 kernel.
pub const EIGHT_LAPLACIAN_3X3: Kernel<'static, i16> =
    Kernel::new(&[1, 1, 1, 1, -8, 1, 1, 1, 1], 3, 3);
