//! Basic manipulation of rectangles.

/// A rectangular region.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    left: i32,
    top: i32,
    width: u32,
    height: u32
}

impl Rect {
    /// Reduces possibility of confusing coordinates and dimensions
    /// when specifying rects.
    pub fn at(x: i32, y: i32) -> RectPosition {
        RectPosition { left: x, top: y}
    }

    /// Smallest y-coordinate reached by rect.
    pub fn top(&self) -> i32 {
        self.top
    }

    /// Smallest x-coordinate reached by rect.
    pub fn left(&self) -> i32 {
        self.left
    }

    /// Width of rect.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of rect.
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Position of the top left of a rectangle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RectPosition {
    left: i32,
    top: i32
}

impl RectPosition {

    pub fn of_size(self, width: u32, height: u32) -> Rect {
        Rect { left: self.left, top: self.top, width: width, height: height }
    }
}
