//! Basic manipulation of rectangles.

use std::cmp;

/// A rectangular region of non-zero width and height.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

impl Rect {
    /// Reduces possibility of confusing coordinates and dimensions
    /// when specifying rects.
    pub fn at(x: i32, y: i32) -> RectPosition {
        RectPosition { left: x, top: y }
    }

    /// Smallest y-coordinate reached by rect.
    pub fn top(&self) -> i32 {
        self.top
    }

    /// Smallest x-coordinate reached by rect.
    pub fn left(&self) -> i32 {
        self.left
    }

    /// Greatest y-coordinate reached by rect.
    pub fn bottom(&self) -> i32 {
        self.top + (self.height as i32) - 1
    }

    /// Greatest x-coordinate reached by rect.
    pub fn right(&self) -> i32 {
        self.left + (self.width as i32) - 1
    }

    /// Width of rect.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of rect.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the intersection of self and other, or none if they are are disjoint.
    pub fn intersect(&self, other: Rect) -> Option<Rect> {
        let left = cmp::max(self.left, other.left);
        let top = cmp::max(self.top, other.top);
        let right = cmp::min(self.right(), other.right());
        let bottom = cmp::min(self.bottom(), other.bottom());

        if right < left || bottom < top {
            return None;
        }

        Some(Rect {
            left: left,
            top: top,
            width: (right - left) as u32 + 1,
            height: (bottom - top) as u32 + 1,
        })
    }
}

/// Position of the top left of a rectangle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RectPosition {
    left: i32,
    top: i32,
}

impl RectPosition {
    /// Construct a rectangle form a position and size. Width and height
    /// are required to be strictly positive.
    pub fn of_size(self, width: u32, height: u32) -> Rect {
        assert!(width > 0, "width must be strictly positive");
        assert!(height > 0, "height must be strictly positive");
        Rect {
            left: self.left,
            top: self.top,
            width: width,
            height: height,
        }
    }
}

#[cfg(test)]
mod test {
    use super::Rect;

    #[test]
    #[should_panic]
    fn test_rejects_empty_rectangle() {
        Rect::at(1, 2).of_size(0, 1);
    }

    #[test]
    fn test_intersect_disjoint() {
        let r = Rect::at(0, 0).of_size(5, 5);
        let s = Rect::at(10, 10).of_size(100, 12);
        assert_eq!(r.intersect(s), None);
    }

    #[test]
    fn test_intersect_overlapping() {
        let r = Rect::at(0, 0).of_size(5, 5);
        let s = Rect::at(1, 4).of_size(10, 12);
        let i = Rect::at(1, 4).of_size(4, 1);
        assert_eq!(r.intersect(s), Some(i));
    }
}
