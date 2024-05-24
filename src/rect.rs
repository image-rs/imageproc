//! Basic manipulation of rectangles.

use std::cmp;

/// A rectangular region.
///
/// # Examples
/// ```
/// use imageproc::rect::Rect;
/// use imageproc::rect::Region;
///
/// // Construct a rectangle with top-left corner at (4, 5), width 6 and height 7.
/// let rect = Rect{x:4, y:5,width:6, height:7};
///
/// // Contains top-left point:
/// assert_eq!(rect.left_x(), 4);
/// assert_eq!(rect.top_y(), 5);
/// assert!(rect.contains(rect.left_x(), rect.top_y()));
///
/// // Contains bottom-right point, at (left + width - 1, top + height - 1):
/// assert_eq!(rect.right_x(), 9);
/// assert_eq!(rect.bottom_y(), 11);
/// assert!(rect.contains(rect.right_x(), rect.bottom_y()));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    /// The x-coordinate of the top-left corner of the rectangle.
    pub x: u32,
    /// The y-coordinate of the top-left corner of the rectangle.
    pub y: u32,
    /// The non-zero width of the rectangle
    pub width: u32,
    /// The non-zero height of the rectangle
    pub height: u32,
}

/// A geometrical representation of a set of 2D points with coordinate type T.
pub trait Region<T> {
    /// Whether this region contains the given point.
    fn contains(&self, x: T, y: T) -> bool;
}

impl Rect {
    /// Smallest y-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](struct.Rect.html) for examples.
    pub fn top_y(&self) -> u32 {
        self.y
    }

    /// Smallest x-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](struct.Rect.html) for examples.
    pub fn left_x(&self) -> u32 {
        self.x
    }

    /// Greatest y-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](struct.Rect.html) for examples.
    pub fn bottom_y(&self) -> u32 {
        self.y + self.height - 1
    }

    /// Greatest x-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](struct.Rect.html) for examples.
    pub fn right_x(&self) -> u32 {
        self.x + self.width - 1
    }

    /// Returns the intersection of self and other, or none if they are are disjoint.
    ///
    /// # Examples
    /// ```
    /// use imageproc::rect::Rect;
    /// use imageproc::rect::Region;
    ///
    /// // Intersecting a rectangle with itself
    /// let r = Rect{x: 4, y: 5, width: 6, height: 7};
    /// assert_eq!(r.intersect(r), Some(r));
    ///
    /// // Intersecting overlapping but non-equal rectangles
    /// let r = Rect{x: 0, y: 0, width:5, height:5};
    /// let s = Rect{x: 1, y: 4, width:10, height: 2};
    /// let i = Rect{x: 1, y: 4, width:4, height:1};
    /// assert_eq!(r.intersect(s), Some(i));
    ///
    /// // Intersecting disjoint rectangles
    /// let r = Rect{x: 0, y: 0, width:5, height:5};
    /// let s = Rect{x: 10, y: 10, width: 100, height: 12};
    /// assert_eq!(r.intersect(s), None);
    /// ```
    pub fn intersect(&self, other: Rect) -> Option<Rect> {
        let left = cmp::max(self.x, other.x);
        let top = cmp::max(self.y, other.y);
        let right = cmp::min(self.right_x(), other.right_x());
        let bottom = cmp::min(self.bottom_y(), other.bottom_y());

        if right < left || bottom < top {
            return None;
        }

        Some(Rect {
            x: left,
            y: top,
            width: (right - left) as u32 + 1,
            height: (bottom - top) as u32 + 1,
        })
    }
}

impl Region<u32> for Rect {
    fn contains(&self, x: u32, y: u32) -> bool {
        self.x <= x && x <= self.right_x() && self.y <= y && y <= self.bottom_y()
    }
}

#[cfg(test)]
mod tests {
    use super::{Rect, Region};

    #[test]
    fn test_contains_i32() {
        let r = Rect {
            x: 5,
            y: 5,
            width: 6,
            height: 6,
        };
        assert!(r.contains(5, 5));
        assert!(r.contains(10, 10));
        assert!(!r.contains(10, 11));
        assert!(!r.contains(11, 10));
    }
}
