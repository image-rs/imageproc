//! Extension trait for `image::Rect`

use std::cmp;

pub use image::math::Rect;

/// Extension methods for [`Rect`] used by `imageproc`.
pub trait RectExt {
    /// Smallest y-coordinate reached by rect.
    fn top_y(&self) -> u32;
    /// Smallest x-coordinate reached by rect.
    fn bottom_y(&self) -> u32;
    /// Greatest y-coordinate reached by rect.
    fn left_x(&self) -> u32;
    /// Greatest x-coordinate reached by rect.
    fn right_x(&self) -> u32;
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
    fn intersect(&self, other: &Self) -> Option<Self>
    where
        Self: std::marker::Sized;
    /// Does `self` contain the given coordinates?
    fn contains(&self, x: u32, y: u32) -> bool;
}

impl RectExt for Rect {
    fn top_y(&self) -> u32 {
        self.y
    }
    fn left_x(&self) -> u32 {
        self.x
    }
    fn bottom_y(&self) -> u32 {
        self.y + self.height - 1
    }
    fn right_x(&self) -> u32 {
        self.x + self.width - 1
    }

    fn intersect(&self, other: &Rect) -> Option<Rect> {
        let left_x = cmp::max(self.x, other.x);
        let top_y = cmp::max(self.y, other.y);
        let right_x = cmp::min(self.right_x(), other.right_x());
        let bottom_y = cmp::min(self.bottom_y(), other.bottom_y());

        if right_x < left_x || bottom_y < top_y {
            return None;
        }

        Some(Rect {
            x: left_x,
            y: top_y,
            width: right_x - left_x + 1,
            height: bottom_y - top_y + 1,
        })
    }

    fn contains(&self, x: u32, y: u32) -> bool {
        self.x <= x && x <= self.right_x() && self.y <= y && y <= self.bottom_y()
    }
}

#[cfg(test)]
mod tests {
    use crate::rect::RectExt;

    use super::Rect;

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
