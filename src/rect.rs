//! Basic manipulation of rectangles.

/// A rectangular region of non-zero width and height.
/// # Examples
/// ```
/// use imageproc::rect::Rect;
/// use imageproc::rect::Region;
///
/// // Construct a rectangle with top-left corner at (4, 5), width 6 and height 7.
/// let rect = Rect::at(4, 5).of_size(6, 7);
///
/// // Contains top-left point:
/// assert_eq!(rect.left(), 4);
/// assert_eq!(rect.top(), 5);
/// assert!(rect.contains(rect.left(), rect.top()));
///
/// // Contains bottom-right point, at (left + width - 1, top + height - 1):
/// assert_eq!(rect.right(), 9);
/// assert_eq!(rect.bottom(), 11);
/// assert!(rect.contains(rect.right(), rect.bottom()));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

/// A geometrical representation of a set of 2D points with coordinate type T.
pub trait Region<T> {
    /// Whether this region contains the given point.
    fn contains(&self, x: T, y: T) -> bool;
}

impl Rect {
    /// Reduces possibility of confusing coordinates and dimensions
    /// when specifying rects.
    ///
    /// See the [struct-level documentation](Rect) for examples.
    pub fn at(x: i32, y: i32) -> RectPosition {
        RectPosition { left: x, top: y }
    }

    /// Smallest y-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](Rect) for examples.
    pub fn top(&self) -> i32 {
        self.top
    }

    /// Smallest x-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](Rect) for examples.
    pub fn left(&self) -> i32 {
        self.left
    }

    /// Greatest y-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](Rect) for examples.
    pub fn bottom(&self) -> i32 {
        self.top + i32::try_from(self.height).unwrap() - 1
    }

    /// Greatest x-coordinate reached by rect.
    ///
    /// See the [struct-level documentation](Rect) for examples.
    pub fn right(&self) -> i32 {
        self.left + i32::try_from(self.width).unwrap() - 1
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
    ///
    /// # Examples
    /// ```
    /// use imageproc::rect::Rect;
    /// use imageproc::rect::Region;
    ///
    /// // Intersecting a rectangle with itself
    /// let r = Rect::at(4, 5).of_size(6, 7);
    /// assert_eq!(r.intersect(r), Some(r));
    ///
    /// // Intersecting overlapping but non-equal rectangles
    /// let r = Rect::at(0, 0).of_size(5, 5);
    /// let s = Rect::at(1, 4).of_size(10, 12);
    /// let i = Rect::at(1, 4).of_size(4, 1);
    /// assert_eq!(r.intersect(s), Some(i));
    ///
    /// // Intersecting disjoint rectangles
    /// let r = Rect::at(0, 0).of_size(5, 5);
    /// let s = Rect::at(10, 10).of_size(100, 12);
    /// assert_eq!(r.intersect(s), None);
    /// ```
    pub fn intersect(&self, other: Rect) -> Option<Rect> {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());

        if right < left || bottom < top {
            return None;
        }

        let width = u32::try_from(right - left).unwrap() + 1;
        let height = u32::try_from(bottom - top).unwrap() + 1;
        Some(Rect::at(left, top).of_size(width, height))
    }
}

impl Region<i32> for Rect {
    fn contains(&self, x: i32, y: i32) -> bool {
        self.left <= x && x <= self.right() && self.top <= y && y <= self.bottom()
    }
}

impl Region<f32> for Rect {
    fn contains(&self, x: f32, y: f32) -> bool {
        self.left as f32 <= x
            && x <= self.right() as f32
            && self.top as f32 <= y
            && y <= self.bottom() as f32
    }
}

/// Position of the top left of a rectangle.
/// Only used when building a [`Rect`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RectPosition {
    left: i32,
    top: i32,
}

impl RectPosition {
    /// Construct a rectangle from a position and size. Width and height
    /// are required to be strictly positive.
    ///
    /// See the [`Rect`] documentation for examples.
    pub fn of_size(self, width: u32, height: u32) -> Rect {
        assert!(width > 0, "width must be strictly positive");
        assert!(height > 0, "height must be strictly positive");
        Rect {
            left: self.left,
            top: self.top,
            width,
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Rect, Region};

    #[test]
    #[should_panic]
    fn test_rejects_empty_rectangle() {
        Rect::at(1, 2).of_size(0, 1);
    }

    #[test]
    fn test_contains_i32() {
        let r = Rect::at(5, 5).of_size(6, 6);
        assert!(r.contains(5, 5));
        assert!(r.contains(10, 10));
        assert!(!r.contains(10, 11));
        assert!(!r.contains(11, 10));
    }

    #[test]
    fn test_contains_f32() {
        let r = Rect::at(5, 5).of_size(6, 6);
        assert!(r.contains(5f32, 5f32));
        assert!(!r.contains(10.1f32, 10f32));
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_intersect(
            (x1, y1) in (-50..50, -50..50),
            (w1, h1) in (1..50u32, 1..50u32),

            (x2, y2) in (-50..50, -50..50),
            (w2, h2) in (1..50u32, 1..50u32),
        ) {
            let rect1 = Rect::at(x1, y1).of_size(w1, h1);
            let rect2 = Rect::at(x2, y2).of_size(w2, h2);

            if let Some(intersect) = rect1.intersect(rect2) {
                assert!(intersect.width() > 0);
                assert!(intersect.height() > 0);
            };
        }
    }
}
