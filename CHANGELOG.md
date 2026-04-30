# Change Log

## [0.23.1]

Bug fixes:
* Fixed out-of-bounds read in interpolation functions (`interpolate_bilinear`, `interpolate_bicubic`, `interpolate_nearest`) when NaN coordinates bypass bounds checks.
* Fixed u32 overflow in `Kernel::new` dimension check that could allow constructing a kernel with mismatched dimensions.
* Fixed compilation on recent rustc versions due to a type inference regression in tests.

## [0.6.1] - 2016-12-28
- Fixed bug in draw_line_segment_mut when line extends outside of image bounds.
- Generalised connected_components to handle arbitrary equatable pixel types.
- Added support for drawing hollow and filled circles.
- Added support for drawing anti-aliased lines, and convex polygons.
- Added adaptive_threshold function.

## [0.6.0] - 2016-05-07
No change log kept for this or earlier versions.
