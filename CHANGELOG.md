# Change Log

## [0.24.0] - 2024-03-16

New features:
* Added BRIEF descriptors
* Added draw_antialiased_polygon
* Added draw_hollow_polygon, draw_hollow_polygon_mut
* Added contour_area
* Added match_template_parallel
* Made Contour clonable
* Re-export image crate and add image/default as default feature

Performance improvements:
* Faster interpolate_nearest
* Faster find_contours_with_threshold
* Faster approximate_polygon_do
* Faster rotating_calipers

Bug fixes:
* Stop window::display_image consuming 100% CPU on Linux

Breaking changes:
* Migrate text rendering from rusttype to ab_glyph
* Updated depenedencies
* Increased MSRV to 1.70

## [0.23.0] - 2022-04-10

...

## [0.6.1] - 2016-12-28
- Fixed bug in draw_line_segment_mut when line extends outside of image bounds.
- Generalised connected_components to handle arbitrary equatable pixel types.
- Added support for drawing hollow and filled circles.
- Added support for drawing anti-aliased lines, and convex polygons.
- Added adaptive_threshold function.

## [0.6.0] - 2016-05-07
No change log kept for this or earlier versions.
