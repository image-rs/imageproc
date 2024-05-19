# Change Log

## [0.25.0] - 2024-05-19

New features:
* Added functions `template_matching::match_template_with_mask` and `template_matching::match_template_with_mask_parallel` to support masked templates in template matching.
* Added `L2` variant to the `distance_transform::Norm` enum used to specify the distance function in `distance_transfrom::distance_transform` and several functions in the `morphology` module.
* Added function `filter::laplacian_filter` using a 3x3 approximation to the Laplacian kernel.
* Added function `stats::min_max()` which reports the min and max intensities in an image for each channel.
* Added support for grayscale morphology operators: `grayscale_(dilate|erode|open|close)`.

Breaking changes:
* Added `ThresholdType` parameter to `contrast::threshold{_mut}` to allow configuration of thresholding behaviour. To match the behaviour of `threshold(image, thresh)` from `imageproc 0.24`, use `threshold(image, thresh, ThresholdType::Binary)`.
* Changed the signature of `contrast::stretch_contrast{_mut}` to make the output intensity range configurable. To match the behaviour of `stretch_contrast(image, lower, upper)` from `imageproc 0.24`, use `stretch_contrast(image, lower, upper, 0u8, 255u8)`.
* Changed input parameter to `convex_hull` from `&[Point<T>]` to `impl Into<Vec<Point<T>>>`.
* Removed dependency on `conv` crate and removed function `math::cast`. This replaces `ValueInto<K>` trait bounds with `Into<K>` in many function signatures.

Bug fixes:
* Fix panic when drawing large ellipses.
* Fix `BresenhamLineIter` panic when using non-integer endpoints.
* Fix text rendering for overlapping glyphs, e.g. Arabic.
* Fix Gaussian blur by normalising kernel values.

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
