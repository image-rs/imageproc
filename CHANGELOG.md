# Change Log

## [0.26.1] - 2026-02-28

[0.26.1]: https://github.com/image-rs/imageproc/releases/tag/v0.26.1

Bug fixes:
* fix `compose::overlay` pixel blending by @cospectrum in https://github.com/image-rs/imageproc/pull/754
* fix `compose::overlay` boundary calculations by @cospectrum in https://github.com/image-rs/imageproc/pull/754
* fix `compose::replace` boundary calculations by @cospectrum in https://github.com/image-rs/imageproc/pull/754

## [0.26.0] - 2026-01-01

[0.26.0]: https://github.com/image-rs/imageproc/releases/tag/v0.26.0

New features:

* Add `AverageHash` by @cospectrum
* Add `PHash` by @cospectrum in https://github.com/image-rs/imageproc/pull/709
* Make `hough::intersection_points` public by @theotherphil in https://github.com/image-rs/imageproc/pull/613
* Color Bilateral Filter by @ripytide in https://github.com/image-rs/imageproc/pull/606
* Parallel Pixel Maps by @ripytide in https://github.com/image-rs/imageproc/pull/602
* Add `filter_map_parallel` by @ripytide in https://github.com/image-rs/imageproc/pull/642
* Add compose module and functions by @ripytide in https://github.com/image-rs/imageproc/pull/662
* Add rotation module and functions by @ripytide in https://github.com/image-rs/imageproc/pull/669
* Add `replace()` and `overlay()` functions by @ripytide in https://github.com/image-rs/imageproc/pull/666
* Expand `stetch_contrast()` to color images also using `u8` by @ripytide in https://github.com/image-rs/imageproc/pull/670
* Flood-fill by @tkallady in https://github.com/image-rs/imageproc/pull/684
* Implement Kapur's algorithm for binary thresholding (#696) by @o-tho in https://github.com/image-rs/imageproc/pull/699
* Add `rotate_about_center_no_crop` to prevent pixel loss during image rotations by @Tikitikitikidesuka in https://github.com/image-rs/imageproc/pull/688

Performance improvements:

* Improved `draw_text_mut` performance by @Icekey in https://github.com/image-rs/imageproc/pull/663

Breaking changes:

* Removed `filter3x3`, use `Kernel` + `filter_clamped` instead
* Add `delta` parameter to `adaptive_threshold` by @Dantsz in https://github.com/image-rs/imageproc/pull/637
* Make external dependencies optional by @OrangeHoopla in https://github.com/image-rs/imageproc/pull/736
* MSRV v1.87 / 2024 Edition / `nalgebra` v0.34 by @paolobarbolini in https://github.com/image-rs/imageproc/pull/748

Bug fixes:

* fix `draw_filled_rect` panic for empty image by @cospectrum in https://github.com/image-rs/imageproc/pull/711
* fix `text_size` by @cospectrum in https://github.com/image-rs/imageproc/pull/689
* rewrite `column_running_sum` with better safety by @cospectrum in https://github.com/image-rs/imageproc/pull/731
* Fix fast corner 9 algorithm by @LMJW in https://github.com/image-rs/imageproc/pull/680
* handle polygon draw case where start point equals end point by @fs-99 in https://github.com/image-rs/imageproc/pull/682
* Detailed error message for generic type parameter of find_contours_with_threshold (#694) by @sakird in https://github.com/image-rs/imageproc/pull/695
* fix integration tests by @bioinformatist in https://github.com/image-rs/imageproc/pull/739

## [0.25.0] - 2024-05-19

[0.25.0]: https://github.com/image-rs/imageproc/releases/tag/v0.25.0

New features:

* Added functions `template_matching::match_template_with_mask`
  and `template_matching::match_template_with_mask_parallel` to support masked templates in template matching.
* Added `L2` variant to the `distance_transform::Norm` enum used to specify the distance function
  in `distance_transfrom::distance_transform` and several functions in the `morphology` module.
* Added function `filter::laplacian_filter` using a 3x3 approximation to the Laplacian kernel.
* Added function `stats::min_max()` which reports the min and max intensities in an image for each channel.
* Added support for grayscale morphology operators: `grayscale_(dilate|erode|open|close)`.

Breaking changes:

* Added `ThresholdType` parameter to `contrast::threshold{_mut}` to allow configuration of thresholding behaviour. To
  match the behaviour of `threshold(image, thresh)` from `imageproc 0.24`,
  use `threshold(image, thresh, ThresholdType::Binary)`.
* Changed the signature of `contrast::stretch_contrast{_mut}` to make the output intensity range configurable. To match
  the behaviour of `stretch_contrast(image, lower, upper)` from `imageproc 0.24`,
  use `stretch_contrast(image, lower, upper, 0u8, 255u8)`.
* Changed input parameter to `convex_hull` from `&[Point<T>]` to `impl Into<Vec<Point<T>>>`.
* Removed dependency on `conv` crate and removed function `math::cast`. This replaces `ValueInto<K>` trait bounds
  with `Into<K>` in many function signatures.

Bug fixes:

* Fix panic when drawing large ellipses.
* Fix `BresenhamLineIter` panic when using non-integer endpoints.
* Fix text rendering for overlapping glyphs, e.g. Arabic.
* Fix Gaussian blur by normalising kernel values.

## [0.24.0] - 2024-03-16

[0.24.0]: https://github.com/image-rs/imageproc/releases/tag/v0.24.0

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

## 0.23.0 - 2022-04-10

...

## 0.6.1 - 2016-12-28

- Fixed bug in draw_line_segment_mut when line extends outside of image bounds.
- Generalised connected_components to handle arbitrary equatable pixel types.
- Added support for drawing hollow and filled circles.
- Added support for drawing anti-aliased lines, and convex polygons.
- Added adaptive_threshold function.

## 0.6.0 - 2016-05-07

No change log kept for this or earlier versions.
