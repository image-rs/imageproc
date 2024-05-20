# imageproc

[![crates.io](https://img.shields.io/crates/v/imageproc.svg)](https://crates.io/crates/imageproc)
[![doc-badge]][doc-link]
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/image-rs/imageproc/blob/master/LICENSE)
[![Dependency status](https://deps.rs/repo/github/image-rs/imageproc/status.svg)](https://deps.rs/repo/github/image-rs/imageproc)

An image processing library, based on the
[`image`](https://github.com/image-rs/image) library.

This is very much a work in progress. If you have ideas for things that
could be done better, or new features you'd like to see, then please create
issues for them. Nothing's set in stone.

[API documentation][doc-link]

[doc-badge]: https://docs.rs/imageproc/badge.svg
[doc-link]: https://docs.rs/imageproc

## Goals

- Performance
- Well-Tested
- Good Documentation
- Consistent API
- Suitable for use as the basis for computed vision and graphics editor
  applications

## Non-goals

- Maximum genericness over image storages or formats, or support for
  higher-dimensional images.
- Full blown computer vision applications (e.g. face recognition or image
  registration) probably also belong elsewhere, but the line is a bit
  blurred here (e.g. is image in-painting an image processing task or a
  computer vision task?). However, worrying about how to structure the code
  can probably wait until we have more code to structure...

## Color Space

Throughout this library functions implicitly assume that pixels colors are
stored in a linear color space like `RGB` as opposed to a non-linear color
space such as `sRGB`. This is because simplifies the implementation
complexity of various functions which average pixels by simply linearly
averaging their channels independently, however, for a non-linear color
space this would not be possible.Therefore if you care about color
correctness it is important to convert your images into a linear color
space before processing them with any functions from this library.

If you pass in a non-linear color space image anyway the functions will
still work, but you might notice incorrect color artifacting which may or may
not be problem for your particular use-case.

## Parallelism

Many image processing function are [embarrassingly
parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel), hence
this library provides both single-threaded and multi-threaded variations
wherever possible by using [rayon](https://github.com/rayon-rs/rayon). It
is important to note that which variation of a function (non-parallel vs
parallel) is faster will entirely depend on both the size of the image and
the complexity of the operation. Generally speaking, operations on larger
images with more complex calculations are faster in parallel than
sequentially and smaller images with simpler calculations are faster
sequentially.

It is often hard to predict which will be faster, therefore, it is important
to benchmark your particular use-cases if you want to be confident you are
using the fastest variations.

## Crate Features

### Default Features

- `rayon`: enables multi-threading variations of various functions

### Optional Features

- `katexit`: enables latex in documentation via
  [katexit](https://github.com/termoshtt/katexit)
- `property-testing`: enables `quickcheck`
- `quickcheck`: exposes helper types and methods to enable property testing
  via [quickcheck](https://github.com/BurntSushi/quickcheck)
- `display-window`: enables `sdl2`
- `sdl2`: enables the displaying of images (using `imageproc::window`) with
  [sdl2](https://github.com/Rust-SDL2/rust-sdl2)

## How to contribute

All contributions are welcome! Read the `CONTRIBUTING.md` for more
information.
