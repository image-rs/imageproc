# imageproc

[![crates.io](https://img.shields.io/crates/v/imageproc.svg)](https://crates.io/crates/imageproc)
[![doc-badge]][doc-link]
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/image-rs/imageproc/blob/master/LICENSE)
[![Dependency status](https://deps.rs/repo/github/image-rs/imageproc/status.svg)](https://deps.rs/repo/github/image-rs/imageproc)

An image processing library, based on the
[`image`](https://github.com/image-rs/image) library.

[API documentation][doc-link]

[doc-badge]: https://docs.rs/imageproc/badge.svg
[doc-link]: https://docs.rs/imageproc

## Goals

A performant, well-tested, well-documented library with a consistent API, suitable for use as
the basis of computer vision applications or graphics editors.

## Non-goals

Maximum genericity over image storages or formats, or support for higher-dimensional images.

## Color Space

Functions in this library implicitly assume that pixels colors are
stored in a linear color space like `RGB` as opposed to a non-linear color
space such as `sRGB`.

If you are not familiar with gamma correction then [this article] contains
an introduction and examples of color artefacts resulting
from not using linear color spaces.

[this article]: https://blog.johnnovak.net/2016/09/21/what-every-coder-should-know-about-gamma/

## Parallelism

This library provides both single-threaded and multi-threaded variations of several functions
by using [rayon](https://github.com/rayon-rs/rayon).

Depending on image size and the amount of work performed per pixel the parallel versions may not 
always be faster - we recommend benchmarking for your specific use-case.

## Crate Features

### Default Features

- `rayon`: enables multi-threaded versions of several functions

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

See [`CONTRIBUTING.md`].