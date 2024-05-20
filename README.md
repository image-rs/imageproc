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
