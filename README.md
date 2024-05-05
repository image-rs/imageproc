imageproc
====

[![crates.io](https://img.shields.io/crates/v/imageproc.svg)](https://crates.io/crates/imageproc)
[![doc-badge]][doc-link]
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/image-rs/imageproc/blob/master/LICENSE)
[![Dependency status](https://deps.rs/repo/github/image-rs/imageproc/status.svg)](https://deps.rs/repo/github/image-rs/imageproc)

An image processing library, based on the [image](https://github.com/image-rs/image) library. There may initially
be overlap between the functions in this library and those in `image::imageops`.

This is very much a work in progress. If you have ideas for things that could be done better, or new features you'd like to see, then please create issues for them. Nothing's set in stone.

[API documentation][doc-link]

[doc-badge]: https://docs.rs/imageproc/badge.svg
[doc-link]: https://docs.rs/imageproc

# Goals

A performant, well-tested, well-documented library with a consistent API, suitable for use as the basis of computer vision applications or graphics editors.

# Non-goals

Maximum genericity over image storages or formats, or support for higher-dimensional images.

Full blown computer vision applications (e.g. face recognition or image registration) probably also belong elsewhere, but the line's a bit blurred here (e.g. is image in-painting an image processing task or a computer vision task?). However, worrying about how to structure the code can probably wait until we have more code to structure...

# Crate Features

Imageproc is built with these features enabled by default:

-   `rayon` enables multithreading for certain operations (e.g., geometric transformations) via [rayon](https://github.com/rayon-rs/rayon)

Optionally, the following dependencies can be enabled:

-   `property-testing` exposes helper types and methods to enable property testing via [quickcheck](https://github.com/BurntSushi/quickcheck)
-   `display-window` enables the displaying of images (using `imageproc::window`) with [sdl2](https://github.com/Rust-SDL2/rust-sdl2)

# How to contribute

All pull requests are welcome. Some specific areas that would be great to get some help with are:

* New features! If you're planning on adding some new functions or modules, please create an issue with a name along the lines of "Add [feature name]" and assign it to yourself (or comment on the issue that you're planning on doing it). This way we'll not have multiple people working on the same functionality.
* Performance - profiling current code, documenting or fixing performance problems, adding benchmarks, comparisons to other libraries.
* Testing - more unit tests and regression tests. Some more property-based testing would be particularly nice.
* APIs - are the current APIs hard to use or inconsistent? Some open questions: Should we return Result types more often? How should functions indicate acceptable input image dimensions? Should we use enum arguments or have lots of similarly named functions? What's the best way to get concise code while still allowing control over allocations?
* Documentation - particularly more example code showing what's currently possible. Pretty pictures in this README.
* Feature requests - are there any functions you'd like to see added? Is the library currently unsuitable for your use case for some reason?

## Documentation

This crate uses `katexit` to render equations in the documentation.
To open the documentation locally with `katexit` enabled, use
```sh
cargo doc --open --features=katexit
```
