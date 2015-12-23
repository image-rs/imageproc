imageproc
====
[![Build Status](https://travis-ci.org/PistonDevelopers/imageproc.svg?branch=master)](https://travis-ci.org/PistonDevelopers/imageproc)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/PistonDevelopers/imageproc/blob/master/LICENSE)

An image processing library, based on the [image](https://github.com/PistonDevelopers/image) library. There may initially
be overlap between the functions in this library and those in image::imageops.

This is very much a work in progress. If you have ideas for things that could be done better, or new features you'd like to see, then please create issues for them. Nothing's set in stone.

[API documentation](http://docs.piston.rs/imageproc/imageproc/)

# Goals

A performant, well-tested, well-documented library with a consistent API, suitable for use as the basis of computer vision applications or graphics editors.

# Non-goals

Maximum genericity over image storages or formats, or support for higher-dimensional images.

Full blown computer vision applications (e.g. face recognition or image registration) probably also belong elsewhere, but the line's a bit blurred here (e.g. is image in-painting an image processing task or a computer vision task?). However, worrying about how to structure the code can probably wait until we have more code to structure...

# How to contribute

All pull requests are welcome. Some specific areas that would be great to get some help with are:

* Performance - profiling current code, documenting or fixing performance problems, adding benchmarks, comparisons to other libraries.
* Testing - more unit tests and regression tests. It would be nice to get some property-based testing, and make it easy to add more.
* APIs - are the current APIs hard to use or inconsistent? Some open questions: Should we return Result types more often? How should functions indicate acceptable input image dimensions? Should we use enum arguments or have lots of similarly named functions? What's the best way to get concise code while still allowing control over allocations?
* Documentation - particularly more example code showing what's currently possible. Pretty pictures in this README.
* Feature requests - are there any functions you'd like to see added? Is the library currently unsuitable for your use case for some reason?
* User feedback - have you tried using the library for something? Were you successful? If not then what were the problems you faced?
