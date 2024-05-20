# Contributing

First off, thank you for considering contributing to `imageproc`.

If your contribution is not straightforward, please consider first creating an
issue to discuss the change as this can save you time by getting the advice of
the maintainers.

If you're working on an issue its a good idea to add a comment to that
issue to tell others to reduce the chance two people work on the same thing
at the same time.

## Testing

Testing this crate requires the `nightly` toolchain due to using the unstable
`test` feature for benchmarks.

## Reporting issues

Before reporting an issue on the [issue
tracker](https://github.com/rust-github/template/issues), please check that it
has not already been reported by searching for some related keywords.

Minimal reproducible examples are appreciated for bug reports.

## Documentation

This crate uses `katexit` to render equations in the documentation.
To open the documentation locally with `katexit` enabled, use:

```sh
cargo doc --open --features=katexit
```

## Pull requests

Try to do one pull request per change.

## Some Areas Needing Work

All pull requests are welcome. Some specific areas that would be great to
get some help with are:

- New features! - are there any functions you'd like to see added? Is the
  library currently unsuitable for your use-case for some reason?
- Performance - profiling current code, documenting or fixing performance
  problems, adding benchmarks, comparisons to other libraries.
- Testing - more unit tests and regression tests. Some more property-based
  testing would be particularly nice.
- Documentation - particularly more example code showing what's currently
  possible. Pretty pictures in this README.
