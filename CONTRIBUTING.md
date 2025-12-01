# Contributing

If your contribution is not straightforward, consider first creating an
issue to discuss the change as this can save you time by getting the advice of
the maintainers.

If you're working on an issue it's a good idea to add a comment to that
issue to tell others and reduce the chance that two people work on the same thing
at the same time.

## Testing

Testing this crate requires the `nightly` toolchain due to using the unstable
`test` feature for benchmarks:

```sh
cargo +nightly test
```

Furthermore, one should also call [Miri](https://github.com/rust-lang/miri) to check for undefined behavior:

```sh
cargo +nightly miri nextest run --no-default-features --features=image/default
```

## Reporting issues

Before reporting an issue on the [issue
tracker](https://github.com/rust-github/template/issues), please check that it
has not already been reported by searching for some related keywords.

Minimal reproducible examples are appreciated for bug reports.

## Documentation

This crate uses `katexit` to render equations in documentation.
To open the documentation locally with `katexit` enabled, run:

```sh
cargo doc --open --features=katexit
```

## Help wanted

All pull requests are welcome. Some specific areas that would be great to
get some help with are:

- Performance - profiling current code, documenting or fixing performance
  problems, adding benchmarks, comparisons to other libraries
- Testing - more unit tests and regression tests. Some more property-based
  testing would be particularly nice
- Documentation - particularly more example code showing what's currently
  possible
