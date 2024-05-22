//! Macros used for generating documentation

/// A macro for generating the doc-comments for mutable versions of various
/// image processing functions. It takes the name of then non-mut function as an
/// argument as a string literal.
///
/// It uses concat! to generate doc-links to the provided original function name
/// in string literal form.
macro_rules! generate_mut_doc_comment {
    ($name:literal) => {
        concat!(
            "An in-place version of [`",
            $name,
            "()`].\n\nThis function does the same operation as [`",
            $name,
            "()`] but on the `&mut image`\npassed rather than cloning an `&image`. This is faster but you lose the\noriginal image."
        )
    };
}

/// A macro for generating the doc-comments for parallel versions of various
/// image processing functions. It takes the name of then non-parallel function as an
/// argument as a string literal.
///
/// It uses concat! to generate doc-links to the provided original function name
/// in string literal form.
macro_rules! generate_parallel_doc_comment {
    ($name:literal) => {
        concat!(
            "An parallel version of [`",
            $name,
            "()`].\n\nThis function does the same operation as [`",
            $name,
            "()`] but using multi-threading via [`rayon`](https://docs.rs/rayon) iterators.\nRead the top-level crate documentation on parallelism for some of the tradeoffs involved with multi-threaded vs single-threaded image processing functions."
        )
    };
}
