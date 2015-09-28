
extern crate image;

/// Panics if any pixels differ between the two input images
#[macro_export]
macro_rules! assert_pixels_eq {
    ($actual:expr, $expected:expr) => ({
        let actual_dim = $actual.dimensions();
        let expected_dim = $expected.dimensions();

        if actual_dim != expected_dim {
            panic!("dimensions do not match. \
                actual: {:?}, expected: {:?}", actual_dim, expected_dim)
        }

        let diffs = $actual.pixels()
            .zip($expected.pixels())
            .filter(|&(p, q)| p != q)
            .collect::<Vec<_>>();

        if !diffs.is_empty() {
            let mut err = "pixels do not match. ".to_string();

            let diff_messages = diffs
                .iter()
                .take(5)
                .map(|d| format!("\nactual: {:?}, expected {:?} ", d.0, d.1))
                .collect::<Vec<_>>()
                .join("");

            err.push_str(&diff_messages);
            panic!(err)
        }
     })
}

use std::path::Path;

/// Loads image at given path, panicking on failure
pub fn load_image_or_panic(path: &Path) -> image::DynamicImage {
     image::open(path)
         .ok()
         .expect(&format!("Could not load image at {:?}", path))
 }
