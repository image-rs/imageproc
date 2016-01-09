//! Functions for suppressing non-maximal values.

use std::cmp;

use definitions::{
    Position,
    Score
};

use image::{
    GenericImage,
    ImageBuffer,
    Luma,
    Primitive};

/// Returned image has zeroes for all inputs pixels which do not have the greatest
/// intensity in the (2 * radius + 1) square block centred on them.
/// Ties are resolved lexicographically.
// TODO: Implement a more efficient version.
// TODO: e.g. https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_00446.pdf
pub fn suppress_non_maximum<I, C>(image: &I, radius: u32) -> ImageBuffer<Luma<C>, Vec<C>>
    where I: GenericImage<Pixel = Luma<C>>,
          C: Primitive + Ord + 'static
{
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);
    out.copy_from(image, 0, 0);

    let irad = radius as i32;

    for y in 0..height {
        'test: for x in 0..width {
            let intensity = unsafe { image.unsafe_get_pixel(x, y)[0] };

            // If any pixel in the box has a higher intensity, don't modify the pixel
            for py in cmp::max(0, y as i32 - irad)..cmp::min(y as i32 + irad + 1, height as i32) {
                for px in cmp::max(0, x as i32 - irad)..cmp::min(x as i32 + irad + 1, width as i32) {
                    let v = unsafe { image.unsafe_get_pixel(px as u32, py as u32)[0] };
                    if v > intensity ||
                       (v == intensity && (px as u32, py as u32) < (x, y)){
                        unsafe { out.unsafe_put_pixel(x, y, Luma([C::zero()])) };
                        continue 'test;
                    }
                }
            }
        }
    }

    out
}

/// Returns all items which have the highest score in the
/// (2 * radius + 1) square block centred on them. Ties are resolved lexicographically.
pub fn local_maxima<T>(ts: &[T], radius: u32) -> Vec<T>
    where T: Position + Score + Copy
 {
    let mut ordered_ts = ts.to_vec();
    ordered_ts.sort_by(|c, d| {(c.y(), c.x()).cmp(&(d.y(), d.x()))});
    let height = match ordered_ts.last() {
        Some(t) => t.y(),
        None => 0
    };

    let mut ts_by_row = vec![vec![]; (height + 1) as usize];
    for t in ordered_ts.iter() {
        ts_by_row[t.y() as usize].push(t);
    }

    let mut max_ts = vec![];
    for t in ordered_ts.iter() {
        let cx = t.x();
        let cy = t.y();
        let cs = t.score();

        let mut is_max = true;
        let row_lower = if radius > cy {0} else {cy - radius};
        let row_upper = if cy + radius + 1 > height {height} else {cy + radius + 1};
        for y in row_lower..row_upper {
            for c in ts_by_row[y as usize].iter() {
                if c.x() + radius < cx {
                    continue;
                }
                if c.x() > cx + radius {
                    break;
                }
                if c.score() > cs {
                    is_max = false;
                    break;
                }
                if c.score() < cs {
                    continue;
                }
                // Break tiebreaks lexicographically
                if (c.y(), c.x()) < (cy, cx) {
                    is_max = false;
                    break;
                }
            }
            if !is_max {
                break;
            }
        }

        if is_max {
            max_ts.push(*t);
        }
    }

    max_ts
}

#[cfg(test)]
mod test {
    use super::{
        local_maxima,
        suppress_non_maximum
    };
    use definitions::{
        Position,
        Score
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma,
        Primitive
    };
    use noise::{
        gaussian_noise_mut
    };
    use std::cmp;
    use quickcheck::quickcheck;
    use utils::GrayTestImage;
    use test::Bencher;

    #[derive(PartialEq, Debug, Copy, Clone)]
    struct T {
        x: u32,
        y: u32,
        score: f32
    }

    impl T {
        fn new(x: u32, y: u32, score: f32) -> T {
            T { x: x, y: y, score: score}
        }
    }

    impl Position for T {
        fn x(&self) -> u32 { self.x }
        fn y(&self) -> u32 { self.y }
    }

    impl Score for T {
        fn score(&self) -> f32 { self.score }
    }

    #[test]
    fn test_local_maxima() {
        let ts = vec![
            // Suppress vertically
            T::new(0, 0, 8f32),
            T::new(0, 3, 10f32),
            T::new(0, 6, 9f32),
            // Suppress horizontally
            T::new(5, 5, 10f32),
            T::new(7, 5, 15f32),
            // Tiebreak
            T::new(12, 20, 10f32),
            T::new(13, 20, 10f32),
            T::new(13, 21, 10f32)
        ];

        let expected = vec![
            T::new(0, 3, 10f32),
            T::new(7, 5, 15f32),
            T::new(12, 20, 10f32)
        ];

        let max = local_maxima(&ts, 3);
        assert_eq!(max, expected);
    }

    #[bench]
    fn bench_local_maxima_dense(b: &mut Bencher) {
        let mut ts = vec![];
        for x in 0..20 {
            for y in 0..20 {
                let score = (x * y) % 15;
                ts.push(T::new(x, y, score as f32));
            }
        }
        b.iter(|| local_maxima(&ts, 15));
    }

    #[bench]
    fn bench_local_maxima_sparse(b: &mut Bencher) {
        let mut ts = vec![];
        for x in 0..20 {
            for y in 0..20 {
                ts.push(T::new(50 * x, 50 * y, 50f32));
            }
        }
        b.iter(|| local_maxima(&ts, 15));
    }

    #[test]
    fn test_suppress_non_maximum() {
        let mut image = GrayImage::new(25, 25);
        // Suppress vertically
        image.put_pixel(0, 0, Luma([8u8]));
        image.put_pixel(0, 3, Luma([10u8]));
        image.put_pixel(0, 6, Luma([9u8]));
        // Suppress horizontally
        image.put_pixel(5, 5, Luma([10u8]));
        image.put_pixel(7, 5, Luma([15u8]));
        // Tiebreak
        image.put_pixel(12, 20, Luma([10u8]));
        image.put_pixel(13, 20, Luma([10u8]));
        image.put_pixel(13, 21, Luma([10u8]));

        let mut expected = GrayImage::new(25, 25);
        expected.put_pixel(0, 3, Luma([10u8]));
        expected.put_pixel(7, 5, Luma([15u8]));
        expected.put_pixel(12, 20, Luma([10u8]));

        let actual = suppress_non_maximum(&image, 3);
        assert_pixels_eq!(actual, expected);
    }

    #[bench]
    fn bench_suppress_non_maximum_increasing_gradient(b: &mut Bencher) {
        // Increasing gradient in both directions. This can be a worst-case for
        // early-abort strategies.
        let img = ImageBuffer::from_fn(40, 20, |x, y| Luma([(x + y) as u8]));
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    #[bench]
    fn bench_suppress_non_maximum_decreasing_gradient(b: &mut Bencher) {
        let width = 40u32;
        let height = 20u32;
        let img = ImageBuffer::from_fn(width, height, |x, y| {
            Luma([((width - x) + (height - y)) as u8])
        });
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    #[bench]
    fn bench_suppress_non_maximum_noise(b: &mut Bencher) {
        let mut img: GrayImage = ImageBuffer::new(40, 20);
        gaussian_noise_mut(&mut img, 128f64, 30f64, 1);
        b.iter(|| suppress_non_maximum(&img, 7));
    }

    /// Reference implementation of suppress_non_maximum. Used to validate
    /// the (presumably faster) actual implementation.
    fn suppress_non_maximum_reference<I, C>(image: &I, radius: u32)
        -> ImageBuffer<Luma<C>, Vec<C>>
        where I: GenericImage<Pixel = Luma<C>>,
              C: Primitive + Ord + 'static
    {
        let (width, height) = image.dimensions();
        let mut out = ImageBuffer::new(width, height);
        out.copy_from(image, 0, 0);

        let iradius = radius as i32;
        let iheight = height as i32;
        let iwidth = width as i32;

        // We update zero values from out as we go, so to check intensities
        // we need to read values from the input image.
        for y in 0..height {
           for x in 0..width {
               let intensity = image.get_pixel(x, y)[0];
               let mut is_max = true;

               let y_lower = cmp::max(0, y as i32 - iradius);
               let y_upper = cmp::min(y as i32 + iradius + 1, iheight);
               let x_lower = cmp::max(0, x as i32 - iradius);
               let x_upper = cmp::min(x as i32 + iradius + 1, iwidth);

               for py in y_lower..y_upper {
                   for px in x_lower..x_upper {
                           let v = image.get_pixel(px as u32, py as u32)[0];
                           // Handle intensity tiebreaks lexicographically
                           let candidate_is_lexically_earlier = (px as u32, py as u32) < (x, y);
                           if v > intensity || (v == intensity && candidate_is_lexically_earlier) {
                               is_max = false;
                               break;
                           }
                   }
               }

               if !is_max {
                   out.put_pixel(x, y, Luma([C::zero()]));
               }
           }
        }

        out
    }

    #[test]
    fn test_suppress_non_maximum_matches_reference_implementation() {

        fn prop(image: GrayTestImage) -> bool {
            let expected = suppress_non_maximum_reference(&image.0, 3);
            let actual = suppress_non_maximum(&image.0, 3);
            actual.pixels().eq(expected.pixels())
        }
        quickcheck(prop as fn(GrayTestImage) -> bool);
    }
}
