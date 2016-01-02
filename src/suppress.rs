//! Functions for suppressing non-maximal values.

use definitions::{Position, Score};

/// Returns all items which have the highest score in the
/// (2 * radius + 1) square block centred on them.
pub fn suppress_non_maximum<T>(ts: &[T], radius: u32) -> Vec<T>
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
        let min_row = if radius > cy {0} else {cy - radius};
        let max_row = if cy + radius > height {height} else {cy + radius};
        for y in min_row..max_row {
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
        suppress_non_maximum
    };
    use definitions::{
        Position,
        Score
    };
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
        fn x(&self) -> u32 {
            self.x
        }

        fn y(&self) -> u32 {
            self.y
        }
    }

    impl Score for T {
        fn score(&self) -> f32 {
            self.score
        }
    }

    #[test]
    fn test_suppress_non_maximum() {
        let ts = vec![
            // Suppress vertically
            T::new(0, 0, 10f32),
            T::new(0, 2, 8f32),
            // Suppress horizontally
            T::new(5, 5, 10f32),
            T::new(7, 5, 15f32),
            // Tiebreak
            T::new(12, 20, 10f32),
            T::new(13, 20, 10f32),
            T::new(13, 21, 10f32)
        ];

        let expected = vec![
            T::new(0, 0, 10f32),
            T::new(7, 5, 15f32),
            T::new(12, 20, 10f32)
        ];

        let max = suppress_non_maximum(&ts, 3);
        assert_eq!(max, expected);
    }

    #[bench]
    fn bench_suppress_non_maximum(b: &mut Bencher) {
        let mut ts = vec![];
        // Large contiguous block
        for x in 0..50 {
            for y in 0..50 {
                let score = (x * y) % 15;
                ts.push(T::new(x, y, score as f32));
            }
        }

        // Isolated values
        for x in 0..50 {
            for y in 0..50 {
                ts.push(T::new(10 * x + 110, 10 * y + 110, 50f32));
            }
        }

        b.iter(|| suppress_non_maximum(&ts, 15));
    }
}
