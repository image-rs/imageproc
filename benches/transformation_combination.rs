#![feature(test)]
extern crate test;

#[cfg(test)]
mod tests {
    use imageproc::{
        _geometric_transformations::Projection as _Projection,
        geometric_transformations::Projection,
    };
    use lazy_static::lazy_static;
    use rand::{thread_rng, Rng};
    use test::{black_box, Bencher};

    lazy_static! {
        static ref TRANSLATIONS: (Vec<Projection>, Vec<_Projection>) = {
            let mut v = Vec::with_capacity(1000);
            let mut w = Vec::with_capacity(1000);
            let mut rng = thread_rng();
            for _ in 0..1000 {
                let (a, b) = (rng.gen_range(-1e5, 1e5), rng.gen_range(-1e5, 1e5));
                v.push(Projection::translate(a, b));
                w.push(_Projection::translate(a, b));
            }
            (v, w)
        };
        static ref TRANSLATIONS_AND_ROTATIONS: (Vec<Projection>, Vec<_Projection>) = {
            let mut v = Vec::with_capacity(1000);
            let mut w = Vec::with_capacity(1000);
            let mut rng = thread_rng();
            for i in 0..1000 {
                if i % 2 == 0 {
                    let (a, b) = (rng.gen_range(-1e5, 1e5), rng.gen_range(-1e5, 1e5));
                    v.push(Projection::translate(a, b));
                    w.push(_Projection::translate(a, b));
                } else {
                    const PI: f32 = std::f32::consts::PI;
                    let a = rng.gen_range(-PI, PI);
                    v.push(Projection::rotate(a));
                    w.push(_Projection::rotate(a));
                }
            }
            (v, w)
        };
    }

    #[bench]
    fn translationxtranslation(b: &mut Bencher) {
        b.iter(|| {
            let first = TRANSLATIONS.0[0].clone();
            // Inner closure, the actual test
            for _ in 1..100 {
                black_box(
                    TRANSLATIONS
                        .0
                        .iter()
                        .skip(1)
                        .fold(first, |acc, curr| *curr * acc),
                );
            }
        });
    }

    #[bench]
    fn translationxtranslation_fast(b: &mut Bencher) {
        b.iter(|| {
            let first = TRANSLATIONS.1[0].clone();
            // Inner closure, the actual test
            for _ in 1..100 {
                black_box(
                    TRANSLATIONS
                        .1
                        .iter()
                        .skip(1)
                        .fold(first, |acc, curr| *curr * acc),
                );
            }
        });
    }

    #[bench]
    fn translationxrotation(b: &mut Bencher) {
        b.iter(|| {
            let first = TRANSLATIONS_AND_ROTATIONS.0[0].clone();
            // Inner closure, the actual test
            for _ in 1..100 {
                black_box(
                    TRANSLATIONS
                        .0
                        .iter()
                        .skip(1)
                        .fold(first, |acc, curr| *curr * acc),
                );
            }
        });
    }

    #[bench]
    fn translationxrotation_fast(b: &mut Bencher) {
        b.iter(|| {
            let first = TRANSLATIONS_AND_ROTATIONS.1[0].clone();
            // Inner closure, the actual test
            for _ in 1..100 {
                black_box(
                    TRANSLATIONS
                        .1
                        .iter()
                        .skip(1)
                        .fold(first, |acc, curr| *curr * acc),
                );
            }
        });
    }
}
