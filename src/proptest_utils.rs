use crate::definitions::Image;
use image::Pixel;
use proptest::{
    arbitrary::{any, Arbitrary},
    sample::SizeRange,
    strategy::{BoxedStrategy, Strategy},
};
use std::{fmt, ops::RangeInclusive};

/// Create a strategy to generate arbitrary images with dimensions selected
/// within the specified ranges.
pub(crate) fn arbitrary_image<P>(
    width_range: impl Into<SizeRange>,
    height_range: impl Into<SizeRange>,
) -> BoxedStrategy<Image<P>>
where
    P: Pixel + fmt::Debug,
    P::Subpixel: Arbitrary + fmt::Debug,
    <P::Subpixel as Arbitrary>::Strategy: Clone + 'static,
{
    arbitrary_image_with(any::<P::Subpixel>(), width_range, height_range)
}

/// Create a strategy to generate images with a given subpixel strategy and
/// dimensions selected within the specified ranges.
pub(crate) fn arbitrary_image_with<P>(
    subpixels: impl Strategy<Value = P::Subpixel> + Clone + 'static,
    width_range: impl Into<SizeRange>,
    height_range: impl Into<SizeRange>,
) -> BoxedStrategy<Image<P>>
where
    P: Pixel + fmt::Debug,
    P::Subpixel: fmt::Debug,
{
    dims(width_range, height_range)
        .prop_flat_map(move |(w, h)| fixed_image_with(subpixels.clone(), w, h))
        .boxed()
}

fn fixed_image_with<P>(
    strategy: impl Strategy<Value = P::Subpixel> + 'static,
    width: u32,
    height: u32,
) -> BoxedStrategy<Image<P>>
where
    P: Pixel + fmt::Debug,
    P::Subpixel: fmt::Debug,
{
    let size = (width * height * P::CHANNEL_COUNT as u32) as usize;
    let vecs = proptest::collection::vec(strategy, size);

    vecs.prop_map(move |v| Image::from_vec(width, height, v).unwrap())
        .boxed()
}

fn dims(width: impl Into<SizeRange>, height: impl Into<SizeRange>) -> BoxedStrategy<(u32, u32)> {
    let width = to_range(width);
    let height = to_range(height);
    width
        .prop_flat_map(move |w| height.clone().prop_map(move |h| (w, h)))
        .boxed()
}

fn to_range(range: impl Into<SizeRange>) -> RangeInclusive<u32> {
    let range = range.into();
    range.start() as u32..=range.end_incl() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_range() {
        macro_rules! to_range {
            ($range:expr) => {
                to_range($range).collect::<Vec<_>>()
            };
        }

        macro_rules! to_vec {
            ($range:expr) => {
                ($range).map(|x| x as u32).collect::<Vec<_>>()
            };
        }

        assert_eq!(to_range!(0), [0]);
        assert_eq!(to_range!(1), [1]);
        assert_eq!(to_range!(..2), to_vec!(0..2));
        assert_eq!(to_range!(..=2), to_vec!(0..=2));
        assert_eq!(to_range!(2..4), to_vec!(2..4));
        assert_eq!(to_range!(2..=4), to_vec!(2..=4));
        assert_eq!(to_range!(2..2), to_vec!(2..2));
        assert_eq!(to_range!(2..=2), to_vec!(2..=2));
    }
}

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use image::{Luma, Rgb};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_arbitrary_fixed_rgb(img in arbitrary_image::<Rgb<u8>>(3, 7)) {
            assert_eq!(img.width(), 3);
            assert_eq!(img.height(), 7);
        }

        #[test]
        fn test_arbitrary_gray(img in arbitrary_image::<Luma<u8>>(1..30, 2..=150)) {
            assert!((1..30).contains(&img.width()));
            assert!((2..=150).contains(&img.height()));
        }
    }
}
