use crate::definitions::Image;
use image::Pixel;
use proptest::{
    arbitrary::{any, Arbitrary},
    sample::SizeRange,
    strategy::{BoxedStrategy, Strategy},
};
use std::{fmt, ops::RangeInclusive};

/// Create a strategy to generate images with arbitrary dimensions selected
/// within the specified ranges.
pub(crate) fn arbitrary_image<P>(
    width_range: impl Into<SizeRange>,
    height_range: impl Into<SizeRange>,
) -> BoxedStrategy<Image<P>>
where
    P: Pixel + fmt::Debug,
    P::Subpixel: Arbitrary,
    <P::Subpixel as Arbitrary>::Strategy: 'static,
{
    dims(width_range, height_range)
        .prop_flat_map(|(w, h)| arbitrary_image_fixed(w, h))
        .boxed()
}

fn arbitrary_image_fixed<P>(width: u32, height: u32) -> BoxedStrategy<Image<P>>
where
    P: Pixel + fmt::Debug,
    P::Subpixel: Arbitrary,
    <P::Subpixel as Arbitrary>::Strategy: 'static,
{
    let size = (width * height * P::CHANNEL_COUNT as u32) as usize;
    let vecs = proptest::collection::vec(any::<P::Subpixel>(), size);

    vecs.prop_map(move |v| Image::from_vec(width, height, v).unwrap())
        .boxed()
}

fn dims(width: impl Into<SizeRange>, height: impl Into<SizeRange>) -> BoxedStrategy<(u32, u32)> {
    let width = dim(width);
    let height = dim(height);
    width
        .prop_flat_map(move |w| height.clone().prop_map(move |h| (w, h)))
        .boxed()
}

fn dim(range: impl Into<SizeRange>) -> RangeInclusive<u32> {
    let range = range.into();
    range.start() as u32..=range.end_incl() as u32
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
