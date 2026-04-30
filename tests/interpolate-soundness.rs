use image::{GrayImage, Luma};
use imageproc::definitions::Image;
use imageproc::geometric_transformations::{warp_into_with, Interpolation};

#[test]
fn warp_with_nan_mapping_into_empty_image_is_oob() {
    let src: Image<Luma<u8>> = GrayImage::new(0, 0);
    let mut out: Image<Luma<u8>> = GrayImage::new(1, 1);

    let nan_mapping = |_x: f32, _y: f32| (f32::NAN, f32::NAN);

    warp_into_with(
        &src,
        nan_mapping,
        Interpolation::Bilinear,
        Luma([0u8]),
        &mut out,
    );
}
