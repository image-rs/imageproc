use image::{GenericImage, ImageBuffer, Rgb};
use imageproc::{drawing::{draw_filled_circle, draw_filled_rect_mut, Canvas}, rect::Rect};

fn main() {
    // Load the image
    let image = ImageBuffer::<Rgb<u8>, Vec<_>>::new(256, 256);

    // Create a mask
    let mask = ImageBuffer::new(image.width(), image.height());
    let shorter_side = mask.width().min(mask.height()) as i32;
    let center_x = mask.width() / 2;
    let center_y = mask.height() / 2;
    let mask = draw_filled_circle(&mask, (center_x as i32, center_y as i32), shorter_side / 4, image::Luma([255u8]));

    let mut canvas = imageproc::drawing::Masked { inner: image, mask };
    draw_filled_rect_mut(&mut canvas, Rect::at(0, 0).of_size(center_x, center_y), Rgb([255u8, 0u8, 0u8]));
    canvas.inner.save("masked.png").unwrap();
}