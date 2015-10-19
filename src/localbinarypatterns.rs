//! Functions for computing [local binary patterns](https://en.wikipedia.org/wiki/Local_binary_patterns).

use image::{
    GenericImage,
    Luma
};
use std::cmp;

/// Computes the basic local binary pattern of a pixel, or None
/// if it's too close to the image boundary.
pub fn local_binary_pattern<I>(image: &I, x: u32, y: u32) -> Option<u8>
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return None;
    }

    // TODO: It might be better to make this function private, and
    // TODO: require the caller to only provide valid x and y coordinates.
    if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
        return None;
    }

    // TODO: As with the fast corner detectors, this would be more efficient if
    // TODO: generated a list of pixel offsets once per image, and used those
    // TODO: offsets directly when reading pixels. To do this we'd need some traits
    // TODO: for images whose pixels are stored in contiguous rows/columns.
    let mut pattern = 0u8;

    let center = image.get_pixel(x, y)[0];

    // The sampled pixels have the following labels.
    //
    // 7  0  1
    // 6  p  2
    // 5  4  3
    //
    // The nth bit of a pattern is 1 if the pixel p
    // is strictly brighter than the neighbor in position n.
    let neighbors = [
        image.get_pixel(x    , y - 1)[0],
        image.get_pixel(x + 1, y - 1)[0],
        image.get_pixel(x + 1, y    )[0],
        image.get_pixel(x + 1, y + 1)[0],
        image.get_pixel(x    , y + 1)[0],
        image.get_pixel(x - 1, y + 1)[0],
        image.get_pixel(x - 1, y    )[0],
        image.get_pixel(x - 1, y - 1)[0]
    ];

    for i in 0..8 {
        pattern |= (1 & (neighbors[i] < center) as u8) << i;
    }

    Some(pattern)
}

// TODO: add lookup tables from pattern to pattern class, one just assigning
// TODO: the least circular shift and the other lumping all non-uniform patterns together

/// Returns the minimum value over all rotations of a byte.
pub fn min_shift(byte: u8) -> u8 {
    let mut min = byte;
    for i in 1..8 {
        min = cmp::min(min, byte.rotate_right(i));
    }
    min
}

#[cfg(test)]
mod test {

    use super::{
        local_binary_pattern,
        min_shift
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma
    };
    use test;

    #[test]
    fn test_local_binary_pattern() {

        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            06, 11, 14,
            09, 10, 10,
            19, 00, 22]).unwrap();

        let expected = 0b11010000;
        let pattern = local_binary_pattern(&image, 1, 1).unwrap();

        assert_eq!(pattern, expected);
    }

    #[test]
    fn test_min_shift() {
        let byte = 0b10110100;
        assert_eq!(min_shift(byte), 0b00101101);
    }
}
