use crate::definitions::Image;
use image::Pixel;

/// Equivalent to bucket tool in MS-PAINT
/// Performs 4-way flood-fill based on this algorithm: https://en.wikipedia.org/wiki/Flood_fill#Span_filling
pub fn flood_fill<P>(image: &Image<P>, x: u32, y: u32, fill_with: P) -> Image<P>
where
    P: Pixel + PartialEq,
{
    let mut filled_image = image.clone();
    flood_fill_mut(&mut filled_image, x, y, fill_with);
    filled_image
}

#[doc=generate_mut_doc_comment!("flood_fill")]
pub fn flood_fill_mut<P>(image: &mut Image<P>, x: u32, y: u32, fill_with: P)
where
    P: Pixel + PartialEq,
{
    let target = image.get_pixel(x, y).clone();

    let mut stack = Vec::new();

    stack.push((x as i32, x as i32, y as i32, 1 as i32));
    stack.push((x as i32, x as i32, y as i32 - 1, -1 as i32));

    while !stack.is_empty() {
        let (x1, x2, y, dy) = stack.pop().unwrap();
        let mut x1 = x1;
        let mut x = x1;
        if inside(image, x, y, target) {
            while inside(image, x - 1, y, target) {
                image.put_pixel(x as u32 - 1, y as u32, fill_with);
                x = x - 1;
            }
            if x < x1 {
                stack.push((x, x1 - 1, y - dy, -dy))
            }
        }
        while x1 <= x2 {
            while inside(image, x1, y, target) {
                image.put_pixel(x1 as u32, y as u32, fill_with);
                x1 = x1 + 1;
            }
            if x1 > x {
                stack.push((x, x1 - 1, y + dy, dy))
            }
            if x1 - 1 > x2 {
                stack.push((x2 + 1, x1 - 1, y - dy, -dy))
            }
            x1 = x1 + 1;
            while x1 < x2 && !inside(image, x1, y, target) {
                x1 = x1 + 1
            }
            x = x1
        }
    }
}

/// Determines whether (x,y) is within the image bounds and if the pixel there is equal to target_color
fn inside<P>(image: &Image<P>, x: i32, y: i32, target_pixel: P) -> bool
where
    P: Pixel + PartialEq,
{
    if x < 0 || y < 0 {
        return false;
    }
    let x = x as u32;
    let y = y as u32;
    let (width, height) = image.dimensions();
    x < width && y < height && *image.get_pixel(x, y) == target_pixel
}
