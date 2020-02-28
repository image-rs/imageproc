use crate::definitions::Image;
use crate::drawing::Canvas;
use image::{GenericImage, ImageBuffer, Pixel};
use std::f32;
use std::i32;
use std::mem::{swap, transmute};

/// Iterates over the coordinates in a line segment using
/// [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub struct BresenhamLineIter {
    dx: f32,
    dy: f32,
    x: i32,
    y: i32,
    error: f32,
    end_x: i32,
    is_steep: bool,
    y_step: i32,
}

impl BresenhamLineIter {
    /// Creates a [`BresenhamLineIter`](struct.BresenhamLineIter.html) which will iterate over the integer coordinates
    /// between `start` and `end`.
    pub fn new(start: (f32, f32), end: (f32, f32)) -> BresenhamLineIter {
        let (mut x0, mut y0) = (start.0, start.1);
        let (mut x1, mut y1) = (end.0, end.1);

        let is_steep = (y1 - y0).abs() > (x1 - x0).abs();
        if is_steep {
            swap(&mut x0, &mut y0);
            swap(&mut x1, &mut y1);
        }

        if x0 > x1 {
            swap(&mut x0, &mut x1);
            swap(&mut y0, &mut y1);
        }

        let dx = x1 - x0;

        BresenhamLineIter {
            dx,
            dy: (y1 - y0).abs(),
            x: x0 as i32,
            y: y0 as i32,
            error: dx / 2f32,
            end_x: x1 as i32,
            is_steep,
            y_step: if y0 < y1 { 1 } else { -1 },
        }
    }
}

impl Iterator for BresenhamLineIter {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<(i32, i32)> {
        if self.x > self.end_x {
            None
        } else {
            let ret = if self.is_steep {
                (self.y, self.x)
            } else {
                (self.x, self.y)
            };

            self.x += 1;
            self.error -= self.dy;
            if self.error < 0f32 {
                self.y += self.y_step;
                self.error += self.dx;
            }

            Some(ret)
        }
    }
}

fn clamp(x: f32, upper_bound: u32) -> f32 {
    if x < 0f32 {
        return 0f32;
    }
    if x >= upper_bound as f32 {
        return (upper_bound - 1) as f32;
    }
    x
}

fn clamp_point<I: GenericImage>(p: (f32, f32), image: &I) -> (f32, f32) {
    (clamp(p.0, image.width()), clamp(p.1, image.height()))
}

/// Iterates over the image pixels in a line segment using
/// [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub struct BresenhamLinePixelIter<'a, P: Pixel + 'static> {
    iter: BresenhamLineIter,
    image: &'a Image<P>,
}

impl<'a, P: Pixel + 'static> BresenhamLinePixelIter<'a, P> {
    /// Creates a [`BresenhamLinePixelIter`](struct.BresenhamLinePixelIter.html) which will iterate over
    /// the image pixels with coordinates between `start` and `end`.
    pub fn new(
        image: &Image<P>,
        start: (f32, f32),
        end: (f32, f32),
    ) -> BresenhamLinePixelIter<'_, P> {
        assert!(
            image.width() >= 1 && image.height() >= 1,
            "BresenhamLinePixelIter does not support empty images"
        );
        let iter = BresenhamLineIter::new(clamp_point(start, image), clamp_point(end, image));
        BresenhamLinePixelIter { iter, image }
    }
}

impl<'a, P: Pixel + 'static> Iterator for BresenhamLinePixelIter<'a, P> {
    type Item = &'a P;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|p| self.image.get_pixel(p.0 as u32, p.1 as u32))
    }
}

/// Iterates over the image pixels in a line segment using
/// [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub struct BresenhamLinePixelIterMut<'a, P: Pixel + 'static> {
    iter: BresenhamLineIter,
    image: &'a mut Image<P>,
}

impl<'a, P: Pixel + 'static> BresenhamLinePixelIterMut<'a, P> {
    /// Creates a [`BresenhamLinePixelIterMut`](struct.BresenhamLinePixelIterMut.html) which will iterate over
    /// the image pixels with coordinates between `start` and `end`.
    pub fn new(
        image: &mut Image<P>,
        start: (f32, f32),
        end: (f32, f32),
    ) -> BresenhamLinePixelIterMut<'_, P> {
        assert!(
            image.width() >= 1 && image.height() >= 1,
            "BresenhamLinePixelIterMut does not support empty images"
        );
        // The next two assertions are for https://github.com/image-rs/imageproc/issues/281
        assert!(P::CHANNEL_COUNT > 0);
        assert!(
            image.width() < i32::max_value() as u32 && image.height() < i32::max_value() as u32,
            "Image dimensions are too large"
        );
        let iter = BresenhamLineIter::new(clamp_point(start, image), clamp_point(end, image));
        BresenhamLinePixelIterMut { iter, image }
    }
}

impl<'a, P: Pixel + 'static> Iterator for BresenhamLinePixelIterMut<'a, P> {
    type Item = &'a mut P;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|p| self.image.get_pixel_mut(p.0 as u32, p.1 as u32))
            .map(|p| unsafe { transmute(p) })
    }
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// Uses [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub fn draw_line_segment<I>(
    image: &I,
    start: (f32, f32),
    end: (f32, f32),
    color: I::Pixel,
) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_line_segment_mut(&mut out, start, end, color);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// Uses [Bresenham's line drawing algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm).
pub fn draw_line_segment_mut<C>(canvas: &mut C, start: (f32, f32), end: (f32, f32), color: C::Pixel)
where
    C: Canvas,
    C::Pixel: 'static,
{
    let (width, height) = canvas.dimensions();
    let in_bounds = |x, y| x >= 0 && x < width as i32 && y >= 0 && y < height as i32;

    let line_iterator = BresenhamLineIter::new(start, end);

    for point in line_iterator {
        let x = point.0;
        let y = point.1;

        if in_bounds(x, y) {
            canvas.draw_pixel(x as u32, y as u32, color);
        }
    }
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// The parameters of blend are (line color, original color, line weight).
/// Consider using [`interpolate`](fn.interpolate.html) for blend.
/// Uses [Xu's line drawing algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm).
pub fn draw_antialiased_line_segment<I, B>(
    image: &I,
    start: (i32, i32),
    end: (i32, i32),
    color: I::Pixel,
    blend: B,
) -> Image<I::Pixel>
where
    I: GenericImage,
    I::Pixel: 'static,
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    let mut out = ImageBuffer::new(image.width(), image.height());
    out.copy_from(image, 0, 0).unwrap();
    draw_antialiased_line_segment_mut(&mut out, start, end, color, blend);
    out
}

/// Draws as much of the line segment between start and end as lies inside the image bounds.
/// The parameters of blend are (line color, original color, line weight).
/// Consider using [`interpolate`](fn.interpolate.html) for blend.
/// Uses [Xu's line drawing algorithm](https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm).
pub fn draw_antialiased_line_segment_mut<I, B>(
    image: &mut I,
    start: (i32, i32),
    end: (i32, i32),
    color: I::Pixel,
    blend: B,
) where
    I: GenericImage,
    I::Pixel: 'static,
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    let (mut x0, mut y0) = (start.0, start.1);
    let (mut x1, mut y1) = (end.0, end.1);

    let is_steep = (y1 - y0).abs() > (x1 - x0).abs();

    if is_steep {
        if y0 > y1 {
            swap(&mut x0, &mut x1);
            swap(&mut y0, &mut y1);
        }
        let plotter = Plotter {
            image,
            transform: |x, y| (y, x),
            blend,
        };
        plot_wu_line(plotter, (y0, x0), (y1, x1), color);
    } else {
        if x0 > x1 {
            swap(&mut x0, &mut x1);
            swap(&mut y0, &mut y1);
        }
        let plotter = Plotter {
            image,
            transform: |x, y| (x, y),
            blend,
        };
        plot_wu_line(plotter, (x0, y0), (x1, y1), color);
    };
}

fn plot_wu_line<I, T, B>(
    mut plotter: Plotter<'_, I, T, B>,
    start: (i32, i32),
    end: (i32, i32),
    color: I::Pixel,
) where
    I: GenericImage,
    I::Pixel: 'static,
    T: Fn(i32, i32) -> (i32, i32),
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let gradient = dy as f32 / dx as f32;
    let mut fy = start.1 as f32;

    for x in start.0..(end.0 + 1) {
        plotter.plot(x, fy as i32, color, 1.0 - fy.fract());
        plotter.plot(x, fy as i32 + 1, color, fy.fract());
        fy += gradient;
    }
}

struct Plotter<'a, I, T, B>
where
    I: GenericImage,
    I::Pixel: 'static,
    T: Fn(i32, i32) -> (i32, i32),
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    image: &'a mut I,
    transform: T,
    blend: B,
}

impl<'a, I, T, B> Plotter<'a, I, T, B>
where
    I: GenericImage,
    I::Pixel: 'static,
    T: Fn(i32, i32) -> (i32, i32),
    B: Fn(I::Pixel, I::Pixel, f32) -> I::Pixel,
{
    fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.image.width() as i32 && y >= 0 && y < self.image.height() as i32
    }

    pub fn plot(&mut self, x: i32, y: i32, line_color: I::Pixel, line_weight: f32) {
        let (x_trans, y_trans) = (self.transform)(x, y);
        if self.in_bounds(x_trans, y_trans) {
            let original = self.image.get_pixel(x_trans as u32, y_trans as u32);
            let blended = (self.blend)(line_color, original, line_weight);
            self.image
                .put_pixel(x_trans as u32, y_trans as u32, blended);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    // As draw_line_segment is implemented in terms of BresenhamLineIter we
    // haven't bothered wriing any tests specifically for BresenhamLineIter itself.

    // Octants for line directions:
    //
    //   \ 5 | 6 /
    //   4 \ | / 7
    //   ---   ---
    //   3 / | \ 0
    //   / 2 | 1 \

    #[test]
    fn test_draw_line_segment_horizontal() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            4, 4, 4, 4, 4;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1);

        let right = draw_line_segment(&image, (-3f32, 1f32), (6f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(right, expected);

        let left = draw_line_segment(&image, (6f32, 1f32), (-3f32, 1f32), Luma([4u8]));
        assert_pixels_eq!(left, expected);
    }

    #[test]
    fn test_draw_line_segment_oct0_and_oct4() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 9, 9, 1, 1;
            1, 1, 1, 9, 9;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1);

        let oct0 = draw_line_segment(&image, (1f32, 1f32), (4f32, 2f32), Luma([9u8]));
        assert_pixels_eq!(oct0, expected);

        let oct4 = draw_line_segment(&image, (4f32, 2f32), (1f32, 1f32), Luma([9u8]));
        assert_pixels_eq!(oct4, expected);
    }

    #[test]
    fn test_draw_line_segment_diagonal() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 6, 1, 1, 1;
            1, 1, 6, 1, 1;
            1, 1, 1, 6, 1;
            1, 1, 1, 1, 1);

        let down_right = draw_line_segment(&image, (1f32, 1f32), (3f32, 3f32), Luma([6u8]));
        assert_pixels_eq!(down_right, expected);

        let up_left = draw_line_segment(&image, (3f32, 3f32), (1f32, 1f32), Luma([6u8]));
        assert_pixels_eq!(up_left, expected);
    }

    #[test]
    fn test_draw_line_segment_oct1_and_oct5() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            5, 1, 1, 1, 1;
            5, 1, 1, 1, 1;
            5, 1, 1, 1, 1;
            1, 5, 1, 1, 1;
            1, 5, 1, 1, 1);

        let oct1 = draw_line_segment(&image, (0f32, 0f32), (1f32, 4f32), Luma([5u8]));
        assert_pixels_eq!(oct1, expected);

        let oct5 = draw_line_segment(&image, (1f32, 4f32), (0f32, 0f32), Luma([5u8]));
        assert_pixels_eq!(oct5, expected);
    }

    #[test]
    fn test_draw_line_segment_vertical() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 1, 8, 1;
            1, 1, 1, 8, 1;
            1, 1, 1, 8, 1;
            1, 1, 1, 1, 1);

        let down = draw_line_segment(&image, (3f32, 1f32), (3f32, 3f32), Luma([8u8]));
        assert_pixels_eq!(down, expected);

        let up = draw_line_segment(&image, (3f32, 3f32), (3f32, 1f32), Luma([8u8]));
        assert_pixels_eq!(up, expected);
    }

    #[test]
    fn test_draw_line_segment_oct2_and_oct6() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 4, 1, 1;
            1, 1, 4, 1, 1;
            1, 4, 1, 1, 1;
            1, 4, 1, 1, 1;
            1, 1, 1, 1, 1);

        let oct2 = draw_line_segment(&image, (2f32, 0f32), (1f32, 3f32), Luma([4u8]));
        assert_pixels_eq!(oct2, expected);

        let oct6 = draw_line_segment(&image, (1f32, 3f32), (2f32, 0f32), Luma([4u8]));
        assert_pixels_eq!(oct6, expected);
    }

    #[test]
    fn test_draw_line_segment_oct3_and_oct7() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 1, 2, 2;
            2, 2, 2, 1, 1);

        let oct3 = draw_line_segment(&image, (0f32, 4f32), (5f32, 3f32), Luma([2u8]));
        assert_pixels_eq!(oct3, expected);

        let oct7 = draw_line_segment(&image, (5f32, 3f32), (0f32, 4f32), Luma([2u8]));
        assert_pixels_eq!(oct7, expected);
    }

    #[test]
    fn test_draw_antialiased_line_segment_horizontal_and_vertical() {
        use crate::pixelops::interpolate;
        use image::imageops::rotate270;

        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 2, 2, 2, 2;
            1, 1, 1, 1, 1);

        let color = Luma([2u8]);
        // Deliberately ends one pixel out of bounds
        let right = draw_antialiased_line_segment(&image, (1, 3), (5, 3), color, interpolate);
        assert_pixels_eq!(right, expected);

        // Deliberately starts one pixel out of bounds
        let left = draw_antialiased_line_segment(&image, (5, 3), (1, 3), color, interpolate);
        assert_pixels_eq!(left, expected);

        // Deliberately starts one pixel out of bounds
        let down = draw_antialiased_line_segment(&image, (3, -1), (3, 3), color, interpolate);
        assert_pixels_eq!(down, rotate270(&expected));

        // Deliberately end one pixel out of bounds
        let up = draw_antialiased_line_segment(&image, (3, 3), (3, -1), color, interpolate);
        assert_pixels_eq!(up, rotate270(&expected));
    }

    #[test]
    fn test_draw_antialiased_line_segment_diagonal() {
        use crate::pixelops::interpolate;
        use image::imageops::rotate90;

        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            1, 1, 2, 1, 1;
            1, 2, 1, 1, 1;
            2, 1, 1, 1, 1;
            1, 1, 1, 1, 1);

        let color = Luma([2u8]);
        let up_right = draw_antialiased_line_segment(&image, (0, 3), (2, 1), color, interpolate);
        assert_pixels_eq!(up_right, expected);

        let down_left = draw_antialiased_line_segment(&image, (2, 1), (0, 3), color, interpolate);
        assert_pixels_eq!(down_left, expected);

        let up_left = draw_antialiased_line_segment(&image, (1, 0), (3, 2), color, interpolate);
        assert_pixels_eq!(up_left, rotate90(&expected));

        let down_right = draw_antialiased_line_segment(&image, (3, 2), (1, 0), color, interpolate);
        assert_pixels_eq!(down_right, rotate90(&expected));
    }

    #[test]
    fn test_draw_antialiased_line_segment_oct7_and_oct3() {
        use crate::pixelops::interpolate;

        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        // Gradient is 3/4
        let expected = gray_image!(
            1,  1,  1,  13, 50;
            1,  1,  25, 37,  1;
            1,  37, 25,  1,  1;
            50, 13, 1,   1,  1;
            1,  1,  1,   1,  1);

        let color = Luma([50u8]);
        let oct7 = draw_antialiased_line_segment(&image, (0, 3), (4, 0), color, interpolate);
        assert_pixels_eq!(oct7, expected);

        let oct3 = draw_antialiased_line_segment(&image, (4, 0), (0, 3), color, interpolate);
        assert_pixels_eq!(oct3, expected);
    }

    macro_rules! bench_antialiased_lines {
        ($name:ident, $start:expr, $end:expr) => {
            #[bench]
            fn $name(b: &mut test::Bencher) {
                use super::draw_antialiased_line_segment_mut;
                use crate::pixelops::interpolate;

                let mut image = GrayImage::new(500, 500);
                let color = Luma([50u8]);
                b.iter(|| {
                    draw_antialiased_line_segment_mut(&mut image, $start, $end, color, interpolate);
                    test::black_box(&image);
                });
            }
        };
    }

    bench_antialiased_lines!(
        bench_draw_antialiased_line_segment_horizontal,
        (10, 10),
        (450, 10)
    );

    bench_antialiased_lines!(
        bench_draw_antialiased_line_segment_vertical,
        (10, 10),
        (10, 450)
    );

    bench_antialiased_lines!(
        bench_draw_antialiased_line_segment_diagonal,
        (10, 10),
        (450, 450)
    );

    bench_antialiased_lines!(
        bench_draw_antialiased_line_segment_shallow,
        (10, 10),
        (450, 80)
    );

    #[test]
    fn test_draw_line_segment_horizontal_using_bresenham_line_pixel_iter_mut() {
        let image = GrayImage::from_pixel(5, 5, Luma([1u8]));

        let expected = gray_image!(
            1, 1, 1, 1, 1;
            4, 4, 4, 4, 4;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1;
            1, 1, 1, 1, 1);

        let mut right = image.clone();
        {
            let right_iter =
                BresenhamLinePixelIterMut::new(&mut right, (-3f32, 1f32), (6f32, 1f32));
            for p in right_iter {
                *p = Luma([4u8]);
            }
        }
        assert_pixels_eq!(right, expected);

        let mut left = image.clone();
        {
            let left_iter = BresenhamLinePixelIterMut::new(&mut left, (6f32, 1f32), (-3f32, 1f32));
            for p in left_iter {
                *p = Luma([4u8]);
            }
        }
        assert_pixels_eq!(left, expected);
    }
}
