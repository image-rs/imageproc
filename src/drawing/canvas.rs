use image::{
    GenericImage,
    GenericImageView,
    Pixel
};

/// A surface for drawing on - many drawing functions in this
/// library are generic over a `Canvas` to allow the user to
/// configure e.g. whether to use blending.
///
/// All instances of `GenericImage` implement `Canvas`, with
/// the behaviour of `draw_pixel` being equivalent to calling
/// `set_pixel` with the same arguments.
pub trait Canvas {
    /// The type of `Pixel` that can be drawn on this canvas.
    type Pixel: Pixel;

    /// The width and height of this canvas.
    fn dimensions(&self) -> (u32, u32);

    /// The width of this canvas.
    fn width(&self) -> u32 {
        self.dimensions().0
    }

    /// The height of this canvas.
    fn height(&self) -> u32 {
        self.dimensions().1
    }

    /// Draw a pixel at the given coordinates. `x` and `y`
    /// should be within `dimensions` - if not then panicking
    /// is a valid implementation behaviour.
    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel);
}

impl <I> Canvas for I
where
    I: GenericImage
{
    type Pixel = I::Pixel;

    fn dimensions(&self) -> (u32, u32) {
        <I as GenericImageView>::dimensions(self)
    }

    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel) {
        self.put_pixel(x, y, color)
    }
}

/// A canvas that blends pixels when drawing.
pub struct Blend<I>(pub I);

impl<I: GenericImage> Canvas for Blend<I> {
    type Pixel = I::Pixel;

    fn dimensions(&self) -> (u32, u32) {
        self.0.dimensions()
    }

    fn draw_pixel(&mut self, x: u32, y: u32, color: Self::Pixel) {
        self.0.get_pixel_mut(x, y).blend(&color)
    }
}
