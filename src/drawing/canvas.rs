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
///
/// See [`Blend`](struct.Blend.html) for another example implementation
/// of this trait - its implementation of `draw_pixel` alpha-blends
/// the input value with the pixel's current value.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::{Pixel, Rgba, RgbaImage};
/// use imageproc::drawing::{Canvas, Blend};
///
/// // A trivial function which draws on a Canvas
/// fn write_a_pixel<C: Canvas>(canvas: &mut C, c: C::Pixel) {
///     canvas.draw_pixel(0, 0, c);
/// }
///
/// // Background color
/// let solid_blue = Rgba([0u8, 0u8, 255u8, 255u8]);
///
/// // Drawing color
/// let translucent_red = Rgba([255u8, 0u8, 0u8, 127u8]);
///
/// // Blended combination of background and drawing colors
/// let mut alpha_blended = solid_blue;
/// alpha_blended.blend(&translucent_red);
///
/// // The implementation of Canvas for GenericImage overwrites existing pixels
/// let mut image = RgbaImage::from_pixel(1, 1, solid_blue);
/// write_a_pixel(&mut image, translucent_red);
/// assert_eq!(*image.get_pixel(0, 0), translucent_red);
///
/// // This behaviour can be customised by using a different Canvas type
/// let mut image = Blend(RgbaImage::from_pixel(1, 1, solid_blue));
/// write_a_pixel(&mut image, translucent_red);
/// assert_eq!(*image.0.get_pixel(0, 0), alpha_blended);
/// # }
/// ```
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
///
/// See the documentation for [`Canvas`](trait.Canvas.html)
/// for an example using this type.
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
