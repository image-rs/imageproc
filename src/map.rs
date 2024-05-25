//! Functions for mapping pixels and subpixels of images.

use image::{Luma, LumaA, Pixel, Primitive, Rgb, Rgba};

use crate::definitions::Image;

/// The type obtained by replacing the channel type of a given `Pixel` type.
/// The output type must have the same name of channels as the input type, or
/// several algorithms will produce incorrect results or panic.
pub trait WithChannel<C: Primitive>: Pixel {
    /// The new pixel type.
    type Pixel: Pixel<Subpixel = C>;
}

/// Alias to make uses of `WithChannel` less syntactically noisy.
pub type ChannelMap<Pix, Sub> = <Pix as WithChannel<Sub>>::Pixel;

impl<T, U> WithChannel<U> for Rgb<T>
where
    Rgb<T>: Pixel<Subpixel = T>,
    Rgb<U>: Pixel<Subpixel = U>,
    T: Primitive,
    U: Primitive,
{
    type Pixel = Rgb<U>;
}

impl<T, U> WithChannel<U> for Rgba<T>
where
    Rgba<T>: Pixel<Subpixel = T>,
    Rgba<U>: Pixel<Subpixel = U>,
    T: Primitive,
    U: Primitive,
{
    type Pixel = Rgba<U>;
}

impl<T, U> WithChannel<U> for Luma<T>
where
    T: Primitive,
    U: Primitive,
{
    type Pixel = Luma<U>;
}

impl<T, U> WithChannel<U> for LumaA<T>
where
    T: Primitive,
    U: Primitive,
{
    type Pixel = LumaA<U>;
}

/// Applies `f` to each subpixel of the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::map::map_subpixels;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let scaled = gray_image!(type: i16,
///     -2, -4;
///     -6, -8);
///
/// assert_pixels_eq!(
///     map_subpixels(&image, |x| -2 * (x as i16)),
///     scaled);
/// # }
/// ```
pub fn map_subpixels<P, F, S>(image: &Image<P>, f: F) -> Image<ChannelMap<P, S>>
where
    P: WithChannel<S>,
    S: Primitive,
    F: Fn(P::Subpixel) -> S,
{
    Image::from_vec(
        image.width(),
        image.height(),
        image.iter().map(|subpixel| f(*subpixel)).collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[doc=generate_mut_doc_comment!("map_subpixels")]
pub fn map_subpixels_mut<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel,
    F: Fn(P::Subpixel) -> P::Subpixel,
{
    image
        .iter_mut()
        .for_each(|subpixel| *subpixel = f(*subpixel));
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_subpixels")]
pub fn map_subpixels_parallel<P, F, S>(image: &Image<P>, f: F) -> Image<ChannelMap<P, S>>
where
    P: WithChannel<S>,
    P::Subpixel: Sync,
    S: Primitive + Send,
    F: Fn(P::Subpixel) -> S + Sync,
{
    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::ParallelIterator;

    Image::from_vec(
        image.width(),
        image.height(),
        image.par_iter().map(|subpixel| f(*subpixel)).collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_subpixels_mut")]
pub fn map_subpixels_mut_parallel<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel,
    P::Subpixel: Send,
    F: Fn(P::Subpixel) -> P::Subpixel + Sync,
{
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;

    image
        .par_iter_mut()
        .for_each(|subpixel| *subpixel = f(*subpixel));
}

/// Applies `f` to each pixel of the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::map::map_pixels;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let rgb = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// assert_pixels_eq!(
///     map_pixels(&image, |p| { Rgb([p[0], (2 * p[0]), (3 * p[0])]) }),
///     rgb);
/// # }
/// ```
pub fn map_pixels<P, Q, F>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel,
    Q: Pixel,
    F: Fn(P) -> Q,
{
    Image::from_vec(
        image.width(),
        image.height(),
        image
            .pixels()
            //optimisation: remove allocation if Pixel ever gets compile-time size information
            .flat_map(|pixel| f(*pixel).channels().to_vec())
            .collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[doc=generate_mut_doc_comment!("map_pixels")]
pub fn map_pixels_mut<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel,
    F: Fn(P) -> P,
{
    image.pixels_mut().for_each(|pixel| *pixel = f(*pixel))
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_pixels")]
pub fn map_pixels_parallel<P, Q, F>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel + Sync,
    P::Subpixel: Sync,
    Q: Pixel,
    Q::Subpixel: Send,
    F: Fn(P) -> Q + Sync,
{
    use rayon::iter::ParallelIterator;

    Image::from_vec(
        image.width(),
        image.height(),
        image
            .par_pixels()
            //optimisation: remove allocation if Pixel ever gets compile-time size information
            .flat_map(|pixel| f(*pixel).channels().to_vec())
            .collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_pixels_mut")]
pub fn map_pixels_mut_parallel<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel + Sync + Send,
    P::Subpixel: Sync + Send,
    F: Fn(P) -> P + Sync,
{
    use rayon::iter::ParallelIterator;

    image.par_pixels_mut().for_each(|pixel| *pixel = f(*pixel));
}

/// Applies `f` to each enumerated pixel of the input image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Rgb;
/// use imageproc::map::map_enumerated_pixels;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let rgb = rgb_image!(
///     [1, 0, 0], [2, 1, 0];
///     [3, 0, 1], [4, 1, 1]);
///
/// assert_pixels_eq!(
///     map_enumerated_pixels(&image, |x, y, p| {
///         Rgb([p[0], x as u8, y as u8])
///     }),
///     rgb);
/// # }
/// ```
pub fn map_enumerated_pixels<P, Q, F>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel,
    Q: Pixel,
    F: Fn(u32, u32, P) -> Q,
{
    Image::from_vec(
        image.width(),
        image.height(),
        image
            .enumerate_pixels()
            //optimisation: remove allocation if Pixel ever gets compile-time size information
            .flat_map(|(x, y, pixel)| f(x, y, *pixel).channels().to_vec())
            .collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[doc=generate_mut_doc_comment!("map_enumerated_pixels")]
pub fn map_enumerated_pixels_mut<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel,
    F: Fn(u32, u32, P) -> P,
{
    image
        .enumerate_pixels_mut()
        .for_each(|(x, y, pixel)| *pixel = f(x, y, *pixel))
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_enumerated_pixels")]
pub fn map_enumerated_pixels_parallel<P, Q, F>(image: &Image<P>, f: F) -> Image<Q>
where
    P: Pixel + Sync,
    P::Subpixel: Sync,
    Q: Pixel,
    Q::Subpixel: Send,
    F: Fn(u32, u32, P) -> Q + Sync,
{
    use rayon::iter::ParallelIterator;

    Image::from_vec(
        image.width(),
        image.height(),
        image
            .par_enumerate_pixels()
            //optimisation: remove allocation if Pixel ever gets compile-time size information
            .flat_map(|(x, y, pixel)| f(x, y, *pixel).channels().to_vec())
            .collect(),
    )
    .expect("of course the length is good, it's just a map")
}
#[cfg(feature = "rayon")]
#[doc = generate_parallel_doc_comment!("map_enumerated_pixels_mut")]
pub fn map_enumerated_pixels_mut_parallel<P, F>(image: &mut Image<P>, f: F)
where
    P: Pixel + Sync + Send,
    P::Subpixel: Sync + Send,
    F: Fn(u32, u32, P) -> P + Sync,
{
    use rayon::iter::ParallelIterator;

    image
        .par_enumerate_pixels_mut()
        .for_each(|(x, y, pixel)| *pixel = f(x, y, *pixel));
}

/// Applies `f` to each pixel of both input images.
///
/// # Panics
///
/// Panics if `image1` and `image2` do not have the same dimensions.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::map_pixels2;
///
/// let image1 = gray_image!(
///     1, 2,
///     3, 4
/// );
///
/// let image2 = gray_image!(
///     10, 20,
///     30, 40
/// );
///
/// let sum = gray_image!(
///     11, 22,
///     33, 44
/// );
///
/// assert_pixels_eq!(
///     map_pixels2(&image1, &image2, |p, q| Luma([p[0] + q[0]])),
///     sum
/// );
/// # }
/// ```
pub fn map_pixels2<P, Q, R, F>(image1: &Image<P>, image2: &Image<Q>, f: F) -> Image<R>
where
    P: Pixel,
    Q: Pixel,
    R: Pixel,
    F: Fn(P, Q) -> R,
{
    Image::from_vec(
        image1.width(),
        image2.height(),
        image1
            .pixels()
            .zip(image2.pixels())
            //optimisation: remove allocation if Pixel ever gets compile-time size information
            .flat_map(|(pixel1, pixel2)| f(*pixel1, *pixel2).channels().to_vec())
            .collect(),
    )
    .expect("of course the length is good, it's just a map")
}

/// Creates a grayscale image by extracting the red channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::red_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     1, 2;
///     3, 4);
///
/// let actual = red_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn into_red_channel<C>(image: &Image<Rgb<C>>) -> Image<Luma<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Luma([p[0]]))
}

/// Creates an RGB image by embedding a grayscale image in its red channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_red_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [1, 0, 0], [2, 0, 0];
///     [3, 0, 0], [4, 0, 0]);
///
/// let actual = as_red_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn from_red_channel<C>(image: &Image<Luma<C>>) -> Image<Rgb<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Rgb([p.0[0], C::zero(), C::zero()]))
}

/// Creates a grayscale image by extracting the green channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::green_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     2, 4;
///     6, 8);
///
/// let actual = green_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn into_green_channel<C>(image: &Image<Rgb<C>>) -> Image<Luma<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Luma([p[1]]))
}

/// Creates an RGB image by embedding a grayscale image in its green channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_green_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [0, 1, 0], [0, 2, 0];
///     [0, 3, 0], [0, 4, 0]);
///
/// let actual = as_green_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn from_green_channel<C>(image: &Image<Luma<C>>) -> Image<Rgb<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Rgb([C::zero(), p.0[0], C::zero()]))
}

/// Creates a grayscale image by extracting the blue channel of an RGB image.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::blue_channel;
///
/// let image = rgb_image!(
///     [1, 2, 3], [2, 4, 6];
///     [3, 6, 9], [4, 8, 12]);
///
/// let expected = gray_image!(
///     3, 6;
///     9, 12);
///
/// let actual = blue_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn into_blue_channel<C>(image: &Image<Rgb<C>>) -> Image<Luma<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Luma([p[2]]))
}

/// Creates an RGB image by embedding a grayscale image in its blue channel.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use image::Luma;
/// use imageproc::map::as_blue_channel;
///
/// let image = gray_image!(
///     1, 2;
///     3, 4);
///
/// let expected = rgb_image!(
///     [0, 0, 1], [0, 0, 2];
///     [0, 0, 3], [0, 0, 4]);
///
/// let actual = as_blue_channel(&image);
/// assert_pixels_eq!(actual, expected);
/// # }
/// ```
pub fn from_blue_channel<C>(image: &Image<Luma<C>>) -> Image<Rgb<C>>
where
    Rgb<C>: Pixel<Subpixel = C>,
    C: Primitive,
{
    map_pixels(image, |p| Rgb([C::zero(), C::zero(), p.0[0]]))
}
