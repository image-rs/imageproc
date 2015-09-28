
use image::{GenericImage,Luma};

fn cumulative_histogram<I: GenericImage<Pixel=Luma<u8>> + 'static>
    (image: &I) -> [i32;256] {

    let mut hist = [0i32;256];

    for pix in image.pixels() {
        hist[pix.2[0] as usize] += 1;
    }

    for i in 1..hist.len() {
        hist[i] += hist[i - 1];
    }

    hist
}

#[cfg(test)]
mod test {

    use super::{cumulative_histogram};
    use image::{GrayImage,ImageBuffer};

    #[test]
    fn test_cumulative_histogram() {
        let image: GrayImage = ImageBuffer::from_raw(5, 1, vec![
            1u8, 2u8, 3u8, 2u8, 1u8]).unwrap();

        let hist = cumulative_histogram(&image);

        assert_eq!(hist[0], 0);
        assert_eq!(hist[1], 2);
        assert_eq!(hist[2], 4);
        assert_eq!(hist[3], 5);
        assert!(hist.iter().skip(4).all(|x| *x == 5));
    }
}
