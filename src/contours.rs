//! Functions for detecting contours of polygons in an image and approximating
//! polygon from set of points.

use crate::definitions::Point;
use image::GrayImage;
use num::{cast, Num, NumCast};
use std::collections::VecDeque;

/// Contour struct containing its points, is_outer flag to determine whether the
/// contour is an outer border or hole border, and the parent option (the index
/// of the parent contour in the contours vec)
#[derive(Debug)]
pub struct Contour<T: Num + NumCast + Copy + PartialEq + Eq> {
    /// All the points on the contour
    pub points: Vec<Point<T>>,
    /// Flag to determine whether the contour is an outer border or hole border
    pub is_outer: bool,
    /// the index of the parent contour in the contours vec, or None if the
    /// contour is the outermost contour in the image
    pub parent: Option<usize>,
}
impl<T: Num + NumCast + Copy + PartialEq + Eq> Contour<T> {
    /// Construct a contour.
    pub fn new(points: Vec<Point<T>>, is_outer: bool, parent: Option<usize>) -> Self {
        Contour {
            points,
            is_outer,
            parent,
        }
    }
}

/// Finds all the points on the contours on the provided image.
/// Handles all non-zero pixels as 1.
pub fn find_contours<T: Num + NumCast + Copy + PartialEq + Eq>(
    original_image: &GrayImage,
) -> Vec<Contour<T>> {
    find_contours_with_thresh(original_image, 0)
}

/// Finds all contours (contour - all the points on the edge of a polygon)
/// in the provided image. The algorithm works only with a binarized image,
/// therefore, the `thresh` param defines the value for which every pixel with
/// value higher then `thresh` will be considered as 1, and 0 otherwise.
///
/// Based on the algorithm proposed by Suzuki and Abe: Topological Structural
/// Analysis of Digitized Binary Images by Border Following
///
pub fn find_contours_with_thresh<T: Num + NumCast + Copy + PartialEq + Eq>(
    original_image: &GrayImage,
    thresh: u8,
) -> Vec<Contour<T>> {
    let width = original_image.width() as usize;
    let height = original_image.height() as usize;
    let mut image_values = vec![vec![0i32; height]; width];

    for y in 0..height {
        for x in 0..width {
            if original_image.get_pixel(x as u32, y as u32).0[0] > thresh {
                image_values[x][y] = 1;
            }
        }
    }
    let mut diffs = VecDeque::from(vec![
        (-1, 0),  // w
        (-1, -1), // nw
        (0, -1),  // n
        (1, -1),  // ne
        (1, 0),   // e
        (1, 1),   // se
        (0, 1),   // s
        (-1, 1),  // sw
    ]);
    let mut x = 0;
    let mut y = 0;
    let last_pixel = (width - 1, height - 1);

    let mut contours: Vec<Contour<T>> = Vec::new();
    let mut skip_tracing;
    let mut nbd = 1;
    let mut lnbd = 1;
    let mut pos2 = Point::new(0, 0);

    while (x, y) != last_pixel {
        if image_values[x][y] != 0 {
            skip_tracing = false;
            if image_values[x][y] == 1 && x > 0 && image_values[x - 1][y] == 0 {
                nbd += 1;
                pos2 = Point::new(x - 1, y);
            } else if image_values[x][y] > 0 && x + 1 < width && image_values[x + 1][y] == 0 {
                nbd += 1;
                pos2 = Point::new(x + 1, y);
                if image_values[x][y] > 1 {
                    lnbd = image_values[x][y] as usize;
                }
            } else {
                skip_tracing = true;
            }

            if !skip_tracing {
                let parent = if lnbd < 2 { None } else { Some(lnbd - 2) };
                let mut is_outer = true;
                if let Some(p_idx) = &parent {
                    is_outer = !contours[*p_idx].is_outer;
                }

                rotate_to_value(
                    &mut diffs,
                    (pos2.x as i32 - x as i32, pos2.y as i32 - y as i32),
                );
                if let Some(pos1) = diffs.iter().find_map(|(x_diff, y_diff)| {
                    get_position_if_non_zero_pixel(
                        &image_values,
                        x as i32 + *x_diff,
                        y as i32 + *y_diff,
                    )
                }) {
                    pos2 = pos1;
                    let mut pos3 = Point::new(x, y);
                    let mut contour_points = Vec::new();
                    loop {
                        contour_points
                            .push(Point::new(cast(pos3.x).unwrap(), cast(pos3.y).unwrap()));
                        rotate_to_value(
                            &mut diffs,
                            (pos2.x as i32 - pos3.x as i32, pos2.y as i32 - pos3.y as i32),
                        );
                        let pos4 = diffs
                            .iter()
                            .rev() // counter-clockwise
                            .find_map(|(x_diff, y_diff)| {
                                get_position_if_non_zero_pixel(
                                    &image_values,
                                    pos3.x as i32 + *x_diff,
                                    pos3.y as i32 + *y_diff,
                                )
                            })
                            .unwrap();

                        if pos3.x + 1 < width && image_values[pos3.x + 1][pos3.y] == 0 {
                            image_values[pos3.x][pos3.y] = -nbd;
                        } else if image_values[pos3.x][pos3.y] == 1 {
                            image_values[pos3.x][pos3.y] = nbd;
                        }
                        if pos4.x == x && pos4.y == y && pos3 == pos1 {
                            break;
                        }
                        pos2 = pos3;
                        pos3 = pos4;
                    }

                    contours.push(Contour::new(contour_points, is_outer, parent));
                } else {
                    image_values[x][y] = -nbd;
                }
            }

            if image_values[x][y] != 1 {
                lnbd = image_values[x][y].abs() as usize;
            }
        }
        if x == last_pixel.0 {
            x = 0;
            y += 1;
            lnbd = 1;
        } else {
            x += 1;
        }
    }

    contours
}

fn rotate_to_value(values: &mut VecDeque<(i32, i32)>, value: (i32, i32)) {
    let rotate_pos = values.iter().position(|&x| x == value).unwrap();
    values.rotate_left(rotate_pos);
}

fn get_position_if_non_zero_pixel(
    image: &[Vec<i32>],
    curr_x: i32,
    curr_y: i32,
) -> Option<Point<usize>> {
    if curr_x > -1
        && curr_x < image.len() as i32
        && curr_y > -1
        && curr_y < image[0].len() as i32
        && image[curr_x as usize][curr_y as usize] != 0
    {
        return Some(Point::new(curr_x as usize, curr_y as usize));
    }
    None
}
