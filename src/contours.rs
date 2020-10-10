//! Functions for detecting contours of polygons in an image and approximating
//! polygon from set of points.

use crate::definitions::Point;
use image::GrayImage;
use num::{cast, Num, NumCast};
use std::collections::VecDeque;

/// Contour struct containing its points, is_outer flag to determine whether the
/// contour is an outer border or hole border, and the parent option (the index
/// of the parent contour in the contours vec).
#[derive(Debug)]
pub struct Contour<T: Num + NumCast + Copy + PartialEq + Eq> {
    /// All the points on the contour.
    pub points: Vec<Point<T>>,
    /// Flag to determine whether the contour is an outer border or hole border.
    pub is_outer: bool,
    /// the index of the parent contour in the contours vec, or None if the
    /// contour is the outermost contour in the image.
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
/// Analysis of Digitized Binary Images by Border Following.
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
    let mut curr_border_num = 1;
    let mut parent_border_num = 1;
    let mut pos2 = Point::new(0, 0);

    while (x, y) != last_pixel {
        if image_values[x][y] != 0 {
            skip_tracing = false;
            if image_values[x][y] == 1 && x > 0 && image_values[x - 1][y] == 0 {
                curr_border_num += 1;
                pos2 = Point::new(x - 1, y);
            } else if image_values[x][y] > 0 && x + 1 < width && image_values[x + 1][y] == 0 {
                curr_border_num += 1;
                pos2 = Point::new(x + 1, y);
                if image_values[x][y] > 1 {
                    parent_border_num = image_values[x][y] as usize;
                }
            } else {
                skip_tracing = true;
            }

            if !skip_tracing {
                let parent = if parent_border_num < 2 {
                    None
                } else {
                    Some(parent_border_num - 2)
                };
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
                            image_values[pos3.x][pos3.y] = -curr_border_num;
                        } else if image_values[pos3.x][pos3.y] == 1 {
                            image_values[pos3.x][pos3.y] = curr_border_num;
                        }
                        if pos4.x == x && pos4.y == y && pos3 == pos1 {
                            break;
                        }
                        pos2 = pos3;
                        pos3 = pos4;
                    }

                    contours.push(Contour::new(contour_points, is_outer, parent));
                } else {
                    image_values[x][y] = -curr_border_num;
                }
            }

            if image_values[x][y] != 1 {
                parent_border_num = image_values[x][y].abs() as usize;
            }
        }
        if x == last_pixel.0 {
            x = 0;
            y += 1;
            parent_border_num = 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contours_structured() {
        use crate::drawing::draw_polygon_mut;
        use image::Luma;

        let white = Luma([255u8]);
        let black = Luma([0u8]);

        let mut image = GrayImage::from_pixel(300, 300, black);
        // border 1 (outer)
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(20, 20),
                Point::new(280, 20),
                Point::new(280, 280),
                Point::new(20, 280),
            ],
            white,
        );
        // border 2 (hole)
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(40, 40),
                Point::new(260, 40),
                Point::new(260, 260),
                Point::new(40, 260),
            ],
            black,
        );
        // border 3 (outer)
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(60, 60),
                Point::new(240, 60),
                Point::new(240, 240),
                Point::new(60, 240),
            ],
            white,
        );
        // border 4 (hole)
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(80, 80),
                Point::new(220, 80),
                Point::new(220, 220),
                Point::new(80, 220),
            ],
            black,
        );
        // rectangle in the corner (outer)
        draw_polygon_mut(
            &mut image,
            &[
                Point::new(290, 290),
                Point::new(300, 290),
                Point::new(300, 300),
                Point::new(290, 300),
            ],
            white,
        );
        let contours = find_contours::<i32>(&image);

        assert_eq!(contours.len(), 5);
        // border 1
        assert!(contours[0].points.contains(&Point::new(20, 20)));
        assert!(contours[0].points.contains(&Point::new(280, 20)));
        assert!(contours[0].points.contains(&Point::new(280, 280)));
        assert!(contours[0].points.contains(&Point::new(20, 280)));
        assert!(contours[0].is_outer, true);
        assert_eq!(contours[0].parent, None);
        // border 2
        assert!(contours[1].points.contains(&Point::new(39, 40)));
        assert!(contours[1].points.contains(&Point::new(261, 40)));
        assert!(contours[1].points.contains(&Point::new(261, 260)));
        assert!(contours[1].points.contains(&Point::new(39, 260)));
        assert_eq!(contours[1].is_outer, false);
        assert_eq!(contours[1].parent, Some(0));
        // border 3
        assert!(contours[2].points.contains(&Point::new(60, 60)));
        assert!(contours[2].points.contains(&Point::new(240, 60)));
        assert!(contours[2].points.contains(&Point::new(240, 240)));
        assert!(contours[2].points.contains(&Point::new(60, 240)));
        assert_eq!(contours[2].is_outer, true);
        assert_eq!(contours[2].parent, Some(1));
        // border 4
        assert!(contours[3].points.contains(&Point::new(79, 80)));
        assert!(contours[3].points.contains(&Point::new(221, 80)));
        assert!(contours[3].points.contains(&Point::new(221, 220)));
        assert!(contours[3].points.contains(&Point::new(79, 220)));
        assert_eq!(contours[3].is_outer, false);
        assert_eq!(contours[3].parent, Some(2));
        // rectangle in the corner
        assert!(contours[4].points.contains(&Point::new(290, 290)));
        assert!(contours[4].points.contains(&Point::new(299, 290)));
        assert!(contours[4].points.contains(&Point::new(299, 299)));
        assert!(contours[4].points.contains(&Point::new(290, 299)));
        assert_eq!(contours[4].is_outer, true);
        assert_eq!(contours[4].parent, None);
    }
}
