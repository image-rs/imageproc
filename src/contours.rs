//! Functions for finding border contours within binary images.

use crate::point::Point;
use image::GrayImage;
use num::{cast, Num, NumCast};
use std::collections::VecDeque;

/// Whether a border of a foreground region borders an enclosing background region or a contained background region.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderType {
    /// A border between a foreground region and the backround region enclosing it.
    /// All points in the border lie within the foreground region.
    Outer,
    /// A border between a foreground region and a background region contained within it.
    /// All points in the border lie within the foreground region.
    Hole,
}

/// A border of an 8-connected foreground region.
#[derive(Debug)]
pub struct Contour<T> {
    /// The points in the border.
    pub points: Vec<Point<T>>,
    /// Whether this is an outer border or a hole border.
    pub border_type: BorderType,
    /// Calls to `find_contours` and `find_contours_with_threshold` return a `Vec` of all borders
    /// in an image. This field provides the index for the parent of the current border in that `Vec`.
    pub parent: Option<usize>,
}

impl<T> Contour<T> {
    /// Construct a contour.
    pub fn new(points: Vec<Point<T>>, border_type: BorderType, parent: Option<usize>) -> Self {
        Contour {
            points,
            border_type,
            parent,
        }
    }
}

/// Finds all borders of foreground regions in an image. All non-zero pixels are
/// treated as belonging to the foreground.
///
/// Based on the algorithm proposed by Suzuki and Abe: Topological Structural
/// Analysis of Digitized Binary Images by Border Following.
pub fn find_contours<T>(image: &GrayImage) -> Vec<Contour<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
    find_contours_with_threshold(image, 0)
}

/// Finds all borders of foreground regions in an image. All pixels with intensity strictly greater
/// than `threshold` are treated as belonging to the foreground.
///
/// Based on the algorithm proposed by Suzuki and Abe: Topological Structural
/// Analysis of Digitized Binary Images by Border Following.
pub fn find_contours_with_threshold<T>(image: &GrayImage, threshold: u8) -> Vec<Contour<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut image_values = vec![vec![0i32; height]; width];

    for y in 0..height {
        for x in 0..width {
            if image.get_pixel(x as u32, y as u32).0[0] > threshold {
                image_values[x][y] = 1;
            }
        }
    }

    let mut diffs = VecDeque::from(vec![
        Point::new(-1, 0),  // w
        Point::new(-1, -1), // nw
        Point::new(0, -1),  // n
        Point::new(1, -1),  // ne
        Point::new(1, 0),   // e
        Point::new(1, 1),   // se
        Point::new(0, 1),   // s
        Point::new(-1, 1),  // sw
    ]);

    let mut contours: Vec<Contour<T>> = Vec::new();
    let mut curr_border_num = 1;

    for y in 0..height {
        let mut parent_border_num = 1;

        for x in 0..width {
            if image_values[x][y] == 0 {
                continue;
            }

            if let Some((adj, border_type)) =
                if image_values[x][y] == 1 && x > 0 && image_values[x - 1][y] == 0 {
                    Some((Point::new(x - 1, y), BorderType::Outer))
                } else if image_values[x][y] > 0 && x + 1 < width && image_values[x + 1][y] == 0 {
                    if image_values[x][y] > 1 {
                        parent_border_num = image_values[x][y] as usize;
                    }
                    Some((Point::new(x + 1, y), BorderType::Hole))
                } else {
                    None
                }
            {
                curr_border_num += 1;

                let parent = if parent_border_num > 1 {
                    let parent_index = parent_border_num - 2;
                    let parent_contour = &contours[parent_index];
                    if (border_type == BorderType::Outer)
                        ^ (parent_contour.border_type == BorderType::Outer)
                    {
                        Some(parent_index)
                    } else {
                        parent_contour.parent
                    }
                } else {
                    None
                };

                let mut contour_points = Vec::new();
                let curr = Point::new(x, y);
                rotate_to_value(&mut diffs, adj.to_i32() - curr.to_i32());

                if let Some(pos1) = diffs.iter().find_map(|diff| {
                    get_position_if_non_zero_pixel(&image_values, curr.to_i32() + *diff)
                }) {
                    let mut pos2 = pos1;
                    let mut pos3 = curr;

                    loop {
                        contour_points
                            .push(Point::new(cast(pos3.x).unwrap(), cast(pos3.y).unwrap()));
                        rotate_to_value(&mut diffs, pos2.to_i32() - pos3.to_i32());
                        let pos4 = diffs
                            .iter()
                            .rev() // counter-clockwise
                            .find_map(|diff| {
                                get_position_if_non_zero_pixel(&image_values, pos3.to_i32() + *diff)
                            })
                            .unwrap();

                        let mut is_right_edge = false;
                        for diff in diffs.iter().rev() {
                            if *diff == (pos4.to_i32() - pos3.to_i32()) {
                                break;
                            }
                            if *diff == Point::new(1, 0) {
                                is_right_edge = true;
                                break;
                            }
                        }

                        if pos3.x + 1 == width || is_right_edge {
                            image_values[pos3.x][pos3.y] = -curr_border_num;
                        } else if image_values[pos3.x][pos3.y] == 1 {
                            image_values[pos3.x][pos3.y] = curr_border_num;
                        }

                        if pos4 == curr && pos3 == pos1 {
                            break;
                        }

                        pos2 = pos3;
                        pos3 = pos4;
                    }
                } else {
                    contour_points.push(Point::new(cast(x).unwrap(), cast(y).unwrap()));
                    image_values[x][y] = -curr_border_num;
                }
                contours.push(Contour::new(contour_points, border_type, parent));
            }

            if image_values[x][y] != 1 {
                parent_border_num = image_values[x][y].abs() as usize;
            }
        }
    }

    contours
}

fn rotate_to_value<T: Eq + Copy>(values: &mut VecDeque<T>, value: T) {
    let rotate_pos = values.iter().position(|x| *x == value).unwrap();
    values.rotate_left(rotate_pos);
}

fn get_position_if_non_zero_pixel(image: &[Vec<i32>], curr: Point<i32>) -> Option<Point<usize>> {
    let (width, height) = (image.len() as i32, image[0].len() as i32);
    let in_bounds = curr.x > -1 && curr.x < width && curr.y > -1 && curr.y < height;

    if in_bounds && image[curr.x as usize][curr.y as usize] != 0 {
        Some(Point::new(curr.x as usize, curr.y as usize))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;

    // Checks that a contour has the expected border type and parent, and
    // that it contains each of a given set of points.
    fn check_contour<T: Eq>(
        contour: &Contour<T>,
        expected_border_type: BorderType,
        expected_parent: Option<usize>,
        required_points: &[Point<T>],
    ) {
        for point in required_points {
            assert!(contour.points.contains(point));
        }
        assert_eq!(contour.border_type, expected_border_type);
        assert_eq!(contour.parent, expected_parent);
    }

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
        check_contour(
            &contours[0],
            BorderType::Outer,
            None,
            &[
                Point::new(20, 20),
                Point::new(280, 20),
                Point::new(280, 280),
                Point::new(20, 280),
            ],
        );

        // border 2
        check_contour(
            &contours[1],
            BorderType::Hole,
            Some(0),
            &[
                Point::new(39, 40),
                Point::new(261, 40),
                Point::new(261, 260),
                Point::new(39, 260),
            ],
        );

        // border 3
        check_contour(
            &contours[2],
            BorderType::Outer,
            Some(1),
            &[
                Point::new(60, 60),
                Point::new(240, 60),
                Point::new(240, 240),
                Point::new(60, 220),
            ],
        );

        // border 4
        check_contour(
            &contours[3],
            BorderType::Hole,
            Some(2),
            &[
                Point::new(79, 80),
                Point::new(221, 80),
                Point::new(221, 220),
                Point::new(79, 220),
            ],
        );

        // rectangle in the corner
        check_contour(
            &contours[4],
            BorderType::Outer,
            None,
            &[
                Point::new(290, 290),
                Point::new(299, 290),
                Point::new(299, 299),
                Point::new(290, 299),
            ],
        );
    }

    #[test]
    fn find_contours_basic_test() {
        use crate::definitions::HasWhite;
        use crate::drawing::draw_polygon_mut;
        use image::Luma;

        let mut image = GrayImage::new(15, 20);
        draw_polygon_mut(
            &mut image,
            &[Point::new(5, 5), Point::new(11, 5)],
            Luma::white(),
        );

        draw_polygon_mut(
            &mut image,
            &[Point::new(11, 5), Point::new(11, 9)],
            Luma::white(),
        );

        draw_polygon_mut(
            &mut image,
            &[Point::new(11, 9), Point::new(5, 9)],
            Luma::white(),
        );

        draw_polygon_mut(
            &mut image,
            &[Point::new(5, 5), Point::new(5, 9)],
            Luma::white(),
        );

        draw_polygon_mut(
            &mut image,
            &[Point::new(8, 5), Point::new(8, 9)],
            Luma::white(),
        );

        *image.get_pixel_mut(13, 6) = Luma::white();

        let contours = find_contours::<u32>(&image);
        assert_eq!(contours.len(), 4);

        check_contour(
            &contours[0],
            BorderType::Outer,
            None,
            &[
                Point::new(5, 5),
                Point::new(11, 5),
                Point::new(5, 9),
                Point::new(11, 9),
            ],
        );
        assert!(!contours[0].points.contains(&Point::new(13, 6)));

        check_contour(
            &contours[1],
            BorderType::Hole,
            Some(0),
            &[
                Point::new(5, 6),
                Point::new(8, 6),
                Point::new(6, 9),
                Point::new(8, 8),
            ],
        );
        assert!(!contours[1].points.contains(&Point::new(10, 5)));
        assert!(!contours[1].points.contains(&Point::new(10, 9)));
        assert!(!contours[1].points.contains(&Point::new(13, 6)));

        check_contour(
            &contours[2],
            BorderType::Hole,
            Some(0),
            &[
                Point::new(8, 6),
                Point::new(10, 5),
                Point::new(8, 8),
                Point::new(10, 9),
            ],
        );
        assert!(!contours[2].points.contains(&Point::new(6, 9)));
        assert!(!contours[2].points.contains(&Point::new(5, 6)));
        assert!(!contours[2].points.contains(&Point::new(13, 6)));

        assert_eq!(contours[3].border_type, BorderType::Outer);
        assert_eq!(contours[3].points, [Point::new(13, 6)]);
        assert_eq!(contours[3].parent, None);
    }

    #[test]
    fn get_contours_approx_points() {
        use crate::drawing::draw_polygon_mut;
        use image::{GrayImage, Luma};
        let mut image = GrayImage::from_pixel(300, 300, Luma([0]));
        let white = Luma([255]);

        let star = vec![
            Point::new(100, 20),
            Point::new(120, 35),
            Point::new(140, 30),
            Point::new(115, 45),
            Point::new(130, 60),
            Point::new(100, 50),
            Point::new(80, 55),
            Point::new(90, 40),
            Point::new(60, 25),
            Point::new(90, 35),
        ];
        draw_polygon_mut(&mut image, &star, white);
        let contours = find_contours::<u32>(&image);

        let c1_approx = crate::geometry::approximate_polygon_dp(
            &contours[0].points,
            crate::geometry::arc_length(&contours[0].points, true) * 0.01,
            true,
        );
        assert_eq!(
            c1_approx,
            vec![
                Point::new(100, 20),
                Point::new(90, 35),
                Point::new(60, 25),
                Point::new(90, 40),
                Point::new(80, 55),
                Point::new(101, 50),
                Point::new(130, 60),
                Point::new(115, 45),
                Point::new(140, 30),
                Point::new(120, 35)
            ]
        );
    }
}
