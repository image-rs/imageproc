//! Functions for detecting contours of polygons in an image and approximating
//! polygon from set of points.

use crate::definitions::Point;
use image::GrayImage;
use num::{cast, traits::bounds::Bounded, Num, NumCast};
use std::cmp::{Ord, Ordering};
use std::collections::VecDeque;
use std::f64::consts::PI;

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
            let is_outer;
            if image_values[x][y] == 1 && x > 0 && image_values[x - 1][y] == 0 {
                curr_border_num += 1;
                pos2 = Point::new(x - 1, y);
                is_outer = true;
            } else if image_values[x][y] > 0 && x + 1 < width && image_values[x + 1][y] == 0 {
                is_outer = false;
                curr_border_num += 1;
                pos2 = Point::new(x + 1, y);
                if image_values[x][y] > 1 {
                    parent_border_num = image_values[x][y] as usize;
                }
            } else {
                is_outer = false;
                skip_tracing = true;
            }

            if !skip_tracing {
                let mut parent = None;
                if parent_border_num > 1 {
                    let p_idx = parent_border_num - 2;
                    if is_outer ^ contours[p_idx].is_outer {
                        parent = Some(p_idx);
                    } else {
                        parent = contours[p_idx].parent;
                    }
                };

                let mut contour_points = Vec::new();
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

                        let mut is_right_edge = false;
                        let pos4_diff =
                            (pos4.x as i32 - pos3.x as i32, pos4.y as i32 - pos3.y as i32);
                        for diff in diffs.iter().rev() {
                            if diff == &pos4_diff {
                                break;
                            }
                            if diff == &(1, 0) {
                                is_right_edge = true;
                                break;
                            }
                        }

                        if pos3.x + 1 == width || is_right_edge {
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
                } else {
                    contour_points.push(Point::new(cast(x).unwrap(), cast(y).unwrap()));
                    image_values[x][y] = -curr_border_num;
                }
                contours.push(Contour::new(contour_points, is_outer, parent));
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

/// Returns the length of the arc constructed with the provided points in
/// incremental order. When the `closed` param is set to `true`, the distance
/// between the last and the first point is included in the total length.
///
pub fn arc_length<T: Num + NumCast + Copy + PartialEq + Eq>(arc: &[Point<T>], closed: bool) -> f64 {
    if arc.len() < 2 {
        return 0.;
    }
    let mut length = arc
        .windows(2)
        .fold(0., |acc, pts| acc + get_distance(&pts[0], &pts[1]));
    if arc.len() > 2 && closed {
        length += get_distance(&arc[0], &arc[arc.len() - 1]);
    }
    length
}

/// Fits the polygon curve to a similar curve with fewer points.
/// The input parameters include an ordered array of points and an distance
/// dimension `epsilon` > 0. Based on the [Douglas–Peucker algorithm].
///
/// [Douglas–Peucker algorithm]: https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
pub fn approx_poly_dp<T: Num + NumCast + Copy + PartialEq + Eq>(
    curve: &[Point<T>],
    epsilon: f64,
    closed: bool,
) -> Vec<Point<T>> {
    if epsilon <= 0. {
        panic!("epsilon must be greater than 0");
    }
    // Find the point with the maximum distance
    let mut dmax = 0.;
    let mut index = 0;
    let end = curve.len() - 1;
    let line_args = line_params(&curve[0], &curve[end]);
    for (i, point) in curve.iter().enumerate().skip(1) {
        let d = perpendicular_distance(line_args, point);
        if d > dmax {
            index = i;
            dmax = d;
        }
    }

    let mut res = Vec::new();

    // If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon {
        // Recursive call
        let mut partial1 = approx_poly_dp(&curve[0..=index], epsilon, false);
        let mut partial2 = approx_poly_dp(&curve[index..=end], epsilon, false);

        // Build the result list
        partial1.pop();
        res.append(&mut partial1);
        res.append(&mut partial2);
    } else {
        res.push(curve[0]);
        res.push(curve[end]);
    }

    if closed {
        res.pop();
    }

    res
}

/// Returns the parameters of the [line equation] (Ax + By + C = 0) that passes through the
/// given points p1 and p2.
///
/// [line equation]: https://en.wikipedia.org/wiki/Linear_equation#Two-point_form
fn line_params<T: Num + NumCast + Copy + PartialEq + Eq>(
    p1: &Point<T>,
    p2: &Point<T>,
) -> (f64, f64, f64) {
    let a = p1.y.to_f64().unwrap() - p2.y.to_f64().unwrap();
    let b = p2.x.to_f64().unwrap() - p1.x.to_f64().unwrap();
    let c = (p1.x * p2.y).to_f64().unwrap() - (p2.x * p1.y).to_f64().unwrap();

    (a, b, c)
}

#[allow(clippy::many_single_char_names)]
fn perpendicular_distance<T: Num + NumCast + Copy + PartialEq + Eq>(
    line_args: (f64, f64, f64),
    point: &Point<T>,
) -> f64 {
    let (a, b, c) = line_args;

    (a * point.x.to_f64().unwrap() + b * point.y.to_f64().unwrap() + c).abs()
        / (a.powf(2.) + b.powf(2.)).sqrt()
}

/// Finds the minimal area rectangle that covers all of the points in the input
/// contour in the following order -> (TL, TR, BR, BL).
///
pub fn min_area_rect<T: Num + NumCast + Copy + PartialEq + Eq + Ord + Bounded>(
    contour: &[Point<T>],
) -> Vec<Point<T>> {
    let hull = convex_hull(&contour);
    match hull.len() {
        0 => panic!("no points are defined"),
        1 => vec![hull[0]; 4],
        2 => vec![hull[0], hull[1], hull[1], hull[0]],
        _ => rotating_calipers(&hull),
    }
}

/// The implementation of the [rotating calipers] used for determining the
/// bounding rectangle with the smallest area.
///
/// [rotating calipers]: https://en.wikipedia.org/wiki/Rotating_calipers
fn rotating_calipers<T: Num + NumCast + Copy + PartialEq + Eq>(
    points: &[Point<T>],
) -> Vec<Point<T>> {
    let n = points.len();
    let edges: Vec<(f64, f64)> = (0..n - 1)
        .map(|i| {
            let next = i + 1;
            (
                points[next].x.to_f64().unwrap() - points[i].x.to_f64().unwrap(),
                points[next].y.to_f64().unwrap() - points[i].y.to_f64().unwrap(),
            )
        })
        .collect();

    let mut edge_angles: Vec<f64> = edges
        .iter()
        .map(|e| ((e.1.atan2(e.0) + PI) % (PI / 2.)).abs())
        .collect();
    edge_angles.dedup();

    let mut min_area = std::f64::MAX;
    let mut res = vec![(0., 0.); 4];
    for angle in edge_angles {
        let r = [[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];

        let rotated_points: Vec<(f64, f64)> = points
            .iter()
            .map(|p| {
                (
                    p.x.to_f64().unwrap() * r[0][0] + p.y.to_f64().unwrap() * r[1][0],
                    p.x.to_f64().unwrap() * r[0][1] + p.y.to_f64().unwrap() * r[1][1],
                )
            })
            .collect();
        let (min_x, max_x, min_y, max_y) = rotated_points.iter().fold(
            (std::f64::MAX, std::f64::MIN, std::f64::MAX, std::f64::MIN),
            |acc, p| {
                (
                    acc.0.min(p.0),
                    acc.1.max(p.0),
                    acc.2.min(p.1),
                    acc.3.max(p.1),
                )
            },
        );
        let width = max_x - min_x;
        let height = max_y - min_y;
        let area = width * height;
        if area < min_area {
            min_area = area;

            res[0] = (
                max_x * r[0][0] + min_y * r[0][1],
                max_x * r[1][0] + min_y * r[1][1],
            );
            res[1] = (
                min_x * r[0][0] + min_y * r[0][1],
                min_x * r[1][0] + min_y * r[1][1],
            );
            res[2] = (
                min_x * r[0][0] + max_y * r[0][1],
                min_x * r[1][0] + max_y * r[1][1],
            );
            res[3] = (
                max_x * r[0][0] + max_y * r[0][1],
                max_x * r[1][0] + max_y * r[1][1],
            );
        }
    }

    res.sort_by(|a, b| {
        if a.0 < b.0 {
            Ordering::Less
        } else if a.0 > b.0 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    let i1 = if res[1].1 > res[0].1 { 0 } else { 1 };
    let i2 = if res[3].1 > res[2].1 { 2 } else { 3 };
    let i3 = if res[3].1 > res[2].1 { 3 } else { 2 };
    let i4 = if res[1].1 > res[0].1 { 1 } else { 0 };
    vec![
        Point::new(
            cast(res[i1].0.floor()).unwrap(),
            cast(res[i1].1.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i2].0.ceil()).unwrap(),
            cast(res[i2].1.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i3].0.ceil()).unwrap(),
            cast(res[i3].1.ceil()).unwrap(),
        ),
        Point::new(
            cast(res[i4].0.floor()).unwrap(),
            cast(res[i4].1.ceil()).unwrap(),
        ),
    ]
}

/// Finds points of the smallest convex polygon that contains all the contour points.
/// Based on the [Graham scan algorithm].
///
/// [Graham scan algorithm]: https://en.wikipedia.org/wiki/Graham_scan
fn convex_hull<T: Num + NumCast + Copy + PartialEq + Eq + Ord + Bounded>(
    points_slice: &[Point<T>],
) -> Vec<Point<T>> {
    if points_slice.is_empty() {
        return Vec::new();
    }
    let mut points: Vec<Point<T>> = points_slice.to_vec();
    let (start_point_pos, start_point) = points.iter().enumerate().fold(
        (usize::MAX, Point::new(T::max_value(), T::max_value())),
        |(pos, acc_point), (i, &point)| {
            if point.y < acc_point.y || point.y == acc_point.y && point.x < acc_point.x {
                return (i, point);
            }
            (pos, acc_point)
        },
    );
    points.swap(0, start_point_pos);
    points.remove(0);
    points.sort_by(|a, b| {
        let orientation = get_orientation(&start_point, a, b);
        if orientation == 0 {
            if get_distance(&start_point, a) < get_distance(&start_point, b) {
                return Ordering::Less;
            }
            return Ordering::Greater;
        }
        if orientation == 2 {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    });

    let mut iter = points.iter().peekable();
    let mut remaining_points = Vec::with_capacity(points.len());
    while let Some(mut p) = iter.next() {
        while iter.peek().is_some() && get_orientation(&start_point, p, iter.peek().unwrap()) == 0 {
            p = iter.next().unwrap();
        }
        remaining_points.push(p);
    }

    let mut stack: Vec<Point<T>> = vec![Point::new(
        cast(start_point.x).unwrap(),
        cast(start_point.y).unwrap(),
    )];

    for p in points {
        while stack.len() > 1
            && get_orientation(&stack[stack.len() - 2], &stack[stack.len() - 1], &p) != 2
        {
            stack.pop();
        }
        stack.push(p);
    }
    stack
}

fn get_orientation<T: Num + NumCast + Copy + PartialEq + Eq>(
    p: &Point<T>,
    q: &Point<T>,
    r: &Point<T>,
) -> u8 {
    let val = (q.y.to_i32().unwrap() - p.y.to_i32().unwrap())
        * (r.x.to_i32().unwrap() - q.x.to_i32().unwrap())
        - (q.x.to_i32().unwrap() - p.x.to_i32().unwrap())
            * (r.y.to_i32().unwrap() - q.y.to_i32().unwrap());
    match val.cmp(&0) {
        Ordering::Equal => 0,   // colinear
        Ordering::Greater => 1, // clockwise (right)
        Ordering::Less => 2,    // counter-clockwise (left)
    }
}

/// Calculates the distance between 2 points.
///
pub fn get_distance<T: Num + NumCast + Copy + PartialEq + Eq>(p1: &Point<T>, p2: &Point<T>) -> f64 {
    ((p1.x.to_f64().unwrap() - p2.x.to_f64().unwrap()).powf(2.)
        + (p1.y.to_f64().unwrap() - p2.y.to_f64().unwrap()).powf(2.))
    .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::Point;
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
        assert_eq!(contours[0].is_outer, true);
        assert!(contours[0].points.contains(&Point::new(5, 5)));
        assert!(contours[0].points.contains(&Point::new(11, 5)));
        assert!(contours[0].points.contains(&Point::new(5, 9)));
        assert!(contours[0].points.contains(&Point::new(11, 9)));
        assert!(!contours[0].points.contains(&Point::new(13, 6)));
        assert_eq!(contours[0].parent, None);

        assert_eq!(contours[1].is_outer, false);
        assert!(contours[1].points.contains(&Point::new(5, 6)));
        assert!(contours[1].points.contains(&Point::new(8, 6)));
        assert!(!contours[1].points.contains(&Point::new(10, 5)));
        assert!(contours[1].points.contains(&Point::new(6, 9)));
        assert!(contours[1].points.contains(&Point::new(8, 8)));
        assert!(!contours[1].points.contains(&Point::new(10, 9)));
        assert!(!contours[1].points.contains(&Point::new(13, 6)));
        assert_eq!(contours[1].parent, Some(0));

        assert_eq!(contours[2].is_outer, false);
        assert!(!contours[2].points.contains(&Point::new(5, 6)));
        assert!(contours[2].points.contains(&Point::new(8, 6)));
        assert!(contours[2].points.contains(&Point::new(10, 5)));
        assert!(!contours[2].points.contains(&Point::new(6, 9)));
        assert!(contours[2].points.contains(&Point::new(8, 8)));
        assert!(contours[2].points.contains(&Point::new(10, 9)));
        assert!(!contours[2].points.contains(&Point::new(13, 6)));
        assert_eq!(contours[2].parent, Some(0));

        assert_eq!(contours[3].is_outer, true);
        assert_eq!(contours[3].points, [Point::new(13, 6)]);
        assert_eq!(contours[3].parent, None);
        assert_eq!(contours.len(), 4);
    }

    #[test]
    fn line_params_test() {
        let p1 = Point::new(5, 7);
        let p2 = Point::new(10, 3);
        assert_eq!(line_params(&p1, &p2), (4., 5., -55.));
    }

    #[test]
    fn perpendicular_distance_test() {
        let line_args = (8., 7., 5.);
        let point = Point::new(2, 3);
        assert!(perpendicular_distance(line_args, &point) - 3.9510276472 < 1e-10);
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
        let c1_approx = approx_poly_dp(
            &contours[0].points,
            arc_length(&contours[0].points, true) * 0.01,
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

    #[test]
    fn get_convex_hull_points() {
        let star = vec![
            Point::new(100, 20),
            Point::new(90, 35),
            Point::new(60, 25),
            Point::new(90, 40),
            Point::new(80, 55),
            Point::new(101, 50),
            Point::new(130, 60),
            Point::new(115, 45),
            Point::new(140, 30),
            Point::new(120, 35),
        ];
        let points = convex_hull(&star);
        assert_eq!(
            points,
            [
                Point::new(100, 20),
                Point::new(140, 30),
                Point::new(130, 60),
                Point::new(80, 55),
                Point::new(60, 25)
            ]
        );
    }

    #[test]
    fn get_convex_hull_points_empty_vec() {
        let points = convex_hull::<i32>(&vec![]);
        assert_eq!(points, []);
    }

    #[test]
    fn get_convex_hull_points_with_negative_values() {
        let star = vec![
            Point::new(100, -20),
            Point::new(90, 5),
            Point::new(60, -15),
            Point::new(90, 0),
            Point::new(80, 15),
            Point::new(101, 10),
            Point::new(130, 20),
            Point::new(115, 5),
            Point::new(140, -10),
            Point::new(120, -5),
        ];
        let points = convex_hull(&star);
        assert_eq!(
            points,
            [
                Point::new(100, -20),
                Point::new(140, -10),
                Point::new(130, 20),
                Point::new(80, 15),
                Point::new(60, -15)
            ]
        );
    }

    #[test]
    fn min_area_test() {
        assert_eq!(
            min_area_rect(&[
                Point::new(100, 20),
                Point::new(140, 30),
                Point::new(130, 60),
                Point::new(80, 55),
                Point::new(60, 25)
            ]),
            [
                Point::new(60, 16),
                Point::new(141, 24),
                Point::new(137, 61),
                Point::new(57, 53)
            ]
        )
    }
}
