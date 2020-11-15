//! Functions for detecting contours of polygons in an image and approximating
//! polygon from set of points.

use crate::point::{distance, Line, Point, Rotation};
use image::GrayImage;
use num::{cast, Num, NumCast};
use std::cmp::{Ord, Ordering};
use std::collections::VecDeque;
use std::f64::{self, consts::PI};

/// Contour struct containing its points, is_outer flag to determine whether the
/// contour is an outer border or hole border, and the parent option (the index
/// of the parent contour in the contours vec).
#[derive(Debug)]
pub struct Contour<T> {
    /// All the points on the contour.
    pub points: Vec<Point<T>>,
    /// Flag to determine whether the contour is an outer border or hole border.
    pub is_outer: bool,
    /// the index of the parent contour in the contours vec, or None if the
    /// contour is the outermost contour in the image.
    pub parent: Option<usize>,
}

impl<T> Contour<T> {
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
pub fn find_contours<T>(original_image: &GrayImage) -> Vec<Contour<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
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
pub fn find_contours_with_thresh<T>(original_image: &GrayImage, thresh: u8) -> Vec<Contour<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq,
{
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
    let mut skip_tracing;
    let mut curr_border_num = 1;
    let mut parent_border_num;
    let mut pos2 = Point::new(0, 0);

    for y in 0..height {
        parent_border_num = 1;

        for x in 0..width {
            if image_values[x][y] == 0 {
                continue;
            }

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
                rotate_to_value(&mut diffs, pos2.to_i32() - Point::new(x, y).to_i32());
                if let Some(pos1) = diffs.iter().find_map(|diff| {
                    get_position_if_non_zero_pixel(
                        &image_values,
                        x as i32 + diff.x,
                        y as i32 + diff.y,
                    )
                }) {
                    pos2 = pos1;
                    let mut pos3 = Point::new(x, y);
                    loop {
                        contour_points
                            .push(Point::new(cast(pos3.x).unwrap(), cast(pos3.y).unwrap()));
                        rotate_to_value(&mut diffs, pos2.to_i32() - pos3.to_i32());
                        let pos4 = diffs
                            .iter()
                            .rev() // counter-clockwise
                            .find_map(|diff| {
                                get_position_if_non_zero_pixel(
                                    &image_values,
                                    pos3.x as i32 + diff.x,
                                    pos3.y as i32 + diff.y,
                                )
                            })
                            .unwrap();

                        let mut is_right_edge = false;
                        let pos4_diff = pos4.to_i32() - pos3.to_i32();
                        for diff in diffs.iter().rev() {
                            if diff == &pos4_diff {
                                break;
                            }
                            if diff == &Point::new(1, 0) {
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
    }

    contours
}

fn rotate_to_value<T: Eq + Copy>(values: &mut VecDeque<T>, value: T) {
    let rotate_pos = values.iter().position(|x| *x == value).unwrap();
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
pub fn arc_length<T: Num + NumCast + Copy + PartialEq + Eq>(arc: &[Point<T>], closed: bool) -> f64 {
    if arc.len() < 2 {
        return 0.;
    }
    let mut length = arc
        .windows(2)
        .fold(0., |acc, pts| acc + distance(pts[0], pts[1]));
    if arc.len() > 2 && closed {
        length += distance(arc[0], arc[arc.len() - 1]);
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
    let line = Line::from_points(curve[0].to_f64(), curve[end].to_f64());
    for (i, point) in curve.iter().enumerate().skip(1) {
        let d = line.distance_from_point(point.to_f64());
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

/// Finds the minimal area rectangle that covers all of the points in the input
/// contour in the following order -> (TL, TR, BR, BL).
pub fn min_area_rect<T: Num + NumCast + Copy + PartialEq + Eq + Ord>(
    contour: &[Point<T>],
) -> [Point<T>; 4] {
    let hull = convex_hull(&contour);
    match hull.len() {
        0 => panic!("no points are defined"),
        1 => [hull[0]; 4],
        2 => [hull[0], hull[1], hull[1], hull[0]],
        _ => rotating_calipers(&hull),
    }
}

/// The implementation of the [rotating calipers] used for determining the
/// bounding rectangle with the smallest area.
///
/// [rotating calipers]: https://en.wikipedia.org/wiki/Rotating_calipers
fn rotating_calipers<T: Num + NumCast + Copy + PartialEq + Eq>(
    points: &[Point<T>],
) -> [Point<T>; 4] {
    let mut edge_angles: Vec<f64> = points
        .windows(2)
        .map(|e| {
            let edge = e[1].to_f64() - e[0].to_f64();
            ((edge.y.atan2(edge.x) + PI) % (PI / 2.)).abs()
        })
        .collect();

    edge_angles.dedup();

    let mut min_area = f64::MAX;
    let mut res = vec![Point::new(0., 0.); 4];
    for angle in edge_angles {
        let rotation = Rotation::new(angle);
        let rotated_points: Vec<Point<f64>> =
            points.iter().map(|p| p.to_f64().rotate(rotation)).collect();

        let (min_x, max_x, min_y, max_y) =
            rotated_points
                .iter()
                .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |acc, p| {
                    (
                        acc.0.min(p.x),
                        acc.1.max(p.x),
                        acc.2.min(p.y),
                        acc.3.max(p.y),
                    )
                });

        let area = (max_x - min_x) * (max_y - min_y);
        if area < min_area {
            min_area = area;
            res[0] = Point::new(max_x, min_y).invert_rotation(rotation);
            res[1] = Point::new(min_x, min_y).invert_rotation(rotation);
            res[2] = Point::new(min_x, max_y).invert_rotation(rotation);
            res[3] = Point::new(max_x, max_y).invert_rotation(rotation);
        }
    }

    res.sort_by(|a, b| {
        if a.x < b.x {
            Ordering::Less
        } else if a.x > b.x {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    let i1 = if res[1].y > res[0].y { 0 } else { 1 };
    let i2 = if res[3].y > res[2].y { 2 } else { 3 };
    let i3 = if res[3].y > res[2].y { 3 } else { 2 };
    let i4 = if res[1].y > res[0].y { 1 } else { 0 };

    [
        Point::new(
            cast(res[i1].x.floor()).unwrap(),
            cast(res[i1].y.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i2].x.ceil()).unwrap(),
            cast(res[i2].y.floor()).unwrap(),
        ),
        Point::new(
            cast(res[i3].x.ceil()).unwrap(),
            cast(res[i3].y.ceil()).unwrap(),
        ),
        Point::new(
            cast(res[i4].x.floor()).unwrap(),
            cast(res[i4].y.ceil()).unwrap(),
        ),
    ]
}

/// Finds points of the smallest convex polygon that contains all the contour points.
/// Based on the [Graham scan algorithm].
///
/// [Graham scan algorithm]: https://en.wikipedia.org/wiki/Graham_scan
fn convex_hull<T>(points_slice: &[Point<T>]) -> Vec<Point<T>>
where
    T: Num + NumCast + Copy + PartialEq + Eq + Ord,
{
    if points_slice.is_empty() {
        return Vec::new();
    }
    let mut points: Vec<Point<T>> = points_slice.to_vec();
    let mut start_point_pos = 0;
    let mut start_point = points[0];
    for (i, &point) in points.iter().enumerate().skip(1) {
        if point.y < start_point.y || point.y == start_point.y && point.x < start_point.x {
            start_point_pos = i;
            start_point = point;
        }
    }
    points.swap(0, start_point_pos);
    points.remove(0);
    points.sort_by(
        |a, b| match orientation(start_point.to_i32(), a.to_i32(), b.to_i32()) {
            Orientation::Collinear => {
                if distance(start_point, *a) < distance(start_point, *b) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            Orientation::Clockwise => Ordering::Greater,
            Orientation::CounterClockwise => Ordering::Less,
        },
    );

    let mut iter = points.iter().peekable();
    let mut remaining_points = Vec::with_capacity(points.len());
    while let Some(mut p) = iter.next() {
        while iter.peek().is_some()
            && orientation(
                start_point.to_i32(),
                p.to_i32(),
                iter.peek().unwrap().to_i32(),
            ) == Orientation::Collinear
        {
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
            && orientation(
                stack[stack.len() - 2].to_i32(),
                stack[stack.len() - 1].to_i32(),
                p.to_i32(),
            ) != Orientation::CounterClockwise
        {
            stack.pop();
        }
        stack.push(p);
    }
    stack
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Orientation {
    Collinear,
    Clockwise,
    CounterClockwise,
}

fn orientation(p: Point<i32>, q: Point<i32>, r: Point<i32>) -> Orientation {
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    match val.cmp(&0) {
        Ordering::Equal => Orientation::Collinear,
        Ordering::Greater => Orientation::Clockwise,
        Ordering::Less => Orientation::CounterClockwise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;
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
    fn convex_hull_points() {
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
    fn convex_hull_points_empty_vec() {
        let points = convex_hull::<i32>(&vec![]);
        assert_eq!(points, []);
    }

    #[test]
    fn convex_hull_points_with_negative_values() {
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
