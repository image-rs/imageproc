//! Functions and a struct for finding and computing ORB descriptors as
//! described in [Rublee, et. al. (2012)][rublee].
//!
//! [rublee]: http://www.gwylab.com/download/ORB_2012.pdf

use image::{GrayImage, Luma};

use crate::{
    corners::{oriented_fast, Corner, OrientedFastCorner},
    integral_image::integral_image,
    point::Point,
};

use super::{
    brief::{brief_impl, BriefDescriptor, TestPair},
    constants::{BRIEF_PATCH_RADIUS, UNROTATED_BRIEF_TEST_PAIRS},
    BinaryDescriptor,
};

/// Oriented FAST and Rotated BRIEF descriptor as described in [Rublee, et. al.
/// (2012)][rublee].
///
/// [rublee]: http://www.gwylab.com/download/ORB_2012.pdf
#[derive(Clone, PartialEq)]
pub struct OrbDescriptor {
    /// Oriented FAST corner associated with this ORB feature.
    pub keypoint: OrientedFastCorner,
    /// BRIEF descriptor associated with this ORB feature. The pairwise pixel
    /// intensity tests for the BRIEF descriptor are rotated according to the
    /// orientation of the associated oFAST corner.
    pub descriptor: BriefDescriptor,
}

impl BinaryDescriptor for OrbDescriptor {
    fn get_size(&self) -> u32 {
        self.descriptor.get_size()
    }
    fn compute_hamming_distance(&self, other: &Self) -> u32 {
        self.descriptor.compute_hamming_distance(&other.descriptor)
    }
    fn get_bit_subset(&self, bits: &[u32]) -> u128 {
        self.descriptor.get_bit_subset(bits)
    }
    fn get_position(&self) -> Point<u32> {
        self.descriptor.corner.into()
    }
}

fn generate_scale_pyramid_bilinear(
    image: &GrayImage,
    num_scales: usize,
    scale_factor: f32,
) -> Vec<GrayImage> {
    let mut pyramid = vec![image.clone()];
    for idx in 0..num_scales - 1 {
        let previous_layer = &pyramid[idx];
        let (previous_width, previous_height) = previous_layer.dimensions();
        let new_width = ((previous_width as f32) / scale_factor) as u32;
        let new_height = ((previous_height as f32) / scale_factor) as u32;
        let mut new_layer = GrayImage::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                let point_x = x as f32 * scale_factor;
                let point_y = y as f32 * scale_factor;
                let left = point_x.floor();
                let top = point_y.floor();
                let right = left + 1.;
                let bottom = top + 1.;
                let top_left = previous_layer.get_pixel(left as u32, top as u32)[0] as f32;
                let top_right = previous_layer.get_pixel(right as u32, top as u32)[0] as f32;
                let bottom_left = previous_layer.get_pixel(left as u32, bottom as u32)[0] as f32;
                let bottom_right = previous_layer.get_pixel(right as u32, bottom as u32)[0] as f32;
                let top_mid = (top_right - top_left) * point_x.fract() + top_left;
                let bottom_mid = (bottom_right - bottom_left) * point_x.fract() + bottom_left;
                let new_px = (bottom_mid - top_mid) * point_y.fract() + top_mid;
                new_layer.put_pixel(x, y, Luma([new_px as u8]));
            }
        }

        pyramid.push(new_layer);
    }
    pyramid
}

/// Finds rotated BRIEF (rBRIEF) descriptors as presented in [Rublee et. al.
/// (2012)][rublee].
///
/// [rublee]: http://www.gwylab.com/download/ORB_2012.pdf
pub fn rotated_brief(image: &GrayImage, keypoints: &[OrientedFastCorner]) -> Vec<BriefDescriptor> {
    let mut descriptors = Vec::with_capacity(keypoints.len());

    let rotated_brief_test_pairs: Vec<Vec<TestPair>> = (0..30)
        .map(|r| {
            UNROTATED_BRIEF_TEST_PAIRS
                .iter()
                .map(|tp| rotate_test_pair(tp, r as f32 * std::f32::consts::TAU / 30., 15, 15))
                .collect()
        })
        .collect();

    let integral_image = integral_image(image);

    for keypoint in keypoints {
        let rotation_index = discretize_steering_angle(keypoint.orientation);
        let test_pairs = rotated_brief_test_pairs[rotation_index].to_vec();
        let descriptor = brief_impl(
            &integral_image,
            &[Point::new(keypoint.corner.x, keypoint.corner.y)],
            &test_pairs,
            256,
        )
        .unwrap();
        descriptors.push(descriptor[0].clone());
    }

    descriptors
}

/// Given some angle `theta` in radians, compute its discretized steered BRIEF
/// index.
fn discretize_steering_angle(theta: f32) -> usize {
    // add or subtract 2*pi until theta is between 0 and 2*pi, inclusive
    let mut theta_in_range = theta;
    while theta_in_range <= 0. {
        theta_in_range += 2. * std::f32::consts::PI;
    }
    while theta_in_range >= 2. * std::f32::consts::PI {
        theta_in_range -= 2. * std::f32::consts::PI;
    }

    // now map theta onto the positive integers less than 30
    (theta_in_range / (2. * std::f32::consts::PI) * 30.) as usize
}

/// Given some test pair, rotate it around an arbitrary point by some angle
/// theta in radians.
fn rotate_test_pair(test_pair: &TestPair, theta: f32, center_x: u32, center_y: u32) -> TestPair {
    let p0_x = ((test_pair.p0.x as f32 - center_x as f32) * theta.cos()
        - (test_pair.p0.y as f32 - center_y as f32) * theta.sin()
        + center_x as f32) as u32;
    let p0_y = ((test_pair.p0.x as f32 - center_x as f32) * theta.sin()
        + (test_pair.p0.y as f32 - center_y as f32) * theta.cos()
        + center_y as f32) as u32;
    let p1_x = ((test_pair.p1.x as f32 - center_x as f32) * theta.cos()
        - (test_pair.p1.y as f32 - center_y as f32) * theta.sin()
        + center_x as f32) as u32;
    let p1_y = ((test_pair.p1.x as f32 - center_x as f32) * theta.sin()
        + (test_pair.p1.y as f32 - center_y as f32) * theta.cos()
        + center_y as f32) as u32;

    TestPair {
        p0: Point { x: p0_x, y: p0_y },
        p1: Point { x: p1_x, y: p1_y },
    }
}

/// Finds keypoints in `image` and computes ORB descriptors for them.
///
/// See [Rublee et. al. (2012)](http://www.gwylab.com/download/ORB_2012.pdf)
pub fn orb(
    image: &GrayImage,
    target_num_features: usize,
    num_scales: usize,
    scale_factor: f32,
    fast_threshold: Option<u8>,
) -> Vec<OrbDescriptor> {
    let pyramid = generate_scale_pyramid_bilinear(image, num_scales, scale_factor);

    // Find oFAST corners in each scale and compute rBRIEF descriptors for them.
    let mut orbs = vec![];
    for (idx, layer) in pyramid.iter().enumerate() {
        let new_corners = oriented_fast(
            layer,
            fast_threshold,
            target_num_features / num_scales,
            BRIEF_PATCH_RADIUS + 1,
        );

        let new_descriptors = rotated_brief(layer, &new_corners);
        let scaled_corners = new_corners
            .iter()
            .map(|c| OrientedFastCorner {
                corner: Corner {
                    x: (c.corner.x as f32 * scale_factor.powi(idx as i32)) as u32,
                    y: (c.corner.y as f32 * scale_factor.powi(idx as i32)) as u32,
                    score: c.corner.score,
                },
                orientation: c.orientation,
            })
            .collect::<Vec<OrientedFastCorner>>();

        for (keypoint, descriptor) in scaled_corners.into_iter().zip(new_descriptors) {
            orbs.push(OrbDescriptor {
                keypoint,
                descriptor,
            });
        }
    }

    orbs
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use test::{black_box, Bencher};

    use crate::utils::gray_bench_image;

    use super::*;

    #[test]
    fn test_discretize_steering_angle_zero() {
        let theta = 0.;
        assert_eq!(discretize_steering_angle(theta), 0);
    }

    #[test]
    fn test_discretize_steering_angle_two_pi() {
        let theta = 2. * std::f32::consts::PI;
        assert_eq!(discretize_steering_angle(theta), 0);
    }

    #[test]
    fn test_discretize_steering_angle_inside_range() {
        let theta = 1.2;
        assert_eq!(discretize_steering_angle(theta), 5);
    }

    #[test]
    fn test_discretize_steering_angle_above_range() {
        let theta = 7.9;
        assert_eq!(discretize_steering_angle(theta), 7);
    }

    #[test]
    fn test_discretize_steering_angle_below_range() {
        let theta = -11.4;
        assert_eq!(discretize_steering_angle(theta), 5);
    }

    #[test]
    fn test_rotate_test_pair() {
        let theta = 1.829;
        let test_pair = TestPair {
            p0: Point { x: 26, y: 12 },
            p1: Point { x: 22, y: 9 },
        };
        assert_eq!(
            rotate_test_pair(&test_pair, theta, 15, 15),
            TestPair {
                p0: Point { x: 15, y: 26 },
                p1: Point { x: 19, y: 23 },
            }
        );

        let theta = -1.778;
        let test_pair = TestPair {
            p0: Point { x: 9, y: 31 },
            p1: Point { x: 28, y: 29 },
        };
        assert_eq!(
            rotate_test_pair(&test_pair, theta, 15, 15),
            TestPair {
                p0: Point { x: 31, y: 17 },
                p1: Point { x: 26, y: 0 },
            }
        );
    }

    #[bench]
    #[ignore]
    fn bench_scale_pyramid(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        b.iter(|| {
            black_box(generate_scale_pyramid_bilinear(&image, 5, 1.414));
        })
    }

    #[bench]
    #[ignore]
    fn bench_orb_target_1000_features(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        b.iter(|| {
            black_box(orb(&image, 1000, 5, 1.414, Some(3)));
        })
    }

    #[bench]
    #[ignore]
    fn bench_rotated_brief_1000_keypoints(b: &mut Bencher) {
        let image = gray_bench_image(640, 480);
        let mut rng = rand::thread_rng();
        let keypoints = (0..1000)
            .into_iter()
            .map(|_| OrientedFastCorner {
                corner: Corner::new(
                    rng.gen_range(24, image.width() - 24),
                    rng.gen_range(24, image.height() - 24),
                    0.,
                ),
                orientation: rng.gen_range(-std::f32::consts::PI, std::f32::consts::PI),
            })
            .collect::<Vec<OrientedFastCorner>>();
        b.iter(|| {
            black_box(rotated_brief(&image, &keypoints));
        })
    }
}
