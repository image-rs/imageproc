//! Functions for computing [local binary patterns](https://en.wikipedia.org/wiki/Local_binary_patterns).

use image::{GenericImage, Luma};
use std::cmp;

/// Computes the basic local binary pattern of a pixel, or None
/// if it's too close to the image boundary.
///
/// The neighbors of a pixel p are enumerated in the following order:
///
/// <pre>
/// 7  0  1
/// 6  p  2
/// 5  4  3
/// </pre>
///
/// The nth most significant bit of the local binary pattern at p is 1
/// if p is strictly brighter than the neighbor in position n.
///
/// # Examples
/// ```
/// # extern crate image;
/// # #[macro_use]
/// # extern crate imageproc;
/// # fn main() {
/// use imageproc::local_binary_patterns::local_binary_pattern;
///
/// let image = gray_image!(
///     06, 11, 14;
///     09, 10, 10;
///     19, 00, 22);
///
/// let expected = 0b11010000;
/// let pattern = local_binary_pattern(&image, 1, 1).unwrap();
/// assert_eq!(pattern, expected);
/// # }
/// ```
pub fn local_binary_pattern<I>(image: &I, x: u32, y: u32) -> Option<u8>
where
    I: GenericImage<Pixel = Luma<u8>>,
{
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return None;
    }

    // TODO: It might be better to make this function private, and
    // TODO: require the caller to only provide valid x and y coordinates
    // TODO: the function may probably need to be unsafe then to leverage
    // TODO: on `unsafe_get_pixel`
    if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
        return None;
    }

    // TODO: As with the fast corner detectors, this would be more efficient if
    // TODO: generated a list of pixel offsets once per image, and used those
    // TODO: offsets directly when reading pixels. To do this we'd need some traits
    // TODO: for images whose pixels are stored in contiguous rows/columns.
    let mut pattern = 0u8;

    // The sampled pixels have the following labels.
    //
    // 7  0  1
    // 6  p  2
    // 5  4  3
    //
    // The nth bit of a pattern is 1 if the pixel p
    // is strictly brighter than the neighbor in position n.
    let (center, neighbors) = unsafe {
        (
            image.unsafe_get_pixel(x, y)[0],
            [
                image.unsafe_get_pixel(x, y - 1)[0],
                image.unsafe_get_pixel(x + 1, y - 1)[0],
                image.unsafe_get_pixel(x + 1, y)[0],
                image.unsafe_get_pixel(x + 1, y + 1)[0],
                image.unsafe_get_pixel(x, y + 1)[0],
                image.unsafe_get_pixel(x - 1, y + 1)[0],
                image.unsafe_get_pixel(x - 1, y)[0],
                image.unsafe_get_pixel(x - 1, y - 1)[0],
            ],
        )
    };

    for i in 0..8 {
        pattern |= (1 & (neighbors[i] < center) as u8) << i;
    }

    Some(pattern)
}

/// Returns the least value of all rotations of a byte.
///
/// # Examples
/// ```
/// use imageproc::local_binary_patterns::min_shift;
///
/// let byte = 0b10110100;
/// assert_eq!(min_shift(byte), 0b00101101);
/// ```
pub fn min_shift(byte: u8) -> u8 {
    let mut min = byte;
    for i in 1..8 {
        min = cmp::min(min, byte.rotate_right(i));
    }
    min
}

/// Number of bit transitions in a byte, counting the last and final bits as adjacent.
///
/// # Examples
/// ```
/// use imageproc::local_binary_patterns::count_transitions;
///
/// let a = 0b11110000;
/// assert_eq!(count_transitions(a), 2);
/// let b = 0b00000000;
/// assert_eq!(count_transitions(b), 0);
/// let c = 0b10011001;
/// assert_eq!(count_transitions(c), 4);
/// let d = 0b10110010;
/// assert_eq!(count_transitions(d), 6);
/// ```
pub fn count_transitions(byte: u8) -> u32 {
    (byte ^ byte.rotate_right(1)).count_ones()
}

/// Maps uniform bytes (i.e. those with at most two bit transitions) to their
/// least circular shifts, and non-uniform bytes to 10101010 (an arbitrarily chosen
/// non-uniform representative).
pub static UNIFORM_REPRESENTATIVE_2: [u8; 256] = [
    0,   // 0
    1,   // 1
    1,   // 2
    3,   // 3
    1,   // 4
    170, // 5
    3,   // 6
    7,   // 7
    1,   // 8
    170, // 9
    170, // 10
    170, // 11
    3,   // 12
    170, // 13
    7,   // 14
    15,  // 15
    1,   // 16
    170, // 17
    170, // 18
    170, // 19
    170, // 20
    170, // 21
    170, // 22
    170, // 23
    3,   // 24
    170, // 25
    170, // 26
    170, // 27
    7,   // 28
    170, // 29
    15,  // 30
    31,  // 31
    1,   // 32
    170, // 33
    170, // 34
    170, // 35
    170, // 36
    170, // 37
    170, // 38
    170, // 39
    170, // 40
    170, // 41
    170, // 42
    170, // 43
    170, // 44
    170, // 45
    170, // 46
    170, // 47
    3,   // 48
    170, // 49
    170, // 50
    170, // 51
    170, // 52
    170, // 53
    170, // 54
    170, // 55
    7,   // 56
    170, // 57
    170, // 58
    170, // 59
    15,  // 60
    170, // 61
    31,  // 62
    63,  // 63
    1,   // 64
    170, // 65
    170, // 66
    170, // 67
    170, // 68
    170, // 69
    170, // 70
    170, // 71
    170, // 72
    170, // 73
    170, // 74
    170, // 75
    170, // 76
    170, // 77
    170, // 78
    170, // 79
    170, // 80
    170, // 81
    170, // 82
    170, // 83
    170, // 84
    170, // 85
    170, // 86
    170, // 87
    170, // 88
    170, // 89
    170, // 90
    170, // 91
    170, // 92
    170, // 93
    170, // 94
    170, // 95
    3,   // 96
    170, // 97
    170, // 98
    170, // 99
    170, // 100
    170, // 101
    170, // 102
    170, // 103
    170, // 104
    170, // 105
    170, // 106
    170, // 107
    170, // 108
    170, // 109
    170, // 110
    170, // 111
    7,   // 112
    170, // 113
    170, // 114
    170, // 115
    170, // 116
    170, // 117
    170, // 118
    170, // 119
    15,  // 120
    170, // 121
    170, // 122
    170, // 123
    31,  // 124
    170, // 125
    63,  // 126
    127, // 127
    1,   // 128
    3,   // 129
    170, // 130
    7,   // 131
    170, // 132
    170, // 133
    170, // 134
    15,  // 135
    170, // 136
    170, // 137
    170, // 138
    170, // 139
    170, // 140
    170, // 141
    170, // 142
    31,  // 143
    170, // 144
    170, // 145
    170, // 146
    170, // 147
    170, // 148
    170, // 149
    170, // 150
    170, // 151
    170, // 152
    170, // 153
    170, // 154
    170, // 155
    170, // 156
    170, // 157
    170, // 158
    63,  // 159
    170, // 160
    170, // 161
    170, // 162
    170, // 163
    170, // 164
    170, // 165
    170, // 166
    170, // 167
    170, // 168
    170, // 169
    170, // 170
    170, // 171
    170, // 172
    170, // 173
    170, // 174
    170, // 175
    170, // 176
    170, // 177
    170, // 178
    170, // 179
    170, // 180
    170, // 181
    170, // 182
    170, // 183
    170, // 184
    170, // 185
    170, // 186
    170, // 187
    170, // 188
    170, // 189
    170, // 190
    127, // 191
    3,   // 192
    7,   // 193
    170, // 194
    15,  // 195
    170, // 196
    170, // 197
    170, // 198
    31,  // 199
    170, // 200
    170, // 201
    170, // 202
    170, // 203
    170, // 204
    170, // 205
    170, // 206
    63,  // 207
    170, // 208
    170, // 209
    170, // 210
    170, // 211
    170, // 212
    170, // 213
    170, // 214
    170, // 215
    170, // 216
    170, // 217
    170, // 218
    170, // 219
    170, // 220
    170, // 221
    170, // 222
    127, // 223
    7,   // 224
    15,  // 225
    170, // 226
    31,  // 227
    170, // 228
    170, // 229
    170, // 230
    63,  // 231
    170, // 232
    170, // 233
    170, // 234
    170, // 235
    170, // 236
    170, // 237
    170, // 238
    127, // 239
    15,  // 240
    31,  // 241
    170, // 242
    63,  // 243
    170, // 244
    170, // 245
    170, // 246
    127, // 247
    31,  // 248
    63,  // 249
    170, // 250
    127, // 251
    63,  // 252
    127, // 253
    127, // 254
    255, // 255
];

/// Lookup table for the least circular shift of a byte.
pub static MIN_SHIFT: [u8; 256] = [
    0,   // 0
    1,   // 1
    1,   // 2
    3,   // 3
    1,   // 4
    5,   // 5
    3,   // 6
    7,   // 7
    1,   // 8
    9,   // 9
    5,   // 10
    11,  // 11
    3,   // 12
    13,  // 13
    7,   // 14
    15,  // 15
    1,   // 16
    17,  // 17
    9,   // 18
    19,  // 19
    5,   // 20
    21,  // 21
    11,  // 22
    23,  // 23
    3,   // 24
    25,  // 25
    13,  // 26
    27,  // 27
    7,   // 28
    29,  // 29
    15,  // 30
    31,  // 31
    1,   // 32
    9,   // 33
    17,  // 34
    25,  // 35
    9,   // 36
    37,  // 37
    19,  // 38
    39,  // 39
    5,   // 40
    37,  // 41
    21,  // 42
    43,  // 43
    11,  // 44
    45,  // 45
    23,  // 46
    47,  // 47
    3,   // 48
    19,  // 49
    25,  // 50
    51,  // 51
    13,  // 52
    53,  // 53
    27,  // 54
    55,  // 55
    7,   // 56
    39,  // 57
    29,  // 58
    59,  // 59
    15,  // 60
    61,  // 61
    31,  // 62
    63,  // 63
    1,   // 64
    5,   // 65
    9,   // 66
    13,  // 67
    17,  // 68
    21,  // 69
    25,  // 70
    29,  // 71
    9,   // 72
    37,  // 73
    37,  // 74
    45,  // 75
    19,  // 76
    53,  // 77
    39,  // 78
    61,  // 79
    5,   // 80
    21,  // 81
    37,  // 82
    53,  // 83
    21,  // 84
    85,  // 85
    43,  // 86
    87,  // 87
    11,  // 88
    43,  // 89
    45,  // 90
    91,  // 91
    23,  // 92
    87,  // 93
    47,  // 94
    95,  // 95
    3,   // 96
    11,  // 97
    19,  // 98
    27,  // 99
    25,  // 100
    43,  // 101
    51,  // 102
    59,  // 103
    13,  // 104
    45,  // 105
    53,  // 106
    91,  // 107
    27,  // 108
    91,  // 109
    55,  // 110
    111, // 111
    7,   // 112
    23,  // 113
    39,  // 114
    55,  // 115
    29,  // 116
    87,  // 117
    59,  // 118
    119, // 119
    15,  // 120
    47,  // 121
    61,  // 122
    111, // 123
    31,  // 124
    95,  // 125
    63,  // 126
    127, // 127
    1,   // 128
    3,   // 129
    5,   // 130
    7,   // 131
    9,   // 132
    11,  // 133
    13,  // 134
    15,  // 135
    17,  // 136
    19,  // 137
    21,  // 138
    23,  // 139
    25,  // 140
    27,  // 141
    29,  // 142
    31,  // 143
    9,   // 144
    25,  // 145
    37,  // 146
    39,  // 147
    37,  // 148
    43,  // 149
    45,  // 150
    47,  // 151
    19,  // 152
    51,  // 153
    53,  // 154
    55,  // 155
    39,  // 156
    59,  // 157
    61,  // 158
    63,  // 159
    5,   // 160
    13,  // 161
    21,  // 162
    29,  // 163
    37,  // 164
    45,  // 165
    53,  // 166
    61,  // 167
    21,  // 168
    53,  // 169
    85,  // 170
    87,  // 171
    43,  // 172
    91,  // 173
    87,  // 174
    95,  // 175
    11,  // 176
    27,  // 177
    43,  // 178
    59,  // 179
    45,  // 180
    91,  // 181
    91,  // 182
    111, // 183
    23,  // 184
    55,  // 185
    87,  // 186
    119, // 187
    47,  // 188
    111, // 189
    95,  // 190
    127, // 191
    3,   // 192
    7,   // 193
    11,  // 194
    15,  // 195
    19,  // 196
    23,  // 197
    27,  // 198
    31,  // 199
    25,  // 200
    39,  // 201
    43,  // 202
    47,  // 203
    51,  // 204
    55,  // 205
    59,  // 206
    63,  // 207
    13,  // 208
    29,  // 209
    45,  // 210
    61,  // 211
    53,  // 212
    87,  // 213
    91,  // 214
    95,  // 215
    27,  // 216
    59,  // 217
    91,  // 218
    111, // 219
    55,  // 220
    119, // 221
    111, // 222
    127, // 223
    7,   // 224
    15,  // 225
    23,  // 226
    31,  // 227
    39,  // 228
    47,  // 229
    55,  // 230
    63,  // 231
    29,  // 232
    61,  // 233
    87,  // 234
    95,  // 235
    59,  // 236
    111, // 237
    119, // 238
    127, // 239
    15,  // 240
    31,  // 241
    47,  // 242
    63,  // 243
    61,  // 244
    95,  // 245
    111, // 246
    127, // 247
    31,  // 248
    63,  // 249
    95,  // 250
    127, // 251
    63,  // 252
    127, // 253
    127, // 254
    255, // 255
];

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use test::{black_box, Bencher};

    #[test]
    fn test_uniform_representative_2() {
        let a = 0b11110000;
        assert_eq!(UNIFORM_REPRESENTATIVE_2[a], 0b00001111);
        let b = 0b00000000;
        assert_eq!(UNIFORM_REPRESENTATIVE_2[b], 0b00000000);
        let c = 0b10011001;
        assert_eq!(UNIFORM_REPRESENTATIVE_2[c], 0b10101010);
    }

    #[bench]
    fn bench_local_binary_pattern(b: &mut Bencher) {
        let image = GrayImage::from_fn(100, 100, |x, y| Luma([x as u8 % 2 + y as u8 % 2]));
        b.iter(|| {
            for y in 0..20 {
                for x in 0..20 {
                    let pattern = local_binary_pattern(&image, x, y);
                    black_box(pattern);
                }
            }
        });
    }
}
