//! Functions for computing [local binary patterns](https://en.wikipedia.org/wiki/Local_binary_patterns).

use image::{
    GenericImage,
    Luma
};
use std::cmp;

/// Computes the basic local binary pattern of a pixel, or None
/// if it's too close to the image boundary.
pub fn local_binary_pattern<I>(image: &I, x: u32, y: u32) -> Option<u8>
    where I: GenericImage<Pixel=Luma<u8>> {

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return None;
    }

    // TODO: It might be better to make this function private, and
    // TODO: require the caller to only provide valid x and y coordinates.
    if x == 0 || x >= width - 1 || y == 0 || y >= height - 1 {
        return None;
    }

    // TODO: As with the fast corner detectors, this would be more efficient if
    // TODO: generated a list of pixel offsets once per image, and used those
    // TODO: offsets directly when reading pixels. To do this we'd need some traits
    // TODO: for images whose pixels are stored in contiguous rows/columns.
    let mut pattern = 0u8;

    let center = image.get_pixel(x, y)[0];

    // The sampled pixels have the following labels.
    //
    // 7  0  1
    // 6  p  2
    // 5  4  3
    //
    // The nth bit of a pattern is 1 if the pixel p
    // is strictly brighter than the neighbor in position n.
    let neighbors = [
        image.get_pixel(x    , y - 1)[0],
        image.get_pixel(x + 1, y - 1)[0],
        image.get_pixel(x + 1, y    )[0],
        image.get_pixel(x + 1, y + 1)[0],
        image.get_pixel(x    , y + 1)[0],
        image.get_pixel(x - 1, y + 1)[0],
        image.get_pixel(x - 1, y    )[0],
        image.get_pixel(x - 1, y - 1)[0]
    ];

    for i in 0..8 {
        pattern |= (1 & (neighbors[i] < center) as u8) << i;
    }

    Some(pattern)
}

// TODO: add lookup tables from pattern to pattern class, one just assigning
// TODO: the least circular shift and the other lumping all non-uniform patterns together

/// Returns the least value of all rotations of a byte.
pub fn min_shift(byte: u8) -> u8 {
    let mut min = byte;
    for i in 1..8 {
        min = cmp::min(min, byte.rotate_right(i));
    }
    min
}

/// Lookup table for the least value of all rotations of a byte.
let static MIN_SHIFT: [u8; 256] = [
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
mod test {

    use super::{
        local_binary_pattern,
        min_shift
    };
    use image::{
        GenericImage,
        GrayImage,
        ImageBuffer,
        Luma
    };
    use test;

    #[test]
    fn test_local_binary_pattern() {

        let image: GrayImage = ImageBuffer::from_raw(3, 3, vec![
            06, 11, 14,
            09, 10, 10,
            19, 00, 22]).unwrap();

        let expected = 0b11010000;
        let pattern = local_binary_pattern(&image, 1, 1).unwrap();

        assert_eq!(pattern, expected);
    }

    #[test]
    fn test_min_shift() {
        let byte = 0b10110100;
        assert_eq!(min_shift(byte), 0b00101101);
    }
}
