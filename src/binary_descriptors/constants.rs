use crate::point::Point;

use super::brief::TestPair;

pub const BRIEF_PATCH_RADIUS: u32 = 15;
pub const BRIEF_PATCH_DIAMETER: u32 = BRIEF_PATCH_RADIUS * 2 + 1;

// The following constant value is copied from OpenCV under the terms of the BSD
// License. Each of the original values are 15 less than those shown here.
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
pub const UNROTATED_BRIEF_TEST_PAIRS: [TestPair; 256] = [
    TestPair {
        p0: Point { x: 23, y: 12 },
        p1: Point { x: 24, y: 20 },
    },
    TestPair {
        p0: Point { x: 19, y: 17 },
        p1: Point { x: 22, y: 3 },
    },
    TestPair {
        p0: Point { x: 4, y: 24 },
        p1: Point { x: 7, y: 17 },
    },
    TestPair {
        p0: Point { x: 22, y: 3 },
        p1: Point { x: 27, y: 2 },
    },
    TestPair {
        p0: Point { x: 17, y: 2 },
        p1: Point { x: 17, y: 27 },
    },
    TestPair {
        p0: Point { x: 16, y: 8 },
        p1: Point { x: 16, y: 21 },
    },
    TestPair {
        p0: Point { x: 13, y: 5 },
        p1: Point { x: 13, y: 11 },
    },
    TestPair {
        p0: Point { x: 2, y: 2 },
        p1: Point { x: 4, y: 7 },
    },
    TestPair {
        p0: Point { x: 2, y: 12 },
        p1: Point { x: 3, y: 6 },
    },
    TestPair {
        p0: Point { x: 25, y: 19 },
        p1: Point { x: 26, y: 24 },
    },
    TestPair {
        p0: Point { x: 2, y: 7 },
        p1: Point { x: 7, y: 6 },
    },
    TestPair {
        p0: Point { x: 4, y: 22 },
        p1: Point { x: 6, y: 27 },
    },
    TestPair {
        p0: Point { x: 22, y: 22 },
        p1: Point { x: 27, y: 21 },
    },
    TestPair {
        p0: Point { x: 11, y: 10 },
        p1: Point { x: 12, y: 15 },
    },
    TestPair {
        p0: Point { x: 2, y: 17 },
        p1: Point { x: 3, y: 12 },
    },
    TestPair {
        p0: Point { x: 6, y: 15 },
        p1: Point { x: 8, y: 20 },
    },
    TestPair {
        p0: Point { x: 27, y: 9 },
        p1: Point { x: 27, y: 14 },
    },
    TestPair {
        p0: Point { x: 12, y: 21 },
        p1: Point { x: 13, y: 27 },
    },
    TestPair {
        p0: Point { x: 9, y: 2 },
        p1: Point { x: 11, y: 7 },
    },
    TestPair {
        p0: Point { x: 26, y: 2 },
        p1: Point { x: 27, y: 7 },
    },
    TestPair {
        p0: Point { x: 19, y: 22 },
        p1: Point { x: 20, y: 16 },
    },
    TestPair {
        p0: Point { x: 20, y: 12 },
        p1: Point { x: 25, y: 12 },
    },
    TestPair {
        p0: Point { x: 18, y: 8 },
        p1: Point { x: 21, y: 27 },
    },
    TestPair {
        p0: Point { x: 7, y: 8 },
        p1: Point { x: 9, y: 13 },
    },
    TestPair {
        p0: Point { x: 13, y: 26 },
        p1: Point { x: 14, y: 5 },
    },
    TestPair {
        p0: Point { x: 2, y: 27 },
        p1: Point { x: 7, y: 25 },
    },
    TestPair {
        p0: Point { x: 8, y: 18 },
        p1: Point { x: 10, y: 12 },
    },
    TestPair {
        p0: Point { x: 11, y: 17 },
        p1: Point { x: 12, y: 22 },
    },
    TestPair {
        p0: Point { x: 5, y: 3 },
        p1: Point { x: 9, y: 26 },
    },
    TestPair {
        p0: Point { x: 20, y: 3 },
        p1: Point { x: 21, y: 8 },
    },
    TestPair {
        p0: Point { x: 20, y: 9 },
        p1: Point { x: 22, y: 14 },
    },
    TestPair {
        p0: Point { x: 16, y: 15 },
        p1: Point { x: 19, y: 10 },
    },
    TestPair {
        p0: Point { x: 24, y: 26 },
        p1: Point { x: 26, y: 2 },
    },
    TestPair {
        p0: Point { x: 19, y: 22 },
        p1: Point { x: 19, y: 27 },
    },
    TestPair {
        p0: Point { x: 17, y: 14 },
        p1: Point { x: 19, y: 19 },
    },
    TestPair {
        p0: Point { x: 11, y: 3 },
        p1: Point { x: 13, y: 22 },
    },
    TestPair {
        p0: Point { x: 7, y: 10 },
        p1: Point { x: 8, y: 5 },
    },
    TestPair {
        p0: Point { x: 19, y: 26 },
        p1: Point { x: 24, y: 27 },
    },
    TestPair {
        p0: Point { x: 15, y: 7 },
        p1: Point { x: 16, y: 2 },
    },
    TestPair {
        p0: Point { x: 2, y: 13 },
        p1: Point { x: 7, y: 17 },
    },
    TestPair {
        p0: Point { x: 12, y: 13 },
        p1: Point { x: 13, y: 18 },
    },
    TestPair {
        p0: Point { x: 9, y: 24 },
        p1: Point { x: 11, y: 6 },
    },
    TestPair {
        p0: Point { x: 23, y: 27 },
        p1: Point { x: 25, y: 22 },
    },
    TestPair {
        p0: Point { x: 15, y: 24 },
        p1: Point { x: 16, y: 18 },
    },
    TestPair {
        p0: Point { x: 22, y: 10 },
        p1: Point { x: 26, y: 5 },
    },
    TestPair {
        p0: Point { x: 2, y: 9 },
        p1: Point { x: 4, y: 15 },
    },
    TestPair {
        p0: Point { x: 25, y: 22 },
        p1: Point { x: 27, y: 16 },
    },
    TestPair {
        p0: Point { x: 9, y: 12 },
        p1: Point { x: 9, y: 27 },
    },
    TestPair {
        p0: Point { x: 25, y: 6 },
        p1: Point { x: 27, y: 11 },
    },
    TestPair {
        p0: Point { x: 2, y: 23 },
        p1: Point { x: 7, y: 3 },
    },
    TestPair {
        p0: Point { x: 2, y: 15 },
        p1: Point { x: 7, y: 11 },
    },
    TestPair {
        p0: Point { x: 18, y: 18 },
        p1: Point { x: 22, y: 23 },
    },
    TestPair {
        p0: Point { x: 20, y: 22 },
        p1: Point { x: 25, y: 8 },
    },
    TestPair {
        p0: Point { x: 14, y: 22 },
        p1: Point { x: 16, y: 3 },
    },
    TestPair {
        p0: Point { x: 18, y: 5 },
        p1: Point { x: 20, y: 21 },
    },
    TestPair {
        p0: Point { x: 17, y: 11 },
        p1: Point { x: 18, y: 5 },
    },
    TestPair {
        p0: Point { x: 2, y: 15 },
        p1: Point { x: 2, y: 20 },
    },
    TestPair {
        p0: Point { x: 2, y: 8 },
        p1: Point { x: 3, y: 27 },
    },
    TestPair {
        p0: Point { x: 2, y: 18 },
        p1: Point { x: 4, y: 23 },
    },
    TestPair {
        p0: Point { x: 8, y: 27 },
        p1: Point { x: 11, y: 22 },
    },
    TestPair {
        p0: Point { x: 21, y: 5 },
        p1: Point { x: 27, y: 23 },
    },
    TestPair {
        p0: Point { x: 6, y: 14 },
        p1: Point { x: 8, y: 9 },
    },
    TestPair {
        p0: Point { x: 13, y: 10 },
        p1: Point { x: 15, y: 27 },
    },
    TestPair {
        p0: Point { x: 3, y: 20 },
        p1: Point { x: 8, y: 20 },
    },
    TestPair {
        p0: Point { x: 18, y: 5 },
        p1: Point { x: 23, y: 2 },
    },
    TestPair {
        p0: Point { x: 8, y: 8 },
        p1: Point { x: 11, y: 20 },
    },
    TestPair {
        p0: Point { x: 12, y: 13 },
        p1: Point { x: 14, y: 8 },
    },
    TestPair {
        p0: Point { x: 17, y: 24 },
        p1: Point { x: 20, y: 4 },
    },
    TestPair {
        p0: Point { x: 4, y: 2 },
        p1: Point { x: 10, y: 2 },
    },
    TestPair {
        p0: Point { x: 14, y: 21 },
        p1: Point { x: 15, y: 14 },
    },
    TestPair {
        p0: Point { x: 20, y: 12 },
        p1: Point { x: 20, y: 17 },
    },
    TestPair {
        p0: Point { x: 11, y: 2 },
        p1: Point { x: 11, y: 27 },
    },
    TestPair {
        p0: Point { x: 6, y: 9 },
        p1: Point { x: 6, y: 21 },
    },
    TestPair {
        p0: Point { x: 3, y: 5 },
        p1: Point { x: 7, y: 11 },
    },
    TestPair {
        p0: Point { x: 25, y: 17 },
        p1: Point { x: 27, y: 12 },
    },
    TestPair {
        p0: Point { x: 22, y: 27 },
        p1: Point { x: 27, y: 27 },
    },
    TestPair {
        p0: Point { x: 8, y: 2 },
        p1: Point { x: 9, y: 20 },
    },
    TestPair {
        p0: Point { x: 11, y: 24 },
        p1: Point { x: 12, y: 19 },
    },
    TestPair {
        p0: Point { x: 22, y: 14 },
        p1: Point { x: 27, y: 17 },
    },
    TestPair {
        p0: Point { x: 8, y: 21 },
        p1: Point { x: 10, y: 16 },
    },
    TestPair {
        p0: Point { x: 2, y: 26 },
        p1: Point { x: 3, y: 20 },
    },
    TestPair {
        p0: Point { x: 12, y: 22 },
        p1: Point { x: 13, y: 9 },
    },
    TestPair {
        p0: Point { x: 22, y: 7 },
        p1: Point { x: 27, y: 8 },
    },
    TestPair {
        p0: Point { x: 2, y: 8 },
        p1: Point { x: 4, y: 3 },
    },
    TestPair {
        p0: Point { x: 16, y: 12 },
        p1: Point { x: 27, y: 27 },
    },
    TestPair {
        p0: Point { x: 17, y: 9 },
        p1: Point { x: 18, y: 15 },
    },
    TestPair {
        p0: Point { x: 11, y: 18 },
        p1: Point { x: 13, y: 2 },
    },
    TestPair {
        p0: Point { x: 14, y: 2 },
        p1: Point { x: 16, y: 24 },
    },
    TestPair {
        p0: Point { x: 22, y: 16 },
        p1: Point { x: 23, y: 9 },
    },
    TestPair {
        p0: Point { x: 16, y: 14 },
        p1: Point { x: 18, y: 27 },
    },
    TestPair {
        p0: Point { x: 24, y: 16 },
        p1: Point { x: 27, y: 21 },
    },
    TestPair {
        p0: Point { x: 14, y: 6 },
        p1: Point { x: 14, y: 18 },
    },
    TestPair {
        p0: Point { x: 2, y: 2 },
        p1: Point { x: 5, y: 20 },
    },
    TestPair {
        p0: Point { x: 22, y: 22 },
        p1: Point { x: 25, y: 27 },
    },
    TestPair {
        p0: Point { x: 27, y: 10 },
        p1: Point { x: 27, y: 24 },
    },
    TestPair {
        p0: Point { x: 21, y: 18 },
        p1: Point { x: 22, y: 26 },
    },
    TestPair {
        p0: Point { x: 20, y: 2 },
        p1: Point { x: 21, y: 25 },
    },
    TestPair {
        p0: Point { x: 17, y: 3 },
        p1: Point { x: 17, y: 18 },
    },
    TestPair {
        p0: Point { x: 18, y: 23 },
        p1: Point { x: 19, y: 9 },
    },
    TestPair {
        p0: Point { x: 17, y: 21 },
        p1: Point { x: 27, y: 2 },
    },
    TestPair {
        p0: Point { x: 24, y: 3 },
        p1: Point { x: 25, y: 18 },
    },
    TestPair {
        p0: Point { x: 7, y: 19 },
        p1: Point { x: 8, y: 24 },
    },
    TestPair {
        p0: Point { x: 4, y: 27 },
        p1: Point { x: 11, y: 9 },
    },
    TestPair {
        p0: Point { x: 16, y: 27 },
        p1: Point { x: 17, y: 7 },
    },
    TestPair {
        p0: Point { x: 21, y: 6 },
        p1: Point { x: 22, y: 11 },
    },
    TestPair {
        p0: Point { x: 17, y: 18 },
        p1: Point { x: 18, y: 13 },
    },
    TestPair {
        p0: Point { x: 21, y: 18 },
        p1: Point { x: 26, y: 15 },
    },
    TestPair {
        p0: Point { x: 18, y: 12 },
        p1: Point { x: 23, y: 7 },
    },
    TestPair {
        p0: Point { x: 22, y: 23 },
        p1: Point { x: 24, y: 18 },
    },
    TestPair {
        p0: Point { x: 4, y: 10 },
        p1: Point { x: 9, y: 11 },
    },
    TestPair {
        p0: Point { x: 5, y: 26 },
        p1: Point { x: 10, y: 25 },
    },
    TestPair {
        p0: Point { x: 10, y: 7 },
        p1: Point { x: 12, y: 27 },
    },
    TestPair {
        p0: Point { x: 5, y: 20 },
        p1: Point { x: 6, y: 15 },
    },
    TestPair {
        p0: Point { x: 23, y: 14 },
        p1: Point { x: 27, y: 9 },
    },
    TestPair {
        p0: Point { x: 19, y: 9 },
        p1: Point { x: 21, y: 4 },
    },
    TestPair {
        p0: Point { x: 5, y: 27 },
        p1: Point { x: 7, y: 22 },
    },
    TestPair {
        p0: Point { x: 19, y: 13 },
        p1: Point { x: 21, y: 22 },
    },
    TestPair {
        p0: Point { x: 13, y: 15 },
        p1: Point { x: 13, y: 27 },
    },
    TestPair {
        p0: Point { x: 10, y: 7 },
        p1: Point { x: 10, y: 17 },
    },
    TestPair {
        p0: Point { x: 22, y: 9 },
        p1: Point { x: 25, y: 27 },
    },
    TestPair {
        p0: Point { x: 6, y: 2 },
        p1: Point { x: 7, y: 7 },
    },
    TestPair {
        p0: Point { x: 10, y: 2 },
        p1: Point { x: 10, y: 13 },
    },
    TestPair {
        p0: Point { x: 23, y: 7 },
        p1: Point { x: 24, y: 2 },
    },
    TestPair {
        p0: Point { x: 6, y: 4 },
        p1: Point { x: 6, y: 15 },
    },
    TestPair {
        p0: Point { x: 16, y: 7 },
        p1: Point { x: 16, y: 13 },
    },
    TestPair {
        p0: Point { x: 22, y: 11 },
        p1: Point { x: 24, y: 16 },
    },
    TestPair {
        p0: Point { x: 13, y: 16 },
        p1: Point { x: 14, y: 11 },
    },
    TestPair {
        p0: Point { x: 26, y: 9 },
        p1: Point { x: 27, y: 4 },
    },
    TestPair {
        p0: Point { x: 3, y: 6 },
        p1: Point { x: 9, y: 19 },
    },
    TestPair {
        p0: Point { x: 18, y: 22 },
        p1: Point { x: 22, y: 27 },
    },
    TestPair {
        p0: Point { x: 20, y: 20 },
        p1: Point { x: 25, y: 23 },
    },
    TestPair {
        p0: Point { x: 15, y: 11 },
        p1: Point { x: 17, y: 23 },
    },
    TestPair {
        p0: Point { x: 6, y: 27 },
        p1: Point { x: 10, y: 2 },
    },
    TestPair {
        p0: Point { x: 15, y: 22 },
        p1: Point { x: 17, y: 27 },
    },
    TestPair {
        p0: Point { x: 14, y: 17 },
        p1: Point { x: 16, y: 22 },
    },
    TestPair {
        p0: Point { x: 20, y: 26 },
        p1: Point { x: 22, y: 6 },
    },
    TestPair {
        p0: Point { x: 18, y: 20 },
        p1: Point { x: 21, y: 7 },
    },
    TestPair {
        p0: Point { x: 2, y: 11 },
        p1: Point { x: 7, y: 24 },
    },
    TestPair {
        p0: Point { x: 10, y: 24 },
        p1: Point { x: 12, y: 12 },
    },
    TestPair {
        p0: Point { x: 11, y: 8 },
        p1: Point { x: 12, y: 3 },
    },
    TestPair {
        p0: Point { x: 21, y: 20 },
        p1: Point { x: 23, y: 15 },
    },
    TestPair {
        p0: Point { x: 8, y: 21 },
        p1: Point { x: 9, y: 27 },
    },
    TestPair {
        p0: Point { x: 2, y: 21 },
        p1: Point { x: 10, y: 13 },
    },
    TestPair {
        p0: Point { x: 16, y: 5 },
        p1: Point { x: 18, y: 25 },
    },
    TestPair {
        p0: Point { x: 19, y: 16 },
        p1: Point { x: 23, y: 11 },
    },
    TestPair {
        p0: Point { x: 13, y: 13 },
        p1: Point { x: 17, y: 2 },
    },
    TestPair {
        p0: Point { x: 17, y: 3 },
        p1: Point { x: 27, y: 27 },
    },
    TestPair {
        p0: Point { x: 13, y: 2 },
        p1: Point { x: 15, y: 9 },
    },
    TestPair {
        p0: Point { x: 19, y: 16 },
        p1: Point { x: 24, y: 18 },
    },
    TestPair {
        p0: Point { x: 9, y: 5 },
        p1: Point { x: 12, y: 10 },
    },
    TestPair {
        p0: Point { x: 12, y: 2 },
        p1: Point { x: 14, y: 16 },
    },
    TestPair {
        p0: Point { x: 22, y: 20 },
        p1: Point { x: 27, y: 4 },
    },
    TestPair {
        p0: Point { x: 19, y: 13 },
        p1: Point { x: 20, y: 8 },
    },
    TestPair {
        p0: Point { x: 2, y: 24 },
        p1: Point { x: 6, y: 10 },
    },
    TestPair {
        p0: Point { x: 22, y: 16 },
        p1: Point { x: 23, y: 21 },
    },
    TestPair {
        p0: Point { x: 22, y: 7 },
        p1: Point { x: 22, y: 21 },
    },
    TestPair {
        p0: Point { x: 8, y: 11 },
        p1: Point { x: 8, y: 16 },
    },
    TestPair {
        p0: Point { x: 7, y: 26 },
        p1: Point { x: 8, y: 7 },
    },
    TestPair {
        p0: Point { x: 2, y: 21 },
        p1: Point { x: 3, y: 7 },
    },
    TestPair {
        p0: Point { x: 17, y: 19 },
        p1: Point { x: 18, y: 24 },
    },
    TestPair {
        p0: Point { x: 25, y: 10 },
        p1: Point { x: 27, y: 18 },
    },
    TestPair {
        p0: Point { x: 9, y: 10 },
        p1: Point { x: 9, y: 22 },
    },
    TestPair {
        p0: Point { x: 23, y: 12 },
        p1: Point { x: 24, y: 7 },
    },
    TestPair {
        p0: Point { x: 17, y: 3 },
        p1: Point { x: 17, y: 23 },
    },
    TestPair {
        p0: Point { x: 4, y: 13 },
        p1: Point { x: 5, y: 18 },
    },
    TestPair {
        p0: Point { x: 3, y: 2 },
        p1: Point { x: 8, y: 6 },
    },
    TestPair {
        p0: Point { x: 4, y: 15 },
        p1: Point { x: 5, y: 10 },
    },
    TestPair {
        p0: Point { x: 20, y: 12 },
        p1: Point { x: 26, y: 23 },
    },
    TestPair {
        p0: Point { x: 13, y: 2 },
        p1: Point { x: 14, y: 27 },
    },
    TestPair {
        p0: Point { x: 14, y: 7 },
        p1: Point { x: 15, y: 24 },
    },
    TestPair {
        p0: Point { x: 2, y: 4 },
        p1: Point { x: 3, y: 10 },
    },
    TestPair {
        p0: Point { x: 5, y: 13 },
        p1: Point { x: 5, y: 26 },
    },
    TestPair {
        p0: Point { x: 12, y: 24 },
        p1: Point { x: 13, y: 2 },
    },
    TestPair {
        p0: Point { x: 17, y: 12 },
        p1: Point { x: 18, y: 17 },
    },
    TestPair {
        p0: Point { x: 6, y: 2 },
        p1: Point { x: 11, y: 15 },
    },
    TestPair {
        p0: Point { x: 11, y: 21 },
        p1: Point { x: 12, y: 5 },
    },
    TestPair {
        p0: Point { x: 11, y: 27 },
        p1: Point { x: 13, y: 8 },
    },
    TestPair {
        p0: Point { x: 9, y: 4 },
        p1: Point { x: 11, y: 24 },
    },
    TestPair {
        p0: Point { x: 21, y: 12 },
        p1: Point { x: 21, y: 26 },
    },
    TestPair {
        p0: Point { x: 2, y: 26 },
        p1: Point { x: 10, y: 20 },
    },
    TestPair {
        p0: Point { x: 26, y: 26 },
        p1: Point { x: 27, y: 21 },
    },
    TestPair {
        p0: Point { x: 22, y: 10 },
        p1: Point { x: 27, y: 13 },
    },
    TestPair {
        p0: Point { x: 14, y: 27 },
        p1: Point { x: 15, y: 22 },
    },
    TestPair {
        p0: Point { x: 11, y: 7 },
        p1: Point { x: 12, y: 13 },
    },
    TestPair {
        p0: Point { x: 8, y: 16 },
        p1: Point { x: 9, y: 22 },
    },
    TestPair {
        p0: Point { x: 2, y: 3 },
        p1: Point { x: 7, y: 2 },
    },
    TestPair {
        p0: Point { x: 8, y: 13 },
        p1: Point { x: 9, y: 7 },
    },
    TestPair {
        p0: Point { x: 7, y: 20 },
        p1: Point { x: 9, y: 6 },
    },
    TestPair {
        p0: Point { x: 10, y: 14 },
        p1: Point { x: 11, y: 20 },
    },
    TestPair {
        p0: Point { x: 2, y: 22 },
        p1: Point { x: 7, y: 25 },
    },
    TestPair {
        p0: Point { x: 16, y: 20 },
        p1: Point { x: 20, y: 2 },
    },
    TestPair {
        p0: Point { x: 16, y: 15 },
        p1: Point { x: 25, y: 2 },
    },
    TestPair {
        p0: Point { x: 24, y: 27 },
        p1: Point { x: 25, y: 14 },
    },
    TestPair {
        p0: Point { x: 20, y: 7 },
        p1: Point { x: 25, y: 6 },
    },
    TestPair {
        p0: Point { x: 14, y: 26 },
        p1: Point { x: 16, y: 2 },
    },
    TestPair {
        p0: Point { x: 6, y: 12 },
        p1: Point { x: 9, y: 17 },
    },
    TestPair {
        p0: Point { x: 14, y: 5 },
        p1: Point { x: 16, y: 27 },
    },
    TestPair {
        p0: Point { x: 2, y: 16 },
        p1: Point { x: 7, y: 5 },
    },
    TestPair {
        p0: Point { x: 23, y: 4 },
        p1: Point { x: 25, y: 9 },
    },
    TestPair {
        p0: Point { x: 17, y: 2 },
        p1: Point { x: 18, y: 9 },
    },
    TestPair {
        p0: Point { x: 22, y: 2 },
        p1: Point { x: 27, y: 6 },
    },
    TestPair {
        p0: Point { x: 5, y: 5 },
        p1: Point { x: 10, y: 8 },
    },
    TestPair {
        p0: Point { x: 5, y: 7 },
        p1: Point { x: 7, y: 2 },
    },
    TestPair {
        p0: Point { x: 19, y: 9 },
        p1: Point { x: 23, y: 20 },
    },
    TestPair {
        p0: Point { x: 18, y: 27 },
        p1: Point { x: 23, y: 2 },
    },
    TestPair {
        p0: Point { x: 11, y: 17 },
        p1: Point { x: 12, y: 12 },
    },
    TestPair {
        p0: Point { x: 20, y: 2 },
        p1: Point { x: 25, y: 3 },
    },
    TestPair {
        p0: Point { x: 19, y: 2 },
        p1: Point { x: 20, y: 14 },
    },
    TestPair {
        p0: Point { x: 6, y: 24 },
        p1: Point { x: 11, y: 18 },
    },
    TestPair {
        p0: Point { x: 15, y: 18 },
        p1: Point { x: 18, y: 6 },
    },
    TestPair {
        p0: Point { x: 3, y: 16 },
        p1: Point { x: 9, y: 16 },
    },
    TestPair {
        p0: Point { x: 18, y: 17 },
        p1: Point { x: 19, y: 7 },
    },
    TestPair {
        p0: Point { x: 5, y: 5 },
        p1: Point { x: 5, y: 24 },
    },
    TestPair {
        p0: Point { x: 23, y: 2 },
        p1: Point { x: 27, y: 27 },
    },
    TestPair {
        p0: Point { x: 7, y: 3 },
        p1: Point { x: 9, y: 10 },
    },
    TestPair {
        p0: Point { x: 17, y: 17 },
        p1: Point { x: 18, y: 22 },
    },
    TestPair {
        p0: Point { x: 25, y: 21 },
        p1: Point { x: 26, y: 7 },
    },
    TestPair {
        p0: Point { x: 21, y: 23 },
        p1: Point { x: 23, y: 3 },
    },
    TestPair {
        p0: Point { x: 8, y: 25 },
        p1: Point { x: 9, y: 20 },
    },
    TestPair {
        p0: Point { x: 12, y: 6 },
        p1: Point { x: 12, y: 24 },
    },
    TestPair {
        p0: Point { x: 14, y: 2 },
        p1: Point { x: 14, y: 20 },
    },
    TestPair {
        p0: Point { x: 12, y: 8 },
        p1: Point { x: 12, y: 19 },
    },
    TestPair {
        p0: Point { x: 7, y: 13 },
        p1: Point { x: 7, y: 18 },
    },
    TestPair {
        p0: Point { x: 19, y: 17 },
        p1: Point { x: 27, y: 27 },
    },
    TestPair {
        p0: Point { x: 17, y: 10 },
        p1: Point { x: 18, y: 26 },
    },
    TestPair {
        p0: Point { x: 21, y: 6 },
        p1: Point { x: 26, y: 2 },
    },
    TestPair {
        p0: Point { x: 18, y: 14 },
        p1: Point { x: 22, y: 27 },
    },
    TestPair {
        p0: Point { x: 26, y: 14 },
        p1: Point { x: 27, y: 19 },
    },
    TestPair {
        p0: Point { x: 12, y: 15 },
        p1: Point { x: 12, y: 21 },
    },
    TestPair {
        p0: Point { x: 19, y: 4 },
        p1: Point { x: 19, y: 27 },
    },
    TestPair {
        p0: Point { x: 17, y: 11 },
        p1: Point { x: 17, y: 16 },
    },
    TestPair {
        p0: Point { x: 5, y: 9 },
        p1: Point { x: 7, y: 16 },
    },
    TestPair {
        p0: Point { x: 2, y: 22 },
        p1: Point { x: 4, y: 16 },
    },
    TestPair {
        p0: Point { x: 2, y: 27 },
        p1: Point { x: 4, y: 2 },
    },
    TestPair {
        p0: Point { x: 21, y: 15 },
        p1: Point { x: 26, y: 2 },
    },
    TestPair {
        p0: Point { x: 15, y: 14 },
        p1: Point { x: 16, y: 19 },
    },
    TestPair {
        p0: Point { x: 2, y: 18 },
        p1: Point { x: 6, y: 13 },
    },
    TestPair {
        p0: Point { x: 6, y: 23 },
        p1: Point { x: 9, y: 12 },
    },
    TestPair {
        p0: Point { x: 2, y: 9 },
        p1: Point { x: 7, y: 13 },
    },
    TestPair {
        p0: Point { x: 20, y: 6 },
        p1: Point { x: 23, y: 25 },
    },
    TestPair {
        p0: Point { x: 17, y: 22 },
        p1: Point { x: 18, y: 6 },
    },
    TestPair {
        p0: Point { x: 14, y: 9 },
        p1: Point { x: 14, y: 14 },
    },
    TestPair {
        p0: Point { x: 24, y: 20 },
        p1: Point { x: 26, y: 13 },
    },
    TestPair {
        p0: Point { x: 26, y: 12 },
        p1: Point { x: 27, y: 7 },
    },
    TestPair {
        p0: Point { x: 18, y: 15 },
        p1: Point { x: 18, y: 20 },
    },
    TestPair {
        p0: Point { x: 14, y: 19 },
        p1: Point { x: 15, y: 25 },
    },
    TestPair {
        p0: Point { x: 18, y: 9 },
        p1: Point { x: 19, y: 20 },
    },
    TestPair {
        p0: Point { x: 2, y: 15 },
        p1: Point { x: 5, y: 20 },
    },
    TestPair {
        p0: Point { x: 20, y: 23 },
        p1: Point { x: 27, y: 26 },
    },
    TestPair {
        p0: Point { x: 23, y: 24 },
        p1: Point { x: 24, y: 9 },
    },
    TestPair {
        p0: Point { x: 22, y: 11 },
        p1: Point { x: 23, y: 3 },
    },
    TestPair {
        p0: Point { x: 5, y: 19 },
        p1: Point { x: 5, y: 24 },
    },
    TestPair {
        p0: Point { x: 22, y: 18 },
        p1: Point { x: 27, y: 19 },
    },
    TestPair {
        p0: Point { x: 24, y: 8 },
        p1: Point { x: 25, y: 13 },
    },
    TestPair {
        p0: Point { x: 22, y: 15 },
        p1: Point { x: 27, y: 13 },
    },
    TestPair {
        p0: Point { x: 14, y: 9 },
        p1: Point { x: 15, y: 4 },
    },
];
// End coverage of Willow Garage license
