#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Bits64(u64);

impl Bits64 {
    const N: usize = 64;

    pub fn new(v: impl IntoIterator<Item = bool>) -> Self {
        let mut bits = Self::zeros();
        let mut n = 0;
        for bit in v {
            if bit {
                bits.set_bit_at(n);
            } else {
                bits.unset_bit_at(n);
            };
            n += 1;
        }
        assert_eq!(n, Self::N);
        bits
    }
    pub fn zeros() -> Self {
        Self(0)
    }
    #[allow(dead_code)]
    pub fn to_bitarray(self) -> [bool; Self::N] {
        let mut bits = [false; Self::N];
        for (n, bit) in bits.iter_mut().enumerate() {
            *bit = self.bit_at(n)
        }
        bits
    }
    pub fn hamming_distance(self, other: Bits64) -> u32 {
        self.xor(other).0.count_ones()
    }
    fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
    fn bit_at(self, n: usize) -> bool {
        assert!(n < Self::N);
        let bit = self.0 & (1 << n);
        if bit == 0 {
            false
        } else {
            true
        }
    }
    fn set_bit_at(&mut self, n: usize) {
        assert!(n < Self::N);
        self.0 |= 1 << n;
    }
    fn unset_bit_at(&mut self, n: usize) {
        assert!(n < Self::N);
        self.0 &= !(1 << n);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_bits64_ops() {
        let mut bits = Bits64::zeros();
        bits.set_bit_at(0);
        assert_eq!(bits, Bits64(1));
        bits.set_bit_at(1);
        assert_eq!(bits, Bits64(1 + 2));
        bits.unset_bit_at(0);
        assert_eq!(bits, Bits64(2));
        bits.unset_bit_at(1);
        assert_eq!(bits, Bits64::zeros());

        bits.set_bit_at(2);
        assert_eq!(bits, Bits64(4));
    }
    #[test]
    fn test_bitarray() {
        let mut v = [false; Bits64::N];
        v[3] = true;
        v[6] = true;
        let bits = Bits64::new(v);
        assert_eq!(bits.to_bitarray(), v);
    }
    #[test]
    fn test_bits64_new() {
        const N: usize = 64;

        let mut v = [false; N];
        v[0] = true;
        assert_eq!(Bits64::new(v), Bits64(1));
        v[1] = true;
        assert_eq!(Bits64::new(v), Bits64(1 + 2));
    }
    #[test]
    #[should_panic]
    fn test_bits64_new_fail() {
        const N: usize = 64;
        let it = (1..N).map(|x| x % 2 == 0);
        let _bits = Bits64::new(it);
    }

    #[test]
    fn test_hash() {
        let one = Bits64(1);
        let map = HashMap::from([(one, "1")]);
        assert_eq!(map[&one], "1");
    }
}
