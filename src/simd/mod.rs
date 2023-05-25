pub use linear::{linear_search, linear_search_generic};
pub use simd_bst::binary_search;

mod linear;
mod simd_bst;

pub trait SIMDField: Sized + Copy + num::Integer {
    fn size_in_bits() -> usize;

    fn unchecked_i8(self) -> i8;

    fn unchecked_u8(self) -> u8;

    fn unchecked_i16(self) -> i16;

    fn unchecked_u16(self) -> u16;

    fn unchecked_i32(self) -> i32;

    fn unchecked_u32(self) -> u32;

    fn unchecked_i64(self) -> i64;

    fn unchecked_u64(self) -> u64;
}

macro_rules! simd_suit {
    ($t:ty, $size:expr) => {
        impl SIMDField for $t {
            fn size_in_bits() -> usize {
                $size
            }

            #[inline(always)]
            fn unchecked_i8(self) -> i8 {
                self as i8
            }

            #[inline(always)]
            fn unchecked_u8(self) -> u8 {
                self as u8
            }

            #[inline(always)]
            fn unchecked_i16(self) -> i16 {
                self as i16
            }

            #[inline(always)]
            fn unchecked_u16(self) -> u16 {
                self as u16
            }

            #[inline(always)]
            fn unchecked_i32(self) -> i32 {
                self as i32
            }

            #[inline(always)]
            fn unchecked_u32(self) -> u32 {
                self as u32
            }

            #[inline(always)]
            fn unchecked_i64(self) -> i64 {
                self as i64
            }

            #[inline(always)]
            fn unchecked_u64(self) -> u64 {
                self as u64
            }
        }
    };
}

simd_suit!(u8, 8);
simd_suit!(i8, 8);
simd_suit!(u16, 16);
simd_suit!(i16, 16);
simd_suit!(u32, 32);
simd_suit!(i32, 32);
simd_suit!(i64, 64);
simd_suit!(u64, 64);
