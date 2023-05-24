#[cfg(target_arch = "x86")]
use std::arch::x86::*;
use std::arch::x86_64::{_mm256_loadu_si256, _mm256_set1_epi8, _mm_cmpeq_epi64, _mm_cmpeq_epi8, _mm_load_si128};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::SIMDField;

#[cfg(all(target_feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn linear_search<T: SIMDField>(nums: &[T], target: T) -> Option<usize>
{
    use std::arch::x86_64::*;
    let len = nums.len();
    let field_size = T::size_in_bytes();
    // store nums in 128 bit
    let unit_size = match field_size {
        8 => 128 / 8,
        16 => 128 / 16,
        32 => 128 / 32,
        64 => 128 / 64,
        _ => unreachable!()
    };
    let round = (len / unit_size * 2) / (unit_size * 2);
    let mut i = 0;
    unsafe {
        match field_size {
            8 => linear_8bytes(i, round * 2, nums, target),
            16 => linear_16bytes(i, unit_size, round, nums, target),
            32 => linear_32bytes(i, unit_size, round, nums, target),
            64 => linear_64bytes(i, unit_size, round, nums, target),
            _ => unreachable!()
        }
    }
}

#[cfg(all(target_feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
unsafe fn linear_8bytes<T: SIMDField>(mut from: usize, round: usize, nums: &[T], target: T) -> Option<usize> {
    let keys = _mm_set1_epi8(target.unchecked_i8());
    while from < round {
        let chunk = _mm_loadu_si128((&nums[from..]).as_ptr() as *const _);
        let cmp0 = _mm_cmpeq_epi8(chunk, keys);
        let mask = _mm_movemask_epi8(cmp0);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize);
        }
        from += 16
    }
    linear_search_generic(nums, &target, round)
}

#[cfg(all(target_feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
unsafe fn linear_16bytes<T: SIMDField>(mut from: usize, unit_size: usize, round: usize, nums: &[T], target: T) -> Option<usize> {
    let keys = _mm_set1_epi16(target.unchecked_i16());
    let step = unit_size * 2;
    while from < round {
        let chunk0 = _mm_loadu_si128((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm_loadu_si128((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm_cmpeq_epi16(chunk0, keys);
        let cmp1 = _mm_cmpeq_epi16(chunk1, keys);
        // vector saturating 8
        let packed = _mm_packs_epi16(cmp0, cmp1);
        let mask = _mm_movemask_epi8(packed);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize);
        }
        from += step;
    }
    linear_search_generic(nums, &target, round)
}

#[cfg(all(target_feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
unsafe fn linear_32bytes<T: SIMDField>(mut from: usize, unit_size: usize, round: usize, nums: &[T], target: T) -> Option<usize> {
    let keys = _mm_set1_epi32(target.unchecked_i32());
    let step = unit_size * 2;
    while from < round {
        let chunk0 = _mm_loadu_si128((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm_loadu_si128((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm_cmpeq_epi32(chunk0, keys);
        let cmp1 = _mm_cmpeq_epi32(chunk1, keys);
        // saturating 16
        let packed = _mm_packs_epi32(cmp0, cmp1);
        let mask = _mm_movemask_epi8(packed);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize / 2);
        }

        from += step
    }
    linear_search_generic(nums, &target, round)
}

#[cfg(all(target_feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
unsafe fn linear_64bytes<T: SIMDField>(mut from: usize, unit_size: usize, round: usize, nums: &[T], target: T) -> Option<usize> {
    let keys = _mm_set1_epi64x(target.unchecked_i64());
    let step = unit_size * 2;
    while from < round {
        let chunk0 = _mm_loadu_si128((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm_loadu_si128((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm_cmpeq_epi64(chunk0, keys);
        let cmp1 = _mm_cmpeq_epi64(chunk1, keys);
        // saturating 16 because there is only 0xFFFF or 0
        let packed = _mm_packs_epi32(cmp0, cmp1);
        let mask = _mm_movemask_epi8(packed);
        // 4 byte mapping result which come from comparing 64bytes
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize / 4);
        }

        from += step;
    }
    linear_search_generic(nums, &target, round)
}


// #[cfg(all(target_feature = "avx2", any(target_arch = "x86_64", target_arch = "x86")))]
pub fn linear_search_avx<T: num::Integer + SIMDField>(nums: &[T], target: T) -> isize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = nums.len();
    let field_size = T::size_in_bytes();
    let unit_size = match field_size {
        8 => 128 / 8,
        16 => 128 / 16,
        32 => 128 / 32,
        64 => 128 / 64,
        _ => unreachable!()
    };
    let round = (len / unit_size * 2) / (unit_size * 2);
    let mut i = 0;
    unsafe {
        let keys = match field_size {
            // 8 => {
            //     _mm256_set1_epi8(target as _)
            // }
            // 16 => {
            //     _mm256_set1_epi16(target as _)
            // }
            // 32 => {
            //     _mm256_set1_epi32(target as _)
            // }
            // 64 => {
            //     _mm256_set1_epi64x(target as _)
            // }
            _ => unreachable!()
        };
        while i < round {
            let chunk0 = _mm256_loadu_si256((&nums[i..]).as_ptr() as *const _);
            let chunk1 = _mm256_loadu_si256((&nums[i + unit_size..]).as_ptr() as *const _);
        }
    }
    panic!()
}


#[inline]
fn linear_search_generic<T: num::Integer + SIMDField>(nums: &[T], target: &T, from: usize) -> Option<usize> {
    let mut i = from;
    while i < nums.len() {
        if nums[i] == *target {
            return Some(i);
        }
        i += 1;
    }
    return None;
}

#[cfg(test)]
mod tests {
    use crate::simd::linear::linear_search;

    #[test]
    fn test_u8s_sse_bst() {
        for size in 1..=u8::MAX {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = linear_search(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }

    #[test]
    fn test_i8s_sse_bst() {
        for size in 1..=u8::MAX {
            let half = (size / 2) as i8;
            let i8s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i8s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = linear_search(&i8s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }

    #[test]
    fn test_u16s_bst() {
        for size in 1u16..=1024 {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = linear_search(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }

    #[test]
    fn test_i16s_bst() {
        for size in 1u16..=1024 {
            let half = (size / 2) as i16;
            let i16s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i16s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = linear_search(&i16s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }

    #[test]
    fn test_u32s_bst() {
        for size in 1u32..=1024 {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = linear_search(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }

    #[test]
    fn test_i32s_bst() {
        for size in 1u32..=1024 {
            let half = (size / 2) as i32;
            let i16s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i16s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = linear_search(&i16s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }

    #[test]
    fn test_u64s_bst() {
        for size in 1u64..=1024 {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = linear_search(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }

    #[test]
    fn test_i64s_bst() {
        for size in 1u32..=1024 {
            let half = (size / 2) as i64;
            let i64s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i64s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = linear_search(&i64s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }
}