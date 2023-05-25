#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::SIMDField;

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
pub fn binary_search<T: SIMDField>(nums: &[T], target: T) -> Option<usize> {
    let end = nums.len() - 1;
    let left = 0;
    let right = end;
    let field_size = T::size_in_bits();
    unsafe {
        match field_size {
            8 => bst_8bits(nums, target, left, right),
            16 => bst_16bits(nums, target, left, right),
            32 => bst_32bits(nums, target, left, right),
            64 => bst_64bits(nums, target, left, right),
            _ => unreachable!(),
        }
    }
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
unsafe fn bst_8bits<T: SIMDField>(
    nums: &[T],
    target: T,
    mut left: usize,
    mut right: usize,
) -> Option<usize> {
    #[cfg(all(feature = "use-sse", target_feature = "sse"))]
    let keys = _mm_set1_epi8(target.unchecked_i8());
    #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
    let keys = _mm256_set1_epi8(target.unchecked_i8());
    let end = nums.len() - 1;
    while left <= right {
        let pivot = left + (right - left);
        if nums[pivot] == target {
            return Some(pivot);
        }
        if target < nums[pivot] {
            right = pivot - 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if right >= 16 {
                    let v = _mm_loadu_si128((&nums[right - 16..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi8(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 16 + mask.trailing_zeros() as usize);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if right >= 32 {
                    let v = _mm256_loadu_si256((&nums[right - 32..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi8(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 32 + mask.trailing_zeros() as usize);
                    }
                }
            }
        } else {
            left = pivot + 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if left + 16 < end {
                    let v = _mm_loadu_si128((&nums[left..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi8(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if left + 32 < end {
                    let v = _mm256_loadu_si256((&nums[left..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi8(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize);
                    }
                }
            }
        }
    }
    None
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
unsafe fn bst_16bits<T: SIMDField>(
    nums: &[T],
    target: T,
    mut left: usize,
    mut right: usize,
) -> Option<usize> {
    #[cfg(all(feature = "use-sse", target_feature = "sse"))]
    let keys = _mm_set1_epi16(target.unchecked_i16());
    #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
    let keys = _mm256_set1_epi16(target.unchecked_i16());
    let end = nums.len() - 1;
    while left <= right {
        let pivot = left + (right - left);
        if nums[pivot] == target {
            return Some(pivot);
        }
        if target < nums[pivot] {
            right = pivot - 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if right >= 8 {
                    let v = _mm_loadu_si128((&nums[right - 8..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi16(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 8 + mask.trailing_zeros() as usize / 2);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if right >= 16 {
                    let v = _mm256_loadu_si256((&nums[right - 16..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi16(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 16 + mask.trailing_zeros() as usize / 2);
                    }
                }
            }
        } else {
            left = pivot + 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if left + 8 < end {
                    let v = _mm_loadu_si128((&nums[left..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi16(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 2);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if left + 16 < end {
                    let v = _mm256_loadu_si256((&nums[left..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi16(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 2);
                    }
                }
            }
        }
    }
    None
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
unsafe fn bst_32bits<T: SIMDField>(
    nums: &[T],
    target: T,
    mut left: usize,
    mut right: usize,
) -> Option<usize> {
    #[cfg(all(feature = "use-sse", target_feature = "sse"))]
    let keys = _mm_set1_epi32(target.unchecked_i32());
    #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
    let keys = _mm256_set1_epi32(target.unchecked_i32());
    let end = nums.len() - 1;
    while left <= right {
        let pivot = left + (right - left);
        if nums[pivot] == target {
            return Some(pivot);
        }
        if target < nums[pivot] {
            right = pivot - 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if right >= 4 {
                    let v = _mm_loadu_si128((&nums[right - 4..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi32(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 4 + mask.trailing_zeros() as usize / 4);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if right >= 8 {
                    let v = _mm256_loadu_si256((&nums[right - 8..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi32(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 8 + mask.trailing_zeros() as usize / 4);
                    }
                }
            }
        } else {
            left = pivot + 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if left + 4 < end {
                    let v = _mm_loadu_si128((&nums[left..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi8(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 4);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if left + 8 < end {
                    let v = _mm256_loadu_si256((&nums[left..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi32(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 4);
                    }
                }
            }
        }
    }
    None
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
unsafe fn bst_64bits<T: SIMDField>(
    nums: &[T],
    target: T,
    mut left: usize,
    mut right: usize,
) -> Option<usize> {
    #[cfg(all(feature = "use-sse", target_feature = "sse"))]
    let keys = _mm_set1_epi64x(target.unchecked_i64());
    #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
    let keys = _mm256_set1_epi64x(target.unchecked_i64());
    let end = nums.len() - 1;
    while left <= right {
        let pivot = left + (right - left);
        if nums[pivot] == target {
            return Some(pivot);
        }
        if target < nums[pivot] {
            right = pivot - 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if right >= 2 {
                    let v = _mm_loadu_si128((&nums[right - 2..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi64(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 2 + mask.trailing_zeros() as usize / 8);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if right >= 4 {
                    let v = _mm256_loadu_si256((&nums[right - 4..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi64(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(right - 4 + mask.trailing_zeros() as usize / 8);
                    }
                }
            }
        } else {
            left = pivot + 1;
            #[cfg(all(feature = "use-sse", target_feature = "sse"))]
            {
                if left + 2 < end {
                    let v = _mm_loadu_si128((&nums[left..]).as_ptr() as *const _);
                    let v = _mm_cmpeq_epi64(v, keys);
                    let mask = _mm_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 8);
                    }
                }
            }
            #[cfg(all(feature = "use-avx2", target_feature = "avx2"))]
            {
                if left + 4 < end {
                    let v = _mm256_loadu_si256((&nums[left..]).as_ptr() as *const _);
                    let v = _mm256_cmpeq_epi64(v, keys);
                    let mask = _mm256_movemask_epi8(v);
                    if mask != 0 {
                        return Some(left + mask.trailing_zeros() as usize / 8);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::simd::simd_bst::binary_search;

    #[test]
    fn test_u8s_sse_bst() {
        for size in 1..=u8::MAX {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = binary_search(&nums, target);
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
                let ans = binary_search(&i8s, *target);
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
                let res = binary_search(&nums, target);
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
                let ans = binary_search(&i16s, *target);
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
                let res = binary_search(&nums, target);
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
                let ans = binary_search(&i16s, *target);
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
                let res = binary_search(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }
    //
    #[test]
    fn test_i64s_bst() {
        for size in 1u32..=1024 {
            let half = (size / 2) as i64;
            let i64s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i64s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = binary_search(&i64s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }
}
