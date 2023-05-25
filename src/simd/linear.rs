#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::SIMDField;

#[cfg(feature = "use-sse")]
fn sse_round<T: SIMDField>() -> usize {
    match T::size_in_bits() {
        8 => 128 / 8,
        16 => 128 / 16,
        32 => 128 / 32,
        64 => 128 / 64,
        _ => unreachable!(),
    }
}

#[cfg(feature = "use-avx2")]
fn avx_round<T: SIMDField>() -> usize {
    match T::size_in_bits() {
        8 => 256 / 8,
        16 => 256 / 16,
        32 => 256 / 32,
        64 => 256 / 64,
        _ => unreachable!(),
    }
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
pub fn linear_search<T: SIMDField>(nums: &[T], target: T) -> Option<usize> {
    let len = nums.len();
    let field_size = T::size_in_bits();
    // store nums in 128 bit
    #[cfg(all(feature = "use-sse", target_feature = "sse"))]
    let unit_size = sse_round::<T>();
    #[cfg(all(feature = "use-avx2", target_feature = "avx"))]
    let unit_size = avx_round::<T>();
    let round = (len / unit_size * 2) / (unit_size * 2);
    let i = 0;
    unsafe {
        #[cfg(all(feature = "use-sse", target_feature = "sse"))]
        {
            match field_size {
                8 => linear_8bits_sse(i, round * 2, nums, target),
                16 => linear_16bits_sse(i, unit_size, round, nums, target),
                32 => linear_32bits_sse(i, unit_size, round, nums, target),
                64 => linear_64bits_sse(i, unit_size, round, nums, target),
                _ => unreachable!(),
            }
        }
        #[cfg(all(feature = "use-avx2", target_feature = "avx"))]
        {
            match field_size {
                8 => linear_8bits_avx(i, round * 2, nums, target),
                16 => linear_16bits_avx(i, unit_size, round, nums, target),
                32 => linear_32bits_avx(i, unit_size, round, nums, target),
                64 => linear_64bits_avx(i, unit_size, round, nums, target),
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(all(
    feature = "use-sse",
    target_feature = "sse",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline(always)]
unsafe fn linear_8bits_sse<T: SIMDField>(
    mut from: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
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

#[cfg(all(
    feature = "use-sse",
    target_feature = "sse",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline(always)]
unsafe fn linear_16bits_sse<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
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

#[cfg(all(
    feature = "use-sse",
    target_feature = "sse",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline(always)]
unsafe fn linear_32bits_sse<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
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

#[cfg(all(
    feature = "use-sse",
    target_feature = "sse",
    any(target_arch = "x86_64", target_arch = "x86")
))]
#[inline(always)]
unsafe fn linear_64bits_sse<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
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

#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn linear_8bits_avx<T: SIMDField>(
    mut from: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
    let keys = _mm256_set1_epi8(target.unchecked_i8());
    while from < round {
        let chunk = _mm256_loadu_si256((&nums[from..]).as_ptr() as *const _);
        let cmp0 = _mm256_cmpeq_epi8(chunk, keys);
        let mask = _mm256_movemask_epi8(cmp0);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize);
        }
        from += 16
    }
    linear_search_generic(nums, &target, round)
}

#[allow(dead_code)]
#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn dump_avx_epu8(v: __m256i) {
    let tmp = [0u8; 32];
    _mm256_storeu_si256((&tmp[..]).as_ptr() as _, v);
    print!("[ {:#010b} ", tmp[0]);
    for i in 1..32 {
        print!("| {:#010b}", tmp[i]);
    }
    print!(" ]\n");
}

#[allow(dead_code)]
#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn dump_avx_epu16(v: __m256i) {
    let tmp = [0u16; 16];
    _mm256_storeu_si256((&tmp[..]).as_ptr() as _, v);
    print!("[ {:#018b} ", tmp[0]);
    for i in 1..16 {
        print!("| {:#018b}", tmp[i]);
    }
    print!(" ]\n");
}

#[allow(dead_code)]
#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn dump_avx_epu32(v: __m256i) {
    let tmp = [0u32; 8];
    _mm256_storeu_si256((&tmp[..]).as_ptr() as _, v);
    print!("[ {:#034b} ", tmp[0]);
    for i in 1..8 {
        print!("| {:#034b}", tmp[i]);
    }
    print!(" ]\n");
}

#[allow(dead_code)]
#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn dump_avx_epu64(v: __m256i) {
    let tmp = [0u64; 4];
    _mm256_storeu_si256((&tmp[..]).as_ptr() as _, v);
    print!("[ {:#066b} ", tmp[0]);
    for i in 1..4 {
        print!("| {:#066b}", tmp[i]);
    }
    print!(" ]\n");
}

#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn linear_16bits_avx<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
    let keys = _mm256_set1_epi16(target.unchecked_i16());
    let shuffle = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    let step = unit_size * 2;
    while from < round {
        let chunk0 = _mm256_loadu_si256((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm256_loadu_si256((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm256_cmpeq_epi16(chunk0, keys);
        // [0x0000 0x0000 0000 0000] [0000 0000 0000 0000] [0xFFFF 0000 0000 0000] [0000 0000 0000 0000]
        let cmp1 = _mm256_cmpeq_epi16(chunk1, keys);
        // twisting, saturating vector 8
        let packed = _mm256_packs_epi16(cmp0, cmp1);
        let SHUFFLED = _mm256_permutevar8x32_epi32(packed, shuffle);
        let mask = _mm256_movemask_epi8(SHUFFLED);
        // [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0] [16 0]
        // println!("mask {:#034b}", mask);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize);
        }
        from += step;
    }
    linear_search_generic(nums, &target, round)
}

#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn linear_32bits_avx<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
    let keys = _mm256_set1_epi32(target.unchecked_i32());
    let step = unit_size * 2;
    let shuffle = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    while from < round {
        let chunk0 = _mm256_loadu_si256((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm256_loadu_si256((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm256_cmpeq_epi32(chunk0, keys);
        // print!("cmp0: ");
        // dump_avx_epu32(cmp0);
        let cmp1 = _mm256_cmpeq_epi32(chunk1, keys);
        // print!("cmp1: ");
        // dump_avx_epu32(cmp1);
        // twisting, saturating 16
        let packed = _mm256_packs_epi32(cmp0, cmp1);
        // print!("packed: ");
        // dump_avx_epu16(packed);
        // let shuffled = _mm256_permute4x64_epi64::<SHUFFLE>(packed);
        let shuffled = _mm256_permutevar8x32_epi32(packed, shuffle);
        // print!("shuffled: ");
        // dump_avx_epu8(shuffled);
        let mask = _mm256_movemask_epi8(shuffled);
        // println!("{:#034b}", mask);
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize / 2);
        }

        from += step
    }
    linear_search_generic(nums, &target, round)
}

#[cfg(all(
    feature = "use-avx2",
    target_feature = "avx2",
    any(target_arch = "x86_64", target_arch = "x86")
))]
unsafe fn linear_64bits_avx<T: SIMDField>(
    mut from: usize,
    unit_size: usize,
    round: usize,
    nums: &[T],
    target: T,
) -> Option<usize> {
    let keys = _mm256_set1_epi64x(target.unchecked_i64());
    let step = unit_size * 2;
    let shuffle = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    while from < round {
        let chunk0 = _mm256_loadu_si256((&nums[from..]).as_ptr() as *const _);
        let chunk1 = _mm256_loadu_si256((&nums[from + unit_size..]).as_ptr() as *const _);
        let cmp0 = _mm256_cmpeq_epi64(chunk0, keys);
        let cmp1 = _mm256_cmpeq_epi64(chunk1, keys);
        // 64bits saturating 16 because there is only 0xFFFF or 0
        let packed = _mm256_packs_epi32(cmp0, cmp1);
        let shuffled = _mm256_permutevar8x32_epi32(packed, shuffle);
        let mask = _mm256_movemask_epi8(shuffled);
        // 4 byte mapping result which come from comparing 64bytes
        if mask != 0 {
            return Some(from + mask.trailing_zeros() as usize / 4);
        }

        from += step;
    }
    linear_search_generic(nums, &target, round)
}

#[inline]
pub fn linear_search_generic<T: num::Integer + SIMDField>(
    nums: &[T],
    target: &T,
    from: usize,
) -> Option<usize> {
    let mut i = from;
    while i < nums.len() {
        if nums[i] == *target {
            return Some(i);
        }
        i += 1;
    }
    return None;
}

#[cfg(all(
    any(feature = "use-sse", feature = "use-avx2"),
    all(
        any(target_feature = "sse", target_feature = "avx2"),
        any(target_arch = "x86_64", target_arch = "x86")
    )
))]
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
    //
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
