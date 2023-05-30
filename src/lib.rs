pub use crate::simd::SIMDField;

mod simd;

pub fn binary_search_auto<T: SIMDField>(nums: &[T], target: T) -> Option<usize> {
    let len = nums.len();
    let field_size = T::size_in_bits();
    let total_size = len as u64 * field_size as u64;
    match total_size {
        total_size if total_size <= 128 * 1024 => simd::linear_search(nums, target),
        _ => simd::binary_search(nums, target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8s_bst() {
        for size in 1..=u8::MAX {
            let nums = (0..size).into_iter().collect::<Vec<_>>();
            for target in 0..size {
                let res = binary_search_auto(&nums, target);
                assert!(res.is_some());
                assert_eq!(res.unwrap(), target as usize);
            }
        }
    }

    #[test]
    fn test_i8s_bst() {
        for size in 1..=u8::MAX {
            let half = (size / 2) as i8;
            let i8s = (0 - half..0 + half).into_iter().collect::<Vec<_>>();
            let indexs = i8s.iter().enumerate().collect::<Vec<_>>();
            for (idx, target) in indexs {
                let ans = binary_search_auto(&i8s, *target);
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
                let res = binary_search_auto(&nums, target);
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
                let ans = binary_search_auto(&i16s, *target);
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
                let res = binary_search_auto(&nums, target);
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
                let ans = binary_search_auto(&i16s, *target);
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
                let res = binary_search_auto(&nums, target);
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
                let ans = binary_search_auto(&i64s, *target);
                assert!(ans.is_some());
                assert_eq!(ans.unwrap(), idx);
            }
        }
    }
}
