#![allow(non_upper_case_globals)]

#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;

use criterion::{black_box, Bencher, BenchmarkId, Criterion};

use bst_rs::*;

lazy_static! {
    static ref U8x4: Vec<u8> = gen_u8s(4);
    static ref U8x16: Vec<u8> = gen_u8s(16);
    static ref U8x128: Vec<u8> = gen_u8s(128);
    //
    static ref U16x4: Vec<u16> = gen_u16s(4);
    static ref U16x16: Vec<u16> = gen_u16s(16);
    static ref U16x128: Vec<u16> = gen_u16s(128);
    static ref U16x512: Vec<u16> = gen_u16s(512);
    static ref U16x2048: Vec<u16> = gen_u16s(2048);
    static ref U16x8192: Vec<u16> = gen_u16s(8192);
    //
    static ref U32x4: Vec<u32> = gen_u32s(4);
    static ref U32x16: Vec<u32> = gen_u32s(16);
    static ref U32x128: Vec<u32> = gen_u32s(128);
    static ref U32x512: Vec<u32> = gen_u32s(512);
    static ref U32x2048: Vec<u32> = gen_u32s(2048);
    static ref U32x8192: Vec<u32> = gen_u32s(8192);
    //
    static ref U64x4: Vec<u64> = gen_u64s(4);
    static ref U64x16: Vec<u64> = gen_u64s(16);
    static ref U64x128: Vec<u64> = gen_u64s(128);
    static ref U64x512: Vec<u64> = gen_u64s(512);
    static ref U64x2048: Vec<u64> = gen_u64s(2048);
    static ref U64x8192: Vec<u64> = gen_u64s(8192);

}

fn gen_u8s(size: usize) -> Vec<u8> {
    assert!(size <= u8::MAX as usize);
    (0..size as u8).into_iter().collect::<Vec<_>>()
}

fn gen_u16s(size: usize) -> Vec<u16> {
    assert!(size <= u16::MAX as usize);
    (0..size as u16).into_iter().collect::<Vec<_>>()
}

fn gen_u32s(size: usize) -> Vec<u32> {
    assert!(size <= u32::MAX as usize);
    (0..size as u32).into_iter().collect::<Vec<_>>()
}

fn gen_u64s(size: usize) -> Vec<u64> {
    (0..size as u64).into_iter().collect::<Vec<_>>()
}

fn do_simd_bench<T: SIMDField>(b: &mut Bencher, nums: &[T]) {
    let last = nums.last().unwrap();
    let last = *last;
    b.iter(|| {
        black_box(binary_search_auto(&nums, last).is_some());
    });
}

fn do_std_bench<T: num::Integer + num::FromPrimitive>(b: &mut Bencher, nums: &[T]) {
    let last = nums.last().unwrap();
    b.iter(|| black_box(nums.binary_search(last).is_ok()));
}

fn optimize_bst_bench(c: &mut Criterion, label: &str) {
    let mut group = c.benchmark_group(label);
    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(10));
    group.bench_with_input(
        BenchmarkId::new("optimize_on_8bit", 4),
        &**U8x4,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_8bit", 16),
        &**U8x16,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_8bit", 128),
        &**U8x128,
        do_simd_bench,
    );
    //
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 4),
        &**U16x4,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 16),
        &**U16x16,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 128),
        &**U16x128,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 512),
        &**U16x512,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 2048),
        &**U16x2048,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_16bit", 8192),
        &**U16x8192,
        do_simd_bench,
    );
    //
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 4),
        &**U32x4,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 16),
        &**U32x16,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 128),
        &**U32x128,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 512),
        &**U32x512,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 2048),
        &**U32x2048,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_32bit", 8192),
        &**U32x8192,
        do_simd_bench,
    );
    //
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 4),
        &**U64x4,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 16),
        &**U64x16,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 128),
        &**U64x128,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 512),
        &**U64x512,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 2048),
        &**U64x2048,
        do_simd_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("optimize_on_64bit", 8192),
        &**U64x8192,
        do_simd_bench,
    );
    group.finish();
}

fn std_bst_bench(c: &mut Criterion, label: &str) {
    let mut group = c.benchmark_group(label);
    group
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(10));
    group.bench_with_input(BenchmarkId::new("std_on_8bit", 4), &**U8x4, do_std_bench);
    group.bench_with_input(BenchmarkId::new("std_on_8bit", 16), &**U8x16, do_std_bench);
    group.bench_with_input(
        BenchmarkId::new("std_on_8bit", 128),
        &**U8x128,
        do_std_bench,
    );
    //
    //
    group.bench_with_input(BenchmarkId::new("std_on_16bit", 4), &**U16x4, do_std_bench);
    group.bench_with_input(
        BenchmarkId::new("std_on_16bit", 16),
        &**U16x16,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_16bit", 128),
        &**U16x128,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_16bit", 512),
        &**U16x512,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_16bit", 2048),
        &**U16x2048,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_16bit", 8192),
        &**U16x8192,
        do_std_bench,
    );
    //
    group.bench_with_input(BenchmarkId::new("std_on_32bit", 4), &**U32x4, do_std_bench);
    group.bench_with_input(
        BenchmarkId::new("std_on_32bit", 16),
        &**U32x16,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_32bit", 128),
        &**U32x128,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_32bit", 512),
        &**U32x512,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_32bit", 2048),
        &**U32x2048,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_32bit", 8192),
        &**U32x8192,
        do_std_bench,
    );
    //
    group.bench_with_input(BenchmarkId::new("std_on_64bit", 4), &**U64x4, do_std_bench);
    group.bench_with_input(
        BenchmarkId::new("std_on_64bit", 16),
        &**U64x16,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_64bit", 128),
        &**U64x128,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_64bit", 512),
        &**U64x512,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_64bit", 2048),
        &**U64x2048,
        do_std_bench,
    );
    group.bench_with_input(
        BenchmarkId::new("std_on_64bit", 8192),
        &**U64x8192,
        do_std_bench,
    );
    group.finish();
}

fn bench(c: &mut Criterion) {
    optimize_bst_bench(c, "SIMDS");
    std_bst_bench(c, "std");
}

criterion_group!(benches, bench);
criterion_main!(benches);
