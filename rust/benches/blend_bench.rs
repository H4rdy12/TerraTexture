//! Benchmark: serial vs parallel, across a size sweep, for both kernels
//! in `src/lib.rs`. Run with `cargo bench` from the `rust/` directory.
//!
//! This exists to answer one question: at what array size does
//! `par_for_each` actually start winning over a plain serial loop on
//! real hardware? `PARALLEL_THRESHOLD` in `src/lib.rs` is currently a
//! guessed placeholder -- after running this, look at the criterion
//! HTML report (target/criterion/report/index.html) for the size at
//! which the `parallel` line drops below the `serial` line, and update
//! `PARALLEL_THRESHOLD` to that many elements (H*W).
//!
//! Deliberately calls `soft_light_serial`/`soft_light_parallel` (and
//! the luminosity_blend equivalents) directly, NOT `soft_light_core`,
//! so the sweep isn't circularly gated by the very threshold it's
//! trying to determine.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use terra_texture_rs::{
    luminosity_blend_parallel, luminosity_blend_serial, soft_light_parallel, soft_light_serial,
};

// Chosen to bracket a wide range around the guessed 65_536-element
// PARALLEL_THRESHOLD (side lengths 16..2048 -> 256 .. ~4.2M elements),
// so the sweep should visibly cross the real threshold somewhere in
// this range regardless of how far off the current guess is.
const SIDES: &[usize] = &[16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];

fn bench_soft_light(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_light");
    for &side in SIDES {
        let n = side * side;
        let a = Array2::<f32>::from_elem((side, side), 0.4);
        let b = Array2::<f32>::from_elem((side, side), 0.6);
        let mut out = Array2::<f32>::zeros((side, side));

        group.bench_with_input(BenchmarkId::new("serial", n), &n, |bencher, _| {
            bencher.iter(|| {
                soft_light_serial(a.view(), b.view(), &mut out);
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |bencher, _| {
            bencher.iter(|| {
                soft_light_parallel(a.view(), b.view(), &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_luminosity_blend(c: &mut Criterion) {
    let mut group = c.benchmark_group("luminosity_blend");
    for &side in SIDES {
        let n = side * side;
        let backdrop = Array2::<f32>::from_elem((side, side), 0.5)
            .insert_axis(ndarray::Axis(2))
            .broadcast((side, side, 3))
            .unwrap()
            .to_owned();
        let luminosity = Array2::<f32>::from_elem((side, side), 0.6);
        let mut out = ndarray::Array3::<f32>::zeros((side, side, 3));

        group.bench_with_input(BenchmarkId::new("serial", n), &n, |bencher, _| {
            bencher.iter(|| {
                luminosity_blend_serial(backdrop.view(), luminosity.view(), &mut out);
                black_box(&out);
            });
        });
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |bencher, _| {
            bencher.iter(|| {
                luminosity_blend_parallel(backdrop.view(), luminosity.view(), &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_soft_light, bench_luminosity_blend);
criterion_main!(benches);
