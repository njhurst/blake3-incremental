//! Demonstration: after a single 64-byte block change in a large file, the
//! full BLAKE3 hash is recomputed from the stored partial (cell) hashes in
//! ~O(cell size) instead of ~O(file size).
//!
//! Run with:  `cargo run --release --example bench [size_mib] [cell_chunks]`
//!
//! Defaults: a 512 MiB file with 1024-chunk (1 MiB) cells. `cell_chunks` must
//! be a power of two. The file lives in memory; the numbers measure hashing
//! work only (no I/O).
//!
//! The full-file rehash baseline uses the official crate's multithreaded
//! `update_rayon` (all cores). The incremental side uses the library API:
//! `refresh` re-hashes only the single dirty cell, and `full_hash` recombines
//! the stored chaining values. The two answers are cross-checked against
//! single-pass BLAKE3 after every phase.

use std::time::{Duration, Instant};

use blake3_incremental::PartialBlake3;

// ---------------------------------------------------------------------------

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::args()
        .nth(
            1 + std::env::args()
                .position(|a| a == name)
                .unwrap_or(usize::MAX),
        )
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// SplitMix64-based content fill, parallelized across cores.
fn fill_content(buf: &mut [u8]) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let per = buf.len().div_ceil(cores);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (w, chunk) in buf.chunks_mut(per).enumerate() {
            handles.push(s.spawn(move || {
                let mut state = 0x243F_6A88_85A3_08D3u64.wrapping_add(w as u64);
                for block in chunk.chunks_mut(8) {
                    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    let bytes = (z ^ (z >> 31)).to_le_bytes();
                    block.copy_from_slice(&bytes[..block.len()]);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });
}

fn mean(times: &[Duration]) -> Duration {
    times.iter().sum::<Duration>() / times.len() as u32
}

fn main() {
    let size_mib: usize = parse_arg("--size", 512);
    let cell_chunks: u64 = parse_arg("--cell-chunks", 1024);
    let trials: usize = parse_arg("--trials", 64);

    let file_len = size_mib * 1024 * 1024;
    let cells = (file_len as u64).div_ceil(cell_chunks * 1024);
    println!("file        : {size_mib} MiB = {file_len} bytes");
    println!(
        "cells       : {cells} x {cell_chunks}-chunk cells ({} KiB each)",
        cell_chunks
    );
    println!(
        "partial state: {} bytes kept after the build pass",
        cells * 32
    );
    println!(
        "cores       : {}",
        std::thread::available_parallelism().unwrap()
    );
    assert!(
        cell_chunks.is_power_of_two(),
        "cell_chunks must be a power of two"
    );

    let mut content = vec![0u8; file_len];
    fill_content(&mut content);

    // ------------------------------------------------------------------
    // Phase 1: baselines. Official single-pass (multithreaded) vs partial
    // build (one cell CV at a time, same primitives).
    // ------------------------------------------------------------------
    let mut full_times = Vec::new();
    let mut build_times = Vec::new();
    let mut build_par_times = Vec::new();
    let mut partial = None;
    for pass in 0..3 {
        let t = Instant::now();
        let mut hasher = blake3::Hasher::new();
        hasher.update_rayon(&content);
        let full = hasher.finalize();
        full_times.push(t.elapsed());

        let t = Instant::now();
        let p = PartialBlake3::build(&content, cell_chunks).unwrap();
        build_times.push(t.elapsed());
        assert_eq!(p.full_hash(&content), full, "partial build mismatch");

        let t = Instant::now();
        let p = PartialBlake3::build_parallel(&content, cell_chunks, 0).unwrap();
        build_par_times.push(t.elapsed());
        assert_eq!(
            p.full_hash(&content),
            full,
            "parallel partial build mismatch"
        );
        if pass == 0 {
            println!("\nphase 1: build partials (serial + parallel) and verify root hashes vs single pass ... ok");
        }
        partial = Some(p);
    }
    let mut partial = partial.unwrap();
    let full_time = mean(&full_times[1..]);
    let build_time = mean(&build_times[1..]);
    let build_par_time = mean(&build_par_times[1..]);
    println!(
        "full single pass (update_rayon): {:>9.2} ms  ({:.2} GiB/s)",
        full_time.as_secs_f64() * 1e3,
        file_len as f64 / full_time.as_secs_f64() / 2f64.powi(30)
    );
    println!(
        "partial build serial ({} cells)  : {:>9.2} ms  ({:.2} GiB/s)",
        cells,
        build_time.as_secs_f64() * 1e3,
        file_len as f64 / build_time.as_secs_f64() / 2f64.powi(30)
    );
    println!(
        "partial build parallel (1/cell) : {:>9.2} ms  ({:.2} GiB/s)",
        build_par_time.as_secs_f64() * 1e3,
        file_len as f64 / build_par_time.as_secs_f64() / 2f64.powi(30)
    );

    // ------------------------------------------------------------------
    // Phase 2: single-block edits. Each trial changes one 64-byte block at
    // a fresh offset (rotating through the file's cells), then recomputes
    // the full hash from the partial pieces: refresh() re-hashes the one
    // dirty cell, full_hash() recombines stored CVs.
    // ------------------------------------------------------------------
    let block_offsets: Vec<usize> = (0..trials)
        .map(|i| (i * 7 * 64 * 1024 + (i * 8191) % (64 * 1024)) % (file_len - 64))
        .collect();

    let mut recompute_times = Vec::new();
    for &off in &block_offsets {
        content[off..off + 64].fill((off >> 8) as u8);
        let t = Instant::now();
        partial.refresh(&content, off as u64..(off + 64) as u64);
        let h = partial.full_hash(&content);
        recompute_times.push(t.elapsed());

        // Verify against single-pass BLAKE3 for this exact content, but only
        // occasionally: a full pass is exactly what we're trying to avoid.
        if off == block_offsets[0] || off == *block_offsets.last().unwrap() {
            assert_eq!(h, blake3::hash(&content), "refresh mismatch");
        }
    }
    let recompute_time = mean(&recompute_times);
    println!(
        "\nphase 2: {} single-block edits, each recomputed from partials ... ok",
        trials
    );

    // ------------------------------------------------------------------
    // Phase 3: the same final file content, but hashed from scratch with a
    // full single pass. This is the "naive" cost of one edit's recompute.
    // ------------------------------------------------------------------
    let t = Instant::now();
    let mut hasher = blake3::Hasher::new();
    hasher.update_rayon(&content);
    let naive = hasher.finalize();
    let naive_time = t.elapsed();
    assert_eq!(partial.full_hash(&content), naive, "final state mismatch");
    println!("phase 3: full rehash of the edited file ... ok\n");

    let dirty_bytes = cell_chunks * 1024;
    println!("{:<42}{:>12}{:>14}", "", "time", "speed");
    println!(
        "{:<42}{:>10.3} ms",
        "full rehash after 1 block edit (all cores)",
        naive_time.as_secs_f64() * 1e3
    );
    println!(
        "{:<42}{:>10.3} ms{:>12.1}x",
        "recompute from partial pieces (refresh 1 cell + merge)",
        recompute_time.as_secs_f64() * 1e3,
        naive_time.as_secs_f64() / recompute_time.as_secs_f64()
    );
    println!(
        "{:<42}{:>10.1} MiB {:>10.1}x",
        "bytes re-hashed per edit (vs whole file)",
        dirty_bytes as f64 / 2f64.powi(20),
        file_len as f64 / dirty_bytes as f64
    );
    println!(
        "\nstate kept after build: {:.2} KiB (vs {:.1} MiB of file data)",
        cells * 32 / 1024,
        file_len as f64 / 2f64.powi(20)
    );
}
