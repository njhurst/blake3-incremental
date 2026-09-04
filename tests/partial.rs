//! Randomized tester: compares the incremental `PartialBlake3` against
//! single-pass official BLAKE3 (`blake3::hash`) over a wide range of file
//! sizes, cell sizes, and in-place edit sequences.
//!
//! All randomness is deterministic (SplitMix64). Heavy test loops partition
//! their work across all available cores; each task derives its own seed from
//! its index, so a failing task is reproducible (the assertion message
//! carries the task index and parameters).
//!
//! Run with `cargo test` (the blake3 dependency is compiled with
//! optimizations even in dev builds, so this stays fast).

use blake3::hazmat::ChainingValue;
use blake3_incremental::subtree_cv;
use blake3_incremental::{CellSizeError, PartialBlake3};

// ---------------------------------------------------------------------------
// Deterministic RNG
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

fn random_content(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    rng.fill(&mut v);
    // Sprinkle in non-random patterns (all-same bytes), which stress chunk/CV
    // coincidence edges cheaply.
    if len > 0 && rng.next_u64() & 3 == 0 {
        let byte = (rng.next_u64() as u8).max(1);
        v.fill(byte);
    }
    v
}

/// Per-task deterministic seed: tasks are (re)runnable in isolation.
fn task_seed(base: u64, task: u64) -> u64 {
    base.wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(task.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Expected cell count: ceil(file_len / (P * 1024)).
fn expected_cells(file_len: u64, cell_chunks: u64) -> usize {
    file_len.div_ceil(cell_chunks * 1024) as usize
}

/// The one assertion that matters: incremental == official single pass.
fn assert_hash_matches(partial: &PartialBlake3, content: &[u8], context: &str) {
    let got = partial.full_hash(content);
    let want = blake3::hash(content);
    assert_eq!(
        got,
        want,
        "hash mismatch ({context}): file_len={} cell_chunks={} cells={}",
        partial.file_len(),
        partial.cell_chunks(),
        partial.cell_count(),
    );
}

fn assert_consistent_state(partial: &PartialBlake3, content: &[u8], context: &str) {
    assert_eq!(
        partial.file_len(),
        content.len() as u64,
        "stale file_len ({context})"
    );
    assert_eq!(
        partial.cell_count(),
        expected_cells(content.len() as u64, partial.cell_chunks()),
        "wrong cell count ({context})"
    );
    assert_hash_matches(partial, content, context);
}

/// Run `task(i)` for every `i in 0..n`, partitioned across the available
/// cores. Task functions must be deterministic in `i` alone (each task seeds
/// its own RNG). A panic in any task fails the test with its message.
fn parallel_for(n: usize, task: impl Fn(usize) + Sync + Send) {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let workers = cores.min(n.max(1));
    if workers <= 1 || n <= 1 {
        for i in 0..n {
            task(i);
        }
        return;
    }
    std::thread::scope(|s| {
        let task = &task; // shared by reference: `task: Sync`, and `&T` is Send + Copy
        let mut handles = Vec::with_capacity(workers);
        let mut start = 0;
        for _ in 0..workers {
            let end = if handles.len() + 1 == workers {
                n
            } else {
                (handles.len() + 1) * n / workers
            };
            handles.push(s.spawn(move || {
                for i in start..end {
                    task(i);
                }
            }));
            start = end;
        }
        for h in handles {
            if let Err(payload) = h.join() {
                std::panic::resume_unwind(payload);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 1. Exhaustive small lengths. With P in {1,2,4} and len <= 700 everything is
//    at most one cell, so this exercises the <=1-cell content path, the empty
//    file, and tiny final chunks.
// ---------------------------------------------------------------------------

#[test]
fn exhaustive_small_lengths() {
    let mut tasks = Vec::new();
    for &cell_chunks in &[1u64, 2, 4] {
        for len in 0..=700usize {
            tasks.push((cell_chunks, len));
        }
    }
    let base = 0x51ED_2701_5E6A_5D4F;
    parallel_for(tasks.len(), |t| {
        let (cell_chunks, len) = tasks[t];
        let content = random_content(&mut Rng::new(task_seed(base, t as u64)), len);
        let partial = PartialBlake3::build(&content, cell_chunks).expect("valid cell size");
        assert_eq!(
            partial.cell_count(),
            expected_cells(len as u64, cell_chunks),
            "task {t}: len={len} P={cell_chunks}"
        );
        assert_consistent_state(
            &partial,
            &content,
            &format!("task {t}: exhaustive len={len} P={cell_chunks}"),
        );
    });
}

// ---------------------------------------------------------------------------
// 2. Boundary lengths: chunk/cell-size transitions, exact multiples, short
//    final chunks, powers of two and neighbors. Main stress test for the
//    cell-merging logic. A length is `chunks` full chunks plus a final chunk
//    of `tail` bytes (tail = 0 means an exact multiple of 1024).
// ---------------------------------------------------------------------------

#[test]
fn boundary_lengths() {
    let mut tasks: Vec<(u64, u64, u64)> = Vec::new(); // (P, chunks, tail)
    for &cell_chunks in &[1u64, 2, 4, 16, 1024] {
        let p = cell_chunks as usize;
        let mut lens: Vec<usize> = Vec::new();

        // Dense small chunk counts (up to a couple of cells for small P; for
        // P = 1024 these are all sub-cell and only exercise the content path,
        // so a short sweep suffices).
        lens.extend(0..=if p <= 16 { 200 } else { 16 });

        // Multi-cell boundaries: counts just below / at / above multiples of
        // P, including the N % P == 0 transitions where the last cell is
        // exactly full.
        let j_max = if p >= 1024 { 8 } else { 24 };
        for j in 1..=j_max {
            lens.push(j * p);
            lens.push(j * p + 1);
            lens.push(j * p - 1);
        }

        // Perfect-tree sizes (power-of-two chunk counts) and neighbors, up to
        // 2048 chunks (2 MiB).
        let mut pow2 = 1usize;
        while pow2 <= 2048 {
            for delta in [0usize, 1, 3] {
                lens.push(pow2 + delta);
                if pow2 > delta {
                    lens.push(pow2 - delta);
                }
            }
            pow2 *= 2;
        }

        lens.sort_unstable();
        lens.dedup();

        // Final-chunk lengths to try per chunk count (a final chunk holds
        // 1..=1024 bytes; 1024 => tail = 0 here).
        let tails: &[u64] = if cell_chunks >= 1024 {
            &[0, 1, 1023]
        } else {
            &[0, 1, 63, 64, 65, 1023]
        };
        for &chunks in &lens {
            for &tail in tails {
                tasks.push((cell_chunks, chunks as u64, tail));
            }
        }
    }

    let base = 0x6E2A_3B4C_1D0F_9E8D;
    parallel_for(tasks.len(), |t| {
        let (cell_chunks, chunks, tail) = tasks[t];
        let len = if tail == 0 {
            chunks * 1024
        } else {
            chunks.saturating_sub(1) * 1024 + tail
        };
        let content = random_content(&mut Rng::new(task_seed(base, t as u64)), len as usize);
        let partial = PartialBlake3::build(&content, cell_chunks).unwrap();
        assert_consistent_state(
            &partial,
            &content,
            &format!("task {t}: boundary len={len} P={cell_chunks}"),
        );
    });
}

// ---------------------------------------------------------------------------
// 3. Random lengths across cell sizes, including fairly large files.
// ---------------------------------------------------------------------------

#[test]
fn random_lengths() {
    const ITERS: u64 = 300;
    const POOL: [u64; 5] = [1, 2, 8, 32, 1024];
    let base = 0xDEAD_BEEF_CAFE_F00D;
    parallel_for(ITERS as usize, |t| {
        let mut rng = Rng::new(task_seed(base, t as u64));
        let cell_chunks = POOL[rng.below(POOL.len() as u64) as usize];
        let tier = rng.below(100);
        let max_len: u64 = if tier < 55 {
            2 * 1024 * 1024
        } else if tier < 85 {
            16 * 1024 * 1024
        } else {
            96 * 1024 * 1024
        };
        let len = rng.below(max_len + 1) as usize;
        let content = random_content(&mut rng, len);
        let partial = PartialBlake3::build(&content, cell_chunks).unwrap();
        assert_consistent_state(
            &partial,
            &content,
            &format!("task {t}: random len={len} P={cell_chunks}"),
        );
    });
}

// ---------------------------------------------------------------------------
// 4. Edit sequences: mutate random in-place ranges, refresh only the dirty
//    cells, and confirm the recomputed hash matches a fresh single pass.
//    Also verifies that cells outside the edited range are bit-for-bit
//    untouched (no hidden full re-hash) and that a from-scratch rebuild over
//    the final content reproduces the same cell state.
// ---------------------------------------------------------------------------

#[test]
fn in_place_edits_match_single_pass() {
    // Each task = one file: (P, number of cells, file index, seed).
    let mut tasks: Vec<(u64, u64, u64)> = Vec::new();
    let configs: &[(u64, u64)] = &[(1, 40), (4, 60), (16, 40), (1024, 6)];
    for (cfg, &(cell_chunks, max_cells)) in configs.iter().enumerate() {
        for file_iter in 0..8 {
            tasks.push((cell_chunks, max_cells, (cfg * 8 + file_iter) as u64));
        }
    }
    let base = 0x0123_4567_89AB_CDEF;
    parallel_for(tasks.len(), |t| {
        let (cell_chunks, max_cells, file_id) = tasks[t];
        let mut rng = Rng::new(task_seed(base, t as u64));
        let n_cells = 2 + rng.below(max_cells - 1); // >= 2 cells: merge path
        let len = n_cells * cell_chunks * 1024 + rng.below(cell_chunks * 1024);
        let mut content = random_content(&mut rng, len as usize);
        let mut partial = PartialBlake3::build(&content, cell_chunks).unwrap();
        assert_consistent_state(
            &partial,
            &content,
            &format!("task {t}: edit build P={cell_chunks} len={len}"),
        );

        let edits = if cell_chunks >= 1024 { 20 } else { 30 };
        for edit in 0..edits {
            // Edit styles: arbitrary small ranges, block-aligned 64-byte
            // writes, ranges straddling a cell boundary, and ranges touching
            // the end of the file (partial final chunk).
            let (offset, edit_len) = match rng.below(4) {
                0 => {
                    let o = rng.below(len);
                    (o, 1 + rng.below(64))
                }
                1 => {
                    let o = rng.below(len / 64) * 64;
                    (o, 64)
                }
                2 => {
                    let o = if rng.below(2) == 0 {
                        cell_chunks * 1024 * rng.below(n_cells)
                    } else {
                        len - (1 + rng.below(2048))
                    };
                    let l = 1 + rng.below(512);
                    (o.min(len - 1), l.min(len - o.min(len - 1)))
                }
                _ => {
                    let o = rng.below(len);
                    (o, (1 + rng.below(4096)).min(len - o))
                }
            };
            let (offset, edit_len) = (offset as usize, edit_len as usize);

            // Save the CVs of cells this edit does *not* intersect.
            let dirty: Vec<bool> = (0..partial.cell_count())
                .map(|m| {
                    let (s, e) = cell_range(&partial, m);
                    s < offset + edit_len && offset < e
                })
                .collect();
            let untouched: Vec<(usize, ChainingValue)> = dirty
                .iter()
                .enumerate()
                .filter(|(_, &d)| !d)
                .map(|(m, _)| (m, partial.cell_cv(m).unwrap()))
                .collect();

            rng.fill(&mut content[offset..offset + edit_len]);
            partial.refresh(&content, offset as u64..(offset + edit_len) as u64);

            // Untouched cells must be bit-for-bit unchanged: this is the
            // whole point of partial hashing.
            for (m, saved) in &untouched {
                assert_eq!(
                    partial.cell_cv(*m),
                    Some(*saved),
                    "cell {m} changed though untouched by edit at {offset}..+{edit_len} \
                     (task {t}: P={cell_chunks} len={len} edit={edit})"
                );
            }
            assert_consistent_state(
                &partial,
                &content,
                &format!("task {t}: edit P={cell_chunks} len={len} file={file_id} edit={edit}"),
            );
        }

        // A from-scratch rebuild over the same content must reproduce the
        // exact same partial state as the accumulated refreshes.
        let rebuilt = PartialBlake3::build(&content, cell_chunks).unwrap();
        assert_eq!(partial.cell_count(), rebuilt.cell_count());
        for m in 0..partial.cell_count() {
            assert_eq!(
                partial.cell_cv(m),
                rebuilt.cell_cv(m),
                "rebuild drift in cell {m} (task {t}: P={cell_chunks})"
            );
        }
    });
}

fn cell_range(partial: &PartialBlake3, m: usize) -> (usize, usize) {
    let cell_bytes = partial.cell_bytes() as usize;
    let s = m * cell_bytes;
    let e = (s + cell_bytes).min(partial.file_len() as usize);
    (s, e)
}

// ---------------------------------------------------------------------------
// 5. refresh() is for in-place edits of unchanged length: length changes
//    must go through resize() (end-only, cheap) or rebuild() (wholesale);
//    refresh must refuse them. resize() itself is covered in depth by
//    append_truncate_resize below.
// ---------------------------------------------------------------------------

#[test]
fn length_changes_require_rebuild() {
    let mut rng = Rng::new(0xFEED_FACE_0BAD_F00D);
    let mut content = random_content(&mut rng, 3 * 1024 * 1024 + 12345);
    let mut partial = PartialBlake3::build(&content, 1024).unwrap();
    assert_consistent_state(&partial, &content, "rebuild initial");

    // Append: refresh must refuse (content length changed).
    content.extend_from_slice(&random_content(&mut rng, 500_000));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        partial.refresh(&content, 0..1);
    }));
    assert!(result.is_err(), "refresh must refuse length changes");
    partial.rebuild(&content);
    assert_consistent_state(&partial, &content, "rebuild after append");

    // Truncate to a non-cell-multiple length.
    content.truncate(2 * 1024 * 1024 + 777);
    partial.rebuild(&content);
    assert_consistent_state(&partial, &content, "rebuild after truncate");

    // Truncate to zero.
    content.clear();
    partial.rebuild(&content);
    assert_consistent_state(&partial, &content, "rebuild to empty");
}

// ---------------------------------------------------------------------------
// 7. Parallel build/refresh must be bit-identical to the serial paths.
// ---------------------------------------------------------------------------

#[test]
fn parallel_paths_match_serial() {
    let mut rng = Rng::new(0xABAD_1DEA_0DDF_00D5);
    for (iter, &cell_chunks) in [1u64, 4, 1024].iter().enumerate() {
        let len = 2 * cell_chunks * 1024 + rng.below(3 * cell_chunks * 1024) + 1;
        let mut content = random_content(&mut rng, len as usize);

        let serial = PartialBlake3::build(&content, cell_chunks).unwrap();
        let parallel = PartialBlake3::build_parallel(&content, cell_chunks, 0).unwrap();
        assert_eq!(serial.cell_count(), parallel.cell_count());
        for m in 0..serial.cell_count() {
            assert_eq!(
                serial.cell_cv(m),
                parallel.cell_cv(m),
                "parallel build drift (iter={iter} P={cell_chunks} cell {m})"
            );
        }
        assert_eq!(
            serial.full_hash(&content),
            parallel.full_hash(&content),
            "parallel build hash drift (iter={iter})"
        );

        // A bulk overwrite (many dirty cells) via refresh_parallel.
        let mid = (len / 4) as usize;
        let span = (len / 2) as usize;
        rng.fill(&mut content[mid..mid + span]);
        let range = mid as u64..(mid + span) as u64;
        let mut serial = serial;
        let mut parallel = parallel;
        serial.refresh(&content, range.clone());
        parallel.refresh_parallel(&content, range, 0);
        for m in 0..serial.cell_count() {
            assert_eq!(
                serial.cell_cv(m),
                parallel.cell_cv(m),
                "parallel refresh drift (iter={iter} P={cell_chunks} cell {m})"
            );
        }
        assert_consistent_state(
            &parallel,
            &content,
            &format!("parallel refresh P={cell_chunks}"),
        );
    }
}

// ---------------------------------------------------------------------------
// 8. Distributed flow: nodes hash their own pieces with subtree_cv, an
//    aggregator assembles the CVs with from_cell_cvs and combines them with
//    finalize() — no file content on the combining side. Also: set_cell_cv
//    for block rewrites, cover validation errors, and the NeedsContent
//    boundary for single-cell files.
// ---------------------------------------------------------------------------

#[test]
fn distributed_combine_flow() {
    use blake3_incremental::{subtree_cv, CoverError, NeedsContent};

    let mut rng = Rng::new(0x0D15_EA5E_C0FF_EE11);
    for &(cell_chunks, len) in &[
        (4u64, 1_000_000u64),
        (4, 2 * 4096),
        (4, 4096),
        (1024, 3 * 1024 * 1024 + 777),
    ] {
        let content = random_content(&mut rng, len as usize);
        let cell_bytes = cell_chunks * 1024;
        let cells = len.div_ceil(cell_bytes);

        // Node side: each "node" has only its own byte range and offset.
        let node_cvs: Vec<ChainingValue> = (0..cells)
            .map(|m| {
                let start = (m * cell_bytes) as usize;
                let end = ((start as u64 + cell_bytes).min(len)) as usize;
                subtree_cv(&content[start..end], start as u64)
            })
            .collect();
        // The node-computed CVs must equal a monolithic build's cells.
        let built = PartialBlake3::build(&content, cell_chunks).unwrap();
        for (m, cv) in node_cvs.iter().enumerate() {
            assert_eq!(
                built.cell_cv(m),
                Some(*cv),
                "node CV != build CV (cell {m})"
            );
        }

        // Aggregator side: no content anywhere below this line.
        let cover = PartialBlake3::from_cell_cvs(len, cell_chunks, node_cvs).unwrap();
        if cells >= 2 {
            assert_eq!(cover.finalize().unwrap(), blake3::hash(&content));
        } else {
            assert_eq!(cover.finalize(), Err(NeedsContent));
            assert_eq!(cover.full_hash(&content), blake3::hash(&content));
        }

        // A block rewrite: node re-hashes its piece, aggregator swaps CV.
        if cells >= 2 {
            let mut cover = cover;
            let mut content = content;
            let block = (rng.below(cells) * cell_bytes) as usize;
            let end = (block as u64 + cell_bytes).min(len) as usize;
            rng.fill(&mut content[block..end]);
            let new_cv = subtree_cv(&content[block..end], block as u64);
            let old = cover.set_cell_cv(block / cell_bytes as usize, new_cv);
            assert!(old.is_some());
            assert_eq!(cover.finalize().unwrap(), blake3::hash(&content));
        }
    }

    // Cover validation.
    let err = PartialBlake3::from_cell_cvs(10_000, 4, vec![[0u8; 32]]);
    assert!(matches!(
        err,
        Err(CoverError::CellCountMismatch {
            expected: 3,
            got: 1
        })
    ));
    assert!(matches!(
        PartialBlake3::from_cell_cvs(10_000, 3, vec![[0u8; 32]; 3]),
        Err(CoverError::BadCellSize(_))
    ));
    // Empty file: zero CVs, content-free finalize.
    let empty = PartialBlake3::from_cell_cvs(0, 4, vec![]).unwrap();
    assert_eq!(empty.finalize().unwrap(), blake3::hash(b""));
    // Wrong CVs are detectable: flipping one cell changes the root.
    let mut cvs = vec![[0u8; 32]; 4];
    cvs[2][0] ^= 0xFF;
    let forged = PartialBlake3::from_cell_cvs(4 * 4096, 4, cvs).unwrap();
    let honest = PartialBlake3::build(&vec![0u8; 4 * 4096], 4).unwrap();
    assert_ne!(forged.finalize().unwrap(), honest.finalize().unwrap());
}

// ---------------------------------------------------------------------------
// 9. End-of-file resize (append/truncate) is a fast path: only cells whose
//    byte range actually changed may be re-hashed; every other surviving
//    cell CV must stay bit-for-bit identical, and resize must agree with a
//    from-scratch rebuild and with single-pass blake3. Also exercises the
//    content-free aggregator variant (resize_from_cvs).
// ---------------------------------------------------------------------------

/// First-principles staleness rule: a surviving cell's CV is stale iff its
/// byte range differs between the old and the new length (same start, end =
/// min((m+1)*cell_bytes, len)). Used to cross-check the implementation's
/// dirty-cell bookkeeping independently.
fn range_changed(old_len: u64, new_len: u64, m: u64, cell_bytes: u64) -> bool {
    let old_end = ((m + 1) * cell_bytes).min(old_len);
    let new_end = ((m + 1) * cell_bytes).min(new_len);
    old_end != new_end
}

fn dirty_cvs_from_content(content: &[u8], old_len: u64, cell_chunks: u64) -> Vec<ChainingValue> {
    // CVs, in cell order, of exactly the cells whose range changed: surviving
    // cells with a changed range, then newly added cells. This is what the
    // storage nodes would report for the pieces they re-hashed.
    let cb = cell_chunks * 1024;
    let new_len = content.len() as u64;
    let old_cells = if old_len == 0 {
        0
    } else {
        old_len.div_ceil(cb)
    };
    let new_cells = if new_len == 0 {
        0
    } else {
        new_len.div_ceil(cb)
    };
    let mut cvs = Vec::new();
    for m in 0..new_cells.min(old_cells) {
        if range_changed(old_len, new_len, m, cb) {
            let start = (m * cb) as usize;
            let end = (((m + 1) * cb).min(new_len)) as usize;
            cvs.push(subtree_cv(&content[start..end], start as u64));
        }
    }
    for m in old_cells..new_cells {
        let start = (m * cb) as usize;
        let end = (((m + 1) * cb).min(new_len)) as usize;
        cvs.push(subtree_cv(&content[start..end], start as u64));
    }
    cvs
}

#[test]
fn append_truncate_resize() {
    use blake3_incremental::ResizeCvError;

    let mut rng = Rng::new(0xC0FF_EEF0_0D5E_ED00);
    for &p in &[4u64, 1024] {
        let cb = p * 1024;
        for &base in &[3 * cb, 3 * cb + 500, 3 * cb + 1025, cb + 1] {
            let content0 = random_content(&mut rng, base as usize);

            // Truncation targets: cell-aligned, mid-cell, mid-chunk, zero.
            for &cut in &[2 * cb, 2 * cb + 999, cb + 1, 0] {
                if cut >= base {
                    continue;
                }
                let content = &content0[..cut as usize];
                let mut partial = PartialBlake3::build(&content0, p).unwrap();
                let snapshot: Vec<Option<ChainingValue>> = (0..partial.cell_count())
                    .map(|m| partial.cell_cv(m))
                    .collect();

                partial.resize(content);

                // Only surviving cells whose range changed may differ.
                for (m, before) in snapshot.iter().enumerate() {
                    if (m as u64) < partial.cell_count() as u64
                        && !range_changed(base, cut, m as u64, cb)
                    {
                        assert_eq!(
                            partial.cell_cv(m),
                            *before,
                            "untouched cell {m} re-hashed by truncate {base}->{cut} (P={p})"
                        );
                    }
                }
                assert_consistent_state(
                    &partial,
                    content,
                    &format!("truncate {base}->{cut} P={p}"),
                );
                // resize must equal a from-scratch rebuild, cell for cell.
                let rebuilt = PartialBlake3::build(content, p).unwrap();
                for m in 0..partial.cell_count() {
                    assert_eq!(partial.cell_cv(m), rebuilt.cell_cv(m), "truncate drift {m}");
                }

                // Aggregator variant: same final state from the CVs of the
                // affected cells only (nodes re-hash just those pieces).
                let mut from_cvs = PartialBlake3::build(&content0, p).unwrap();
                let dirty = dirty_cvs_from_content(content, base, p);
                from_cvs
                    .resize_from_cvs(cut, dirty)
                    .unwrap_or_else(|e| panic!("resize_from_cvs: {e}"));
                for m in 0..partial.cell_count() {
                    assert_eq!(
                        from_cvs.cell_cv(m),
                        partial.cell_cv(m),
                        "from_cvs drift after truncate (cell {m})"
                    );
                }
                assert_eq!(from_cvs.finalize().unwrap(), blake3::hash(content));

                // Append random bytes back onto the truncated state.
                let mut content = content0[..cut as usize].to_vec();
                let extra = rng.below(base - cut) + 1;
                content.extend_from_slice(&random_content(&mut rng, extra as usize));
                let new_len = content.len() as u64;
                let snapshot: Vec<Option<ChainingValue>> = (0..partial.cell_count())
                    .map(|m| partial.cell_cv(m))
                    .collect();
                partial.resize(&content);
                for (m, before) in snapshot.iter().enumerate() {
                    if (m as u64) < partial.cell_count() as u64
                        && !range_changed(cut, new_len, m as u64, cb)
                    {
                        assert_eq!(
                            partial.cell_cv(m),
                            *before,
                            "untouched cell {m} re-hashed by append (P={p})"
                        );
                    }
                }
                assert_consistent_state(&partial, &content, &format!("append P={p}"));

                // Aggregator variant of the append.
                let mut from_cvs = PartialBlake3::build(&content0[..cut as usize], p).unwrap();
                let dirty = dirty_cvs_from_content(&content, cut, p);
                from_cvs
                    .resize_from_cvs(new_len, dirty)
                    .unwrap_or_else(|e| panic!("resize_from_cvs append: {e}"));
                for m in 0..partial.cell_count() {
                    assert_eq!(
                        from_cvs.cell_cv(m),
                        partial.cell_cv(m),
                        "from_cvs drift after append (cell {m})"
                    );
                }
            }

            // Truncate to zero: nothing left, empty-file hash.
            let mut partial = PartialBlake3::build(&content0, p).unwrap();
            partial.resize(&[]);
            assert_eq!(partial.cell_count(), 0);
            assert_eq!(partial.finalize().unwrap(), blake3::hash(b""));
            // ...and grow from empty again.
            partial.resize(&content0);
            let rebuilt = PartialBlake3::build(&content0, p).unwrap();
            for m in 0..partial.cell_count() {
                assert_eq!(
                    partial.cell_cv(m),
                    rebuilt.cell_cv(m),
                    "grow-from-empty drift {m}"
                );
            }
        }
    }

    // Wrong dirty-CV count must error and leave the state unchanged.
    let content = random_content(&mut rng, 3 * 1024 * 1024);
    let mut partial = PartialBlake3::build(&content, 1024).unwrap();
    let before: Vec<ChainingValue> = (0..partial.cell_count())
        .map(|m| partial.cell_cv(m).unwrap())
        .collect();
    // Truncating into the middle of the final cell needs exactly one CV.
    let err = partial.resize_from_cvs(2 * 1024 * 1024 + 500, vec![]);
    assert!(matches!(
        err,
        Err(ResizeCvError::CellCountMismatch {
            expected: 1,
            got: 0,
            ..
        })
    ));
    assert_eq!(
        partial.file_len(),
        3 * 1024 * 1024,
        "state changed on error"
    );
    for (m, cv) in before.iter().enumerate() {
        assert_eq!(partial.cell_cv(m), Some(*cv), "cell {m} changed on error");
    }
    // Cell-aligned truncation needs zero CVs and keeps every survivor.
    partial.resize_from_cvs(2 * 1024 * 1024, vec![]).unwrap();
    assert_consistent_state(
        &partial,
        &content[..2 * 1024 * 1024],
        "aligned truncate from_cvs",
    );
}

// ---------------------------------------------------------------------------
// 6. API validation.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 6. API validation.
// ---------------------------------------------------------------------------

#[test]
fn cell_size_validation() {
    assert!(matches!(PartialBlake3::new(0), Err(CellSizeError::Zero)));
    assert!(matches!(
        PartialBlake3::new(3),
        Err(CellSizeError::NotPowerOfTwo)
    ));
    assert!(matches!(
        PartialBlake3::new(6),
        Err(CellSizeError::NotPowerOfTwo)
    ));
    assert!(matches!(
        PartialBlake3::new(1025),
        Err(CellSizeError::NotPowerOfTwo)
    ));
    for p in [1u64, 2, 4, 8, 16, 1024, 1 << 40] {
        assert!(PartialBlake3::new(p).is_ok(), "P={p} should be valid");
    }
}

#[test]
fn empty_and_tiny_files() {
    // The canonical BLAKE3 hash of the empty input, for documentation value.
    assert_eq!(
        blake3::hash(b"").to_hex().to_string(),
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    );
    for &cell_chunks in &[1u64, 1024, 1 << 20] {
        for len in 0..=1025usize {
            let content = vec![0x5Au8; len];
            let partial = PartialBlake3::build(&content, cell_chunks).unwrap();
            assert_consistent_state(
                &partial,
                &content,
                &format!("tiny len={len} P={cell_chunks}"),
            );
        }
    }
}
