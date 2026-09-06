# Code review: unnecessary complexity and design concerns

Review of commit `44b961e` (all code in `src/`, `tests/`, `examples/`, `Cargo.toml`).
Baseline verified before changes: `cargo check` clean, all 12 test groups + 2 doc
tests pass, `cargo clippy` reports only the 2 warnings in C2 below.

Verdict: the crate is well designed and unusually well tested. The findings below
are refinements, not redesigns. The core scheme (power-of-two cell cover + replay
of the reference subtree stack at cell granularity) is correct and appropriately
hammered against single-pass `blake3::hash`.

Legend: **A** = unnecessary complexity, **B** = design concern, **C** = minor
correctness / robustness / lint.

## Findings

### A1. Misleading cast in `compute_cells` share splitting (`src/partial.rs`)

```rust
let share_last = first
    + if w + 1 == workers {
        count
    } else {
        (w + 1) * count / workers
    } as u64
    - 1;
```

Reads as if the `as u64` applies to the else arm only (which would be a type
error). In fact it binds to the *whole* `if` expression, so the expression is
correct-but-obscure, and one edit away from a real bug. Also: each worker
allocates a `Vec` of results that is then spliced back into `cvs`.

**Resolution:** rewrite to write directly into disjoint mutable slices via
`cvs.chunks_mut(count.div_ceil(workers))`, dropping the per-worker allocation
and the splice loop. (Split-by-division load balancing is preserved in effect:
per-cell work is uniform, and `chunks_mut` gives shares that differ by at most
one cell.)

### A2. Redundant zero-fill + copy in `rebuild_parallel` (`src/partial.rs`)

```rust
self.cells.resize(cells as usize, [0u8; 32]);
let computed = compute_cells(...);
self.cells.copy_from_slice(&computed);
```

The resize is immediately overwritten.

**Resolution:** `self.cells = compute_cells(...)`.

### A3. `resize_inner` callback indirection (`src/partial.rs`)

`resize_inner` takes a `FnOnce(RangeInclusive<u64>, &mut [ChainingValue])`
closure; one caller uses the range, the other ignores it (`|_, cells_dirty|`).
Over-abstraction to save a few lines.

**Resolution:** rename to `resize_dirty_tail` and return
`Option<(RangeInclusive<u64>, &mut [ChainingValue])>`; each caller does its own
one-liner with the dirty slice.

### A4. `parse_arg` in `examples/bench.rs`

Re-scanned `env::args()` per flag, and computed `1 + position.unwrap_or(usize::MAX)` —
which **overflows and panics in debug builds when a flag is absent**. The demo
only ever worked because the README runs it in release mode, where the
overflow wraps silently. Malformed values were also swallowed without notice.

**Resolution:** single-pass scan with an explicit `--name <value>` read;
keeps the silent fallback to defaults (acceptable for a demo).

### B1. Unpinned `blake3::hazmat` dependency (`Cargo.toml`)

The whole scheme rests on `blake3::hazmat`, which upstream labels unstable
"hazardous material". `version = "1.8.7"` is a caret requirement, so a
semver-compatible `1.9` could rename/re-signature `HasherExt`, `Mode`,
`merge_subtrees_*` and silently break this crate on `cargo update`. The README
says "official blake3 crate (v1.8.7)".

**Resolution:** pin `=1.8.7` with a comment explaining why; add a
golden-vector test (hardcoded expected hashes, independent of the oracle) so
an upstream change fails loudly in CI even if it compiles.

### B2. Panics vs `Result` in the public API

`build`/`from_cell_cvs`/`resize_from_cvs` return `Result`; `refresh`/
`full_hash` panic on misuse; `subtree_cv` panics on non-chunk-aligned offsets,
empty pieces, and invalid subtree shapes (inherited from hazmat). The README's
target user is a long-running integrity service.

**Resolution (partially applied):**
- `subtree_cv`: a fallible version could only check offset alignment and
  emptiness — the subtree-shape rule needs the file length, which a node
  legitimately lacks. Partial validation would give false confidence, so the
  signature stays; the doc now states the exact panic conditions instead.
- `refresh`/`full_hash` panics are documented programming-error contracts
  (Rust convention); left as panics, with an added assert for reversed ranges
  (C1).
- The decision is recorded here rather than forcing churn across tests and
  doc examples.

### B3. `full_hash(content)` footgun (`src/partial.rs`)

For files spanning >1 cell, `content` is never read — only its length is
checked. Stale-but-same-length content silently yields the old hash. The
signature invites exactly this mistake.

**Resolution:** can't be fully fixed without hashing (which would defeat the
purpose); `finalize()` already exists as the content-free path. The doc now
warns loudly and points callers who can't trust their content copy at
`finalize()`.

### B4. Missing `Clone`/`Debug`/`PartialEq` on `PartialBlake3`

The aggregator workflow (snapshot before mutation, compare covers, log state)
wants all three; they're free to derive (`ChainingValue` is `[u8; 32]`).

**Resolution:** derive `Clone, Debug, PartialEq, Eq`.

### B5. `set_cell_cv` silently ignores out-of-range indices

Inconsistent with `from_cell_cvs`, which rejects count mismatches with a typed
error. For an aggregator acting on a node-reported cell index, silent no-op is
the worst failure mode.

**Resolution:** keep the `Option` signature (changing it churns doc examples
and tests for little gain — the index comes from trusted metadata, and the
return value is there to be checked); document the asymmetry explicitly.

### B6. `resize` silently no-ops on unchanged length

Correct for its contract (grow/shrink), but a caller who switches an
in-place-edit call site to `resize` gets a silent stale hash.

**Resolution:** doc note.

### B7. Latent `u64` overflow in `compute_cell_cv` (`src/partial.rs`)

`(m + 1) * cell_bytes` wraps for files near 16 EiB (`file_len` near
`u64::MAX`), silently hashing a truncated range. The crate documents no maximum
file size and BLAKE3 itself accepts `u64` lengths.

**Resolution:** `(m + 1).saturating_mul(cell_bytes)`.

### C1. `refresh` silently ignores reversed ranges

`Range::is_empty()` short-circuits, so `changed = 5..3` does nothing instead
of failing.

**Resolution:** assert `changed.start <= changed.end` before the empty check.

### C2. Clippy: manual `% m == 0` instead of `is_multiple_of`

Two warnings in `resize_dirty_from` (`src/partial.rs`).

**Resolution:** use `is_multiple_of` (toolchain ≥ 1.87).

### C3. `bench.rs` panics on degenerate CLI input

`--size 0` underflows `file_len - 64`; `--trials 0` indexes an empty vec;
`--trials 1` divides by zero in `mean`.

**Resolution:** up-front asserts with clear messages.

### C4. No golden vectors independent of the oracle

Every test compares against `blake3::hash` at runtime; a blake3 upgrade that
changed tree behavior while still compiling would be caught only by
self-consistent-but-wrong output (or not at all, if the oracle changed in
lockstep with hazmat).

**Resolution:** `golden_vectors` test with hardcoded expected hashes computed
from the verified implementation:
- 2 MiB + 1234 bytes, `b = i % 251`, P = 1024 →
  `99fbe418630571742d7757cbca04f598d1b348e645abd5407109585fa84b33eb`
- 500,000 bytes of `0x07`, P = 1024 (single-cell content path) →
  `cf927c07bbd25b88456a2ac1a93ba98d9e8101e290078a26b479f452daee1376`
- same content, bytes 100..164 filled with `0xAB`, after `refresh` →
  `62b7e654209d8c81ab2dae0ebe5759251d443aa484bba3bdb35ceb30d5fbb993`

## Resolution status

- [x] A1 `compute_cells` rewrite (chunks_mut, no splice)
- [x] A2 `rebuild_parallel` single assignment
- [x] A3 `resize_inner` → `resize_dirty_tail` returning `Option`
- [x] A4 `parse_arg` rewritten (was a latent debug-build overflow panic, not
      just clunky)
- [x] B1 pin `blake3 = "=1.8.7"` + golden test (C4)
- [x] B2 `subtree_cv` docs list panic conditions; panics otherwise kept
- [x] B3 `full_hash` doc warning
- [x] B4 derives on `PartialBlake3`
- [x] B5 `set_cell_cv` doc note on the asymmetry
- [x] B6 `resize` doc note on the unchanged-length no-op
- [x] B7 saturating mul in `compute_cell_cv`
- [x] C1 reversed-range assert in `refresh`
- [x] C2 `is_multiple_of`
- [x] C3 bench input guards
- [x] C4 `golden_vectors` test
