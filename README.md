# blake3-incremental

Incremental BLAKE3 hashing in Rust: keep partial hashes of a power-of-two
"cell" cover of a large file, and recompute the full 32-byte hash after an
in-place edit by re-hashing only the edited cells and recombining the stored
pieces.

Built on the **official `blake3` crate** (v1.8.7 from crates.io):
its `hazmat` module supplies the chunk/subtree chaining values
(`HasherExt::set_input_offset` + `finalize_non_root`), the parent-node merges
(`merge_subtrees_non_root` / `merge_subtrees_root`), and the single-pass
oracle (`blake3::hash`). The compression itself is exactly the official one;
only the tree bookkeeping lives here.

## Why this works

BLAKE3 is a Merkle tree over 1024-byte chunks. Two facts make it
incrementally recomputable:

1. Every node of the canonical tree (a chunk, or a parent over two children)
   reduces to a 32-byte chaining value that depends only on that node's own
   bytes and its position — never on the rest of the file.
2. In the canonical tree, every **aligned power-of-two range of chunks is a
   subtree node**. So if the file is partitioned into aligned cells of
   `P = 2^k` chunks (the last cell may be short), each cell's chaining value
   is a real subtree node of the file tree, computable independently and
   combinable without touching the file again.

```text
                    root  <- full hash = merge_subtrees_root(left, right)
              /------------\------------ ... ------\
            cell0          cell1                   cell(M-1)
   chunks [0, P)    chunks [P, 2P)         chunks [.., N)     N = ceil(len/1024)
        cv0             cv1                        cv(M-1)     (each 32 bytes)
```

* **build / rebuild** — one full pass, hashing each cell as a subtree with its
  global chunk counters (`Hasher::set_input_offset(m·P·1024)`). Costs the same
  as a single-pass hash; afterwards only the cell chaining values are kept:
  `32 · M` bytes, e.g. **16 KiB for a 512 MiB file** with 1 MiB cells.
* **full_hash** — recombines the stored chaining values by replaying the
  reference implementation's subtree-stack algorithm (`add_chunk_chaining_value`
  / `finalize`) at cell granularity, finishing with a ROOT parent compression.
  `O(M)` parent compressions, no file bytes touched (when the file spans more
  than one cell).
* **refresh** — after an in-place edit, re-hashes only the cells the edit
  intersected. A single 64-byte block change re-hashes one cell, then
  `full_hash` recombines as above.

`blake3::hazmat` blesses exactly this scheme: *"It's also common to choose some
fixed power-of-two subtree size, say 64 chunks, and divide your input up into
slices of that fixed length (with the final slice possibly short)."*

Notes:

* The cell size (`P` chunks, power of two) trades state size vs. per-edit
  re-hash work. Default `1024` chunks = 1 MiB cells.
* **End-of-file length changes are cheap.** A cell chaining value depends
  only on its own bytes and its offset, never on the file length, so an
  append or truncation leaves every untouched cell valid: `resize` re-hashes
  at most the boundary cell (a truncation to a cell-aligned length re-hashes
  nothing), and `resize_from_cvs` does the same from node-supplied CVs. Only
  *middle* insertions/deletions shift every subsequent chunk (its content and
  its global counter), so cells from the edit point on need a `rebuild`.
  `refresh` refuses length changes; `finalize` re-derives the (new) top-level
  merge shape from the stored CVs in `O(#cells)` parent compressions.
* Files that fit in a single cell have no pieces to reuse; `full_hash` falls
  back to a plain single pass for them.
* A file whose length is an exact multiple of the chunk size is handled
  correctly: the final (full) chunk/cell stays the "open" right edge of the
  tree, exactly as in the reference implementation.

## Distributed filesystems (blocks + CVs on storage nodes)

The crate is organized so the content side and the combining side never need
to meet. In a distributed filesystem where each node stores its block(s)
*and* the block's partial checksum, pick `P` so that a cell equals one block
(or a few blocks stored together), and:

* **Write / verify a block (storage node):** [`subtree_cv`]`(block_bytes, block_offset)` —
  the 32-byte BLAKE3 chaining value of that piece. It depends only on the
  piece's bytes and its offset within the file (chunk-aligned); the node
  needs no other file knowledge. Every node computes the same function, so a
  node can later re-hash its block to prove the stored CV still matches.

* **Full file checksum (integrity service):** collect the CVs — 32 bytes per
  block, not the blocks — and combine them without any file content:

  ```rust
  let cover = PartialBlake3::from_cell_cvs(file_len, cell_chunks, node_cvs)?;
  let checksum = cover.finalize()?; // == blake3::hash(&whole_file)
  ```

  `from_cell_cvs` validates the cover (correct cell size, exactly
  `ceil(file_len / cell_bytes)` CVs); `finalize` replays the tree merge in
  `O(#cells)` parent compressions.

* **Block rewrite:** the node re-hashes its piece with `subtree_cv`, the
  service does `cover.set_cell_cv(m, new_cv)`, and the next `finalize()`
  reflects the new content. If a block is corrupt, its CV changes; comparing
  `finalize()` against the checksum stored at write time detects it, and the
  same CVs identify *which* block to re-fetch.

Operational notes:

* **Small files** (≤ one cell, i.e. ≤ `cell_chunks * 1024` bytes) cannot be
  finalized from CVs alone — the root node sits inside the cell, and for a
  one-chunk file even the root output block is content. `finalize` returns a
  [`NeedsContent`] error rather than a wrong answer. Mitigations: keep a
  whole-file hash for small files (write-time `blake3::hash`), or use small
  cells (e.g. 4 KiB → 4 chunks) so the boundary is low. Empty files finalize
  fine (canonical empty hash).
* **Append / truncate (metadata + nodes):** end-of-file length changes never
  re-hash surviving cells. Truncation: keep the CVs of the surviving cells
  (cell-aligned truncation needs nothing more; a truncation into the middle
  of the final cell needs that one cell's node to re-hash the surviving
  short tail). Append: nodes hash only the new pieces, and the metadata
  extends the cover — the old final cell's node re-hashes that cell only if
  it was short and got extended. `resize_from_cvs(new_len, cvs_of_affected_cells)`
  is the single aggregator call (it validates the count; a cell-aligned
  truncation passes an empty list). Middle insertion/deletion is the
  expensive case: subsequent content shifts, so every cell from the edit
  point on must be re-hashed.
* **Combining cost** is `O(#cells)` parent compressions (~µs per thousand
  cells). For extremely many small blocks, aggregate hierarchically: each
  node pre-merges the CVs of its contiguous run (same merge logic, via
  `from_cell_cvs` + inspecting `cell_cv`) and ships far fewer values.
* **Trust model:** BLAKE3 chaining values are unkeyed. This scheme detects
  accidental corruption and misreporting *against a trusted expected
  checksum*, and it lets you pinpoint the bad block by re-hashing content.
  It does not by itself authenticate nodes that control both a block and its
  CV; for that, keep the expected checksum (or CVs) on a trusted path, or
  layer authentication on top.

## Layout

| Path | What it is |
| --- | --- |
| `src/partial.rs` | `PartialBlake3`: cell-cover state, `build`/`rebuild`, `refresh`, `resize` (+ `*_parallel` variants). Distributed pieces: `subtree_cv` (node side), `from_cell_cvs` + `finalize` + `set_cell_cv` + `resize_from_cvs` (aggregator side, content-free) |
| `src/lib.rs` | crate root / docs |
| `tests/partial.rs` | randomized tester against single-pass `blake3::hash` |
| `examples/bench.rs` | timing demo: one-block-edit recompute vs full rehash |

(Development reference: the upstream BLAKE3 source tree was consulted
locally, but is not part of this repository.)

## Randomized tester

`cargo test` runs ten deterministic (SplitMix64-seeded) test groups that
compare `PartialBlake3` output against official single-pass `blake3::hash`,
with heavy loops partitioned across all cores:

* **exhaustive** lengths 0..=700 for cell sizes 1/2/4 chunks (empty files,
  sub-chunk lengths, the ≤1-cell content fallback);
* **boundary lengths** — for P ∈ {1, 2, 4, 16, 1024}: dense small lengths,
  chunk counts just below/at/above multiples of P (the N % P == 0
  transitions), perfect-tree sizes and neighbors, and final chunks of
  1/63/64/65/1023 bytes;
* **random lengths** up to 96 MiB across cell sizes;
* **edit sequences** — hundreds of randomized in-place edits (arbitrary
  ranges, 64-byte block writes, cell-straddling, final-chunk edits); after
  each `refresh` the hash must match a fresh single pass, cells untouched by
  the edit must be bit-for-bit unchanged, and a from-scratch rebuild must
  reproduce the exact same cell state;
* **end-of-file length changes** (`append_truncate_resize`) — `resize` must
  re-hash exactly the cells whose byte range changed (cross-checked against a
  first-principles staleness rule) and agree with `rebuild` and single-pass
  blake3; `resize_from_cvs` must reach the same state from node-supplied CVs
  and refuse wrong counts without mutating state; `refresh` must refuse
  length changes;
* **parallel paths** (`build_parallel`, `refresh_parallel`) must be
  bit-identical to the serial ones.

A failing task reports its index and parameters, and every task derives its
own seed, so failures reproduce in isolation.

## Demo

```console
$ cargo run --release --example bench -- --size 1024      # 1 GiB, 1 MiB cells
```

Measured on a Ryzen 5 7640U (6 cores / 12 threads):

```text
file        : 1024 MiB, 1024 x 1024-KiB cells
partial state: 32 KiB kept after the build pass

full single pass (update_rayon) :   35.4 ms  (28.9 GiB/s)
partial build parallel (1/cell)  :   37.1 ms  (27.0 GiB/s)

full rehash after 1 block edit (all cores)   35.425 ms
recompute from partial pieces (refresh + merge)  0.207 ms    171x
bytes re-hashed per edit vs whole file         1 MiB vs 1024 MiB
```

Changing one 64-byte block costs re-hashing **1 MiB instead of 1 GiB**
(~1000× less work), plus a few dozen parent-node compressions; measured
~171× wall-clock because the full rehash uses all 12 cores while a
single-cell refresh is one core's worth of work. After many edits the same
state keeps working — every recompute touches only the edited cells.

## License

CC0-1.0 OR Apache-2.0 (this crate's own code; the `blake3` dependency is
`CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception`).
