//! # blake3-incremental
//!
//! Incremental BLAKE3 hashing: keep partial hashes (chaining values) of a
//! power-of-two cell cover of a large file, and recompute the full 32-byte
//! hash after an in-place edit by re-hashing only the edited cells and
//! recombining the stored pieces.
//!
//! Built on the official [`blake3`] crate — its `hazmat` module provides the
//! subtree chaining values and parent-node merges, so the compression itself
//! is exactly the official one. The randomized test suite compares against
//! single-pass [`blake3::hash`] across many file sizes, cell sizes, and edit
//! sequences.
//!
//! See [`partial`] for the scheme, and `examples/bench.rs` for a
//! before/after timing demonstration of a single-block edit.

pub mod partial;

pub use partial::{
    subtree_cv, CellSizeError, CoverError, NeedsContent, PartialBlake3, ResizeCvError,
    DEFAULT_CELL_CHUNKS,
};
