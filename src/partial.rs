//! Incremental BLAKE3 hashing from partial hashes of a power-of-two cell cover.
//!
//! The trick that makes BLAKE3 incrementally recomputable is that its tree
//! structure is fixed by the *number* of chunks in the file, and every node of
//! the tree (a parent node or a chunk) hashes to a 32-byte "chaining value"
//! that depends only on the node's own content and position. So if we keep the
//! chaining values of the power-of-two blocks ("cells") that cover the file,
//! we can recompute the root hash after an edit by re-hashing only the cells
//! the edit touched, and recombining the stored chaining values.
//!
//! Concretely, for a cell size of `P = 2^k` chunks (each chunk is 1024 bytes):
//!
//! ```text
//!             root
//!      /------/------\--------- ... ------\        <- recomputed from
//!    cell0           cell1              cell(M-1)      stored 32-byte CVs
//!  [0, P) chunks   [P, 2P) chunks    [.., N) chunks
//! ```
//!
//! * [`PartialBlake3::build`] computes one chaining value per cell (a full
//!   pass over the file, but afterwards the file bytes are not needed again —
//!   only the ~`32 * len / (P * 1024)` bytes of chaining values are kept).
//! * [`PartialBlake3::full_hash`] reconstructs the root hash from the stored
//!   cell chaining values in `O(#cells)` parent-node compressions, without
//!   touching the file content (as long as the file spans more than one cell).
//! * After an in-place edit, [`PartialBlake3::refresh`] re-hashes only the
//!   cells the edit touched, and then [`PartialBlake3::full_hash`] is fast
//!   again: changing one 64-byte block inside a 1 GiB file costs one ~1 MiB
//!   cell re-hash plus a handful of parent compressions, instead of hashing
//!   the whole file.
//! * For a distributed setup (blocks on storage nodes, checksumming done by
//!   an integrity service that never sees the content), the same pieces are
//!   available without any content: nodes hash their own block with
//!   [`subtree_cv`], the service assembles the collected chaining values with
//!   [`PartialBlake3::from_cell_cvs`] and combines them with
//!   [`PartialBlake3::finalize`]; block rewrites update a single cell via
//!   [`PartialBlake3::set_cell_cv`].
//!
//! # Correctness contract
//!
//! The cell cover is exactly the decomposition used by the official
//! implementation: every aligned power-of-two range of chunks is a subtree
//! node of the canonical BLAKE3 tree, and the crate's `hazmat` module (used
//! here for chunk/subtree hashing and parent merges) blesses the
//! fixed-power-of-two-slice scheme explicitly. The final combination step
//! mirrors the single-pass `blake3::Hasher`'s own subtree stack (see
//! `add_chunk_chaining_value`/`finalize` in the BLAKE3 source), with cells in
//! place of chunks. The randomized tests in `tests/partial.rs` verify the
//! whole thing end-to-end against single-pass `blake3::hash` for many file
//! sizes, cell sizes, and edit sequences.
//!
//! # Example
//!
//! ```
//! use blake3_incremental::PartialBlake3;
//!
//! # let mut content = vec![0u8; 3 * 1024 * 1024];
//! // First pass: hash the file into partial chaining values, one per cell.
//! // (1024 chunks = 1 MiB per cell by default.)
//! let mut partial = PartialBlake3::build(&content, 1024).unwrap();
//!
//! // The full hash can be recomputed from the partial pieces alone.
//! assert_eq!(partial.full_hash(&content), blake3::hash(&content));
//!
//! // Some time later: one block of the file changes in place...
//! let offset: u64 = 2 * 1024 * 1024 + 12345;
//! content[offset as usize..offset as usize + 64].fill(0xAB);
//! // ...only the cell containing it is re-hashed...
//! partial.refresh(&content, offset..offset + 64);
//! // ...and the full hash comes out of the updated pieces.
//! assert_eq!(partial.full_hash(&content), blake3::hash(&content));
//! ```
//!
//! # Distributed filesystems
//!
//! If the file's blocks live on different storage nodes, the pieces line up
//! exactly with that layout: pick `cell_chunks` so a cell is one block (or a
//! few blocks stored together), and each node computes the chaining value of
//! its own piece with [`subtree_cv`] — a function of the piece's bytes and
//! its offset only, no knowledge of the rest of the file. An integrity
//! service that wants the file's checksum then requests the stored chaining
//! values (32 bytes per piece — not the pieces themselves), assembles them
//! with [`PartialBlake3::from_cell_cvs`], and combines them with
//! [`PartialBlake3::finalize`]:
//!
//! ```
//! use blake3_incremental::{PartialBlake3, subtree_cv};
//! use blake3::hazmat::ChainingValue;
//!
//! # let file_len: u64 = 5 * 1024 * 1024 + 300;
//! # let cell_chunks: u64 = 4; // one 4 KiB block per cell
//! # let content = vec![7u8; file_len as usize];
//! // Each storage node, independently, at write time:
//! let node_cvs: Vec<ChainingValue> = (0..file_len.div_ceil(cell_chunks * 1024))
//!     .map(|m| {
//!         let start = (m * cell_chunks * 1024) as usize;
//!         let end = ((start as u64 + cell_chunks * 1024).min(file_len)) as usize;
//!         subtree_cv(&content[start..end], start as u64) // bytes + offset, nothing else
//!     })
//!     .collect();
//!
//! // The integrity service, with no file content, only the collected CVs:
//! let cover = PartialBlake3::from_cell_cvs(file_len, cell_chunks, node_cvs).unwrap();
//! assert_eq!(cover.finalize().unwrap(), blake3::hash(&content));
//!
//! // Later, a block is rewritten on its node; the node re-hashes its piece
//! // and reports the new CV, and the service swaps it in place:
//! # let mut content = content;
//! # content[2 * 4096..3 * 4096].fill(9);
//! let mut cover = cover;
//! cover.set_cell_cv(2, subtree_cv(&content[2 * 4096..3 * 4096], 2 * 4096));
//! // The new checksum comes from the updated CVs — no file bytes were sent.
//! assert_eq!(cover.finalize().unwrap(), blake3::hash(&content));
//! ```

use std::fmt;
use std::ops::{Range, RangeInclusive};

use blake3::hazmat::{
    merge_subtrees_non_root, merge_subtrees_root, ChainingValue, HasherExt, Mode,
};
use blake3::{Hash, Hasher};

/// Default number of 1024-byte chunks per cell: 1024 chunks = 1 MiB per cell.
pub const DEFAULT_CELL_CHUNKS: u64 = 1024;

const CHUNK_LEN: u64 = blake3::CHUNK_LEN as u64; // 1024 bytes per chunk

/// Reasons why a requested cell size is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellSizeError {
    /// A cell must contain at least one chunk.
    Zero,
    /// The number of chunks per cell must be a power of two.
    NotPowerOfTwo,
    /// `cell_chunks * 1024` overflows `u64`.
    BytesOverflow,
}

impl fmt::Display for CellSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellSizeError::Zero => write!(f, "cell size must be at least 1 chunk"),
            CellSizeError::NotPowerOfTwo => {
                write!(f, "cell size in chunks must be a power of two")
            }
            CellSizeError::BytesOverflow => write!(f, "cell size in bytes overflows u64"),
        }
    }
}

impl std::error::Error for CellSizeError {}

/// Reasons why a set of remote chaining values cannot form a valid cover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverError {
    /// The cell size is not a valid power of two.
    BadCellSize(CellSizeError),
    /// The number of collected chaining values does not match the number of
    /// cells the file length implies (`ceil(file_len / cell_bytes)`).
    CellCountMismatch {
        /// Number of cells the file length requires.
        expected: u64,
        /// Number of chaining values that were supplied.
        got: usize,
    },
}

impl fmt::Display for CoverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoverError::BadCellSize(e) => write!(f, "invalid cell size: {e}"),
            CoverError::CellCountMismatch { expected, got } => write!(
                f,
                "wrong number of chaining values: file length implies {expected} cells, got {got}"
            ),
        }
    }
}

impl std::error::Error for CoverError {}

/// The file's root hash cannot be derived from the stored chaining values
/// alone because the file fits in a single cell (see
/// [`PartialBlake3::finalize`]). Fetch the file content and use
/// [`PartialBlake3::full_hash`], or keep a whole-file hash for small files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedsContent;

impl fmt::Display for NeedsContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the file fits in a single cell: its root hash needs the file content \
             (finalize() only works for files spanning more than one cell)"
        )
    }
}

impl std::error::Error for NeedsContent {}

/// Errors from [`PartialBlake3::resize_from_cvs`]: the supplied chaining
/// values do not match the number of cells whose content the length change
/// affects. The state is left unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResizeCvError {
    /// `got` chaining values were supplied, but the length change from
    /// `file_len_before` to `file_len_after` affects `expected` cells.
    ///
    /// For a truncation to a cell-aligned length that is `expected == 0`;
    /// for a truncation into the middle of the final cell, `expected == 1`
    /// (the surviving short tail cell); for an append, `expected` is the
    /// number of cells from the (possibly extended) old final cell on.
    CellCountMismatch {
        /// File length before the change.
        file_len_before: u64,
        /// File length after the change.
        file_len_after: u64,
        /// Number of cell chaining values required.
        expected: usize,
        /// Number of cell chaining values supplied.
        got: usize,
    },
}

impl fmt::Display for ResizeCvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResizeCvError::CellCountMismatch {
                file_len_before,
                file_len_after,
                expected,
                got,
            } => write!(
                f,
                "resizing {file_len_before} -> {file_len_after} bytes affects {expected} cell(s), \
                 but {got} chaining values were supplied",
            ),
        }
    }
}

impl std::error::Error for ResizeCvError {}

/// Partial BLAKE3 state: the chaining values of the power-of-two cells that
/// cover a file, which are enough to recompute the file's root hash.
///
/// See the [module documentation](self) for the scheme and an example.
pub struct PartialBlake3 {
    /// Number of 1024-byte chunks per full cell (`P`, a power of two).
    cell_chunks: u64,
    /// Cell size in bytes (`P * 1024`).
    cell_bytes: u64,
    /// Total file length in bytes.
    file_len: u64,
    /// One chaining value per cell; `cells[m]` covers chunks
    /// `[mP, min((m+1)P, N))` where `N` is the file's chunk count. All cells
    /// except possibly the last hold exactly `P` chunks. Empty file => empty.
    cells: Vec<ChainingValue>,
}

impl PartialBlake3 {
    /// Create an empty incremental hasher with the given cell size in chunks.
    ///
    /// `cell_chunks` must be a non-zero power of two. It is the number of
    /// 1024-byte chunks per partial, i.e. the granularity at which edits are
    /// re-hashed. The default is [`DEFAULT_CELL_CHUNKS`] (1 MiB cells).
    pub fn new(cell_chunks: u64) -> Result<Self, CellSizeError> {
        if cell_chunks == 0 {
            return Err(CellSizeError::Zero);
        }
        if !cell_chunks.is_power_of_two() {
            return Err(CellSizeError::NotPowerOfTwo);
        }
        let cell_bytes = cell_chunks
            .checked_mul(CHUNK_LEN)
            .ok_or(CellSizeError::BytesOverflow)?;
        Ok(Self {
            cell_chunks,
            cell_bytes,
            file_len: 0,
            cells: Vec::new(),
        })
    }

    /// Hash `content` into partial chaining values in one full pass and return
    /// the incremental state. Afterwards the content is not needed to compute
    /// the hash, as long as the file spans more than one cell.
    ///
    /// This is the "first compute partial hashes" step. It costs the same as
    /// a single-pass BLAKE3 hash of the file (it uses the same subtree
    /// primitives), and stores `32 * cell_count` bytes.
    ///
    /// See also [`Self::build_parallel`], which divides the cells across
    /// multiple threads.
    pub fn build(content: &[u8], cell_chunks: u64) -> Result<Self, CellSizeError> {
        let mut partial = Self::new(cell_chunks)?;
        partial.rebuild(content);
        Ok(partial)
    }

    /// Like [`Self::build`], but hash the cells on `threads` worker threads
    /// (the cells are independent, so this scales linearly). `threads == 0`
    /// means [`std::thread::available_parallelism`].
    pub fn build_parallel(
        content: &[u8],
        cell_chunks: u64,
        threads: usize,
    ) -> Result<Self, CellSizeError> {
        let mut partial = Self::new(cell_chunks)?;
        partial.rebuild_parallel(content, threads);
        Ok(partial)
    }

    /// Number of 1024-byte chunks per cell (`P`).
    pub fn cell_chunks(&self) -> u64 {
        self.cell_chunks
    }

    /// Cell size in bytes (`P * 1024`).
    pub fn cell_bytes(&self) -> u64 {
        self.cell_bytes
    }

    /// Length of the file this state covers, in bytes.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Number of stored partial chaining values (cells covering the file).
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// The stored chaining value of cell `m`, if any.
    pub fn cell_cv(&self, m: usize) -> Option<ChainingValue> {
        self.cells.get(m).copied()
    }

    /// Replace the stored chaining value of cell `m` and return the previous
    /// value — or `None` (changing nothing) if `m` is out of range.
    ///
    /// This is the aggregator-side counterpart of a block rewrite: after a
    /// storage node rewrites the block a cell covers, it reports the new CV
    /// ([`subtree_cv`]) and the aggregator swaps it in place. The next
    /// [`Self::finalize`] reflects the new content; no file bytes ever reach
    /// the aggregator.
    pub fn set_cell_cv(&mut self, m: usize, cv: ChainingValue) -> Option<ChainingValue> {
        let slot = self.cells.get_mut(m)?;
        Some(std::mem::replace(slot, cv))
    }

    /// Assemble the incremental state from chaining values that were computed
    /// elsewhere (e.g. requested from the storage nodes that hold the file's
    /// blocks). No file content is needed.
    ///
    /// `cvs` must contain exactly one chaining value per cell, in cell order:
    /// cell `m` covers bytes `[m * cell_bytes, min((m+1) * cell_bytes,
    /// file_len))`, i.e. every cell holds `cell_chunks` full chunks except the
    /// last one, which may be shorter. The caller (typically the metadata or
    /// integrity service) is responsible for knowing `file_len` and
    /// `cell_chunks` and for collecting the CVs in order; the number of CVs
    /// is validated against `file_len` and `cell_chunks`.
    ///
    /// The chaining values themselves are self-consistent only if each node
    /// hashed its piece with the correct global offset (see [`subtree_cv`]).
    /// Combine the result with [`Self::finalize`] and compare against the
    /// file's expected checksum.
    pub fn from_cell_cvs(
        file_len: u64,
        cell_chunks: u64,
        cvs: Vec<ChainingValue>,
    ) -> Result<Self, CoverError> {
        let mut partial = Self::new(cell_chunks).map_err(CoverError::BadCellSize)?;
        let expected = file_len.div_ceil(partial.cell_bytes);
        if cvs.len() as u64 != expected {
            return Err(CoverError::CellCountMismatch {
                expected,
                got: cvs.len(),
            });
        }
        partial.file_len = file_len;
        partial.cells = cvs;
        Ok(partial)
    }

    /// Compute the full 32-byte BLAKE3 hash of the file from the stored
    /// partial chaining values alone — the file content is never needed (as
    /// long as the file spans more than one cell). This is the combining step
    /// for a distributed integrity check: request the cells' CVs, assemble
    /// them with [`Self::from_cell_cvs`], and compare the result against the
    /// expected checksum.
    ///
    /// # Errors
    ///
    /// Returns [`NeedsContent`] when the file fits in a single cell: the
    /// root node then sits *inside* the cell, so its hash cannot be derived
    /// from the cell CV alone (for a one-chunk file even the root output
    /// block is file content). Fetch the content and use
    /// [`Self::full_hash`], or store the whole-file hash separately for small
    /// files. The empty file needs no content and finalizes fine.
    pub fn finalize(&self) -> Result<Hash, NeedsContent> {
        match self.cells.len() {
            0 => Ok(blake3::hash(b"")),
            1 => Err(NeedsContent),
            _ => Ok(self.root_from_cells()),
        }
    }

    /// (Re)compute the partial chaining values for `content` from scratch.
    ///
    /// Use this when the content changed wholesale (restore from backup,
    /// middle insertion/deletion, ...). For a file that only grew or shrank
    /// at the end, [`Self::resize`] re-hashes just the affected boundary
    /// cells; for in-place edits of unchanged length, [`Self::refresh`] is
    /// much cheaper.
    ///
    /// See also [`Self::rebuild_parallel`].
    pub fn rebuild(&mut self, content: &[u8]) {
        self.rebuild_parallel(content, 1);
    }

    /// Like [`Self::rebuild`], but hash the cells on `threads` worker threads
    /// (`0` = [`std::thread::available_parallelism`]). Cells are independent,
    /// so this scales nearly linearly until memory bandwidth is saturated.
    pub fn rebuild_parallel(&mut self, content: &[u8], threads: usize) {
        self.file_len = content.len() as u64;
        self.cells.clear();
        if content.is_empty() {
            return;
        }
        let cells = self.file_len.div_ceil(self.cell_bytes);
        self.cells.resize(cells as usize, [0u8; 32]);
        let computed = compute_cells(
            content,
            self.cell_bytes,
            self.file_len,
            0..=cells - 1,
            threads,
        );
        self.cells.copy_from_slice(&computed);
    }

    /// Grow or shrink the file at the end (`content` must be the file's new
    /// full content): append or truncate.
    ///
    /// A cell chaining value depends only on its own bytes and its offset, so
    /// surviving cells are untouched: truncation re-hashes at most the new
    /// final cell (and only when it is now short — a cell-aligned truncation
    /// re-hashes nothing), and an append re-hashes at most the old final cell
    /// plus the newly added cells. Cells below the change are kept
    /// bit-for-bit. The top-level merge shape changes with the new length,
    /// but that is recomputed from the stored chaining values by
    /// [`Self::finalize`] / [`Self::full_hash`] in `O(#cells)` parent
    /// compressions — no content re-hashing.
    ///
    /// Only end-of-file length changes are cheap. Inserting or deleting bytes
    /// in the middle shifts every subsequent chunk (its global counter and
    /// its content), so all cells from the edit point on must be re-hashed:
    /// use [`Self::rebuild`] for that.
    ///
    /// See also [`Self::resize_parallel`] and [`Self::resize_from_cvs`]
    /// (the content-free aggregator variant).
    pub fn resize(&mut self, content: &[u8]) {
        self.resize_parallel(content, 1);
    }

    /// Like [`Self::resize`], but hash the affected cells on `threads` worker
    /// threads (`0` = [`std::thread::available_parallelism`]).
    pub fn resize_parallel(&mut self, content: &[u8], threads: usize) {
        let new_len = content.len() as u64;
        let old_len = self.file_len;
        if new_len == old_len {
            return;
        }
        let cell_bytes = self.cell_bytes;
        let dirty_from = resize_dirty_from(old_len, new_len, cell_bytes);
        self.resize_inner(new_len, dirty_from, |dirty, cells_dirty| {
            let computed = compute_cells(content, cell_bytes, new_len, dirty, threads);
            cells_dirty.copy_from_slice(&computed);
        });
    }

    /// Aggregator-side resize — same semantics as [`Self::resize`], but the
    /// chaining values of the affected cells are supplied instead of the
    /// content: `cvs` must hold exactly one chaining value per cell whose
    /// content changed, in cell order. Those are computed by the storage
    /// nodes that hold the affected pieces ([`subtree_cv`] over the piece's
    /// new bytes). For a truncation to a cell-aligned length (or to zero)
    /// nothing changes below the top, so `cvs` must be empty; for a
    /// truncation into the middle of the final cell it must hold the one CV
    /// of the surviving short tail; for an append it must hold the CV of the
    /// old final cell (if it was short and got extended) plus one CV per new
    /// cell.
    ///
    /// # Errors
    ///
    /// Returns [`ResizeCvError::CellCountMismatch`] (leaving the state
    /// unchanged) if `cvs` does not have exactly the expected length.
    pub fn resize_from_cvs(
        &mut self,
        new_len: u64,
        cvs: Vec<ChainingValue>,
    ) -> Result<(), ResizeCvError> {
        let old_len = self.file_len;
        let dirty_from = resize_dirty_from(old_len, new_len, self.cell_bytes);
        let new_cells = if new_len == 0 {
            0
        } else {
            new_len.div_ceil(self.cell_bytes)
        };
        let expected = (new_cells - dirty_from) as usize;
        if cvs.len() != expected {
            return Err(ResizeCvError::CellCountMismatch {
                file_len_before: old_len,
                file_len_after: new_len,
                expected,
                got: cvs.len(),
            });
        }
        self.resize_inner(new_len, dirty_from, |_, cells_dirty| {
            cells_dirty.copy_from_slice(&cvs);
        });
        Ok(())
    }

    /// Shared tail of the resize paths: set the new length, size the cell
    /// list, and hand the caller the slice of cells that need fresh values
    /// (those from `dirty_from` on; cells below are kept as they are).
    fn resize_inner(
        &mut self,
        new_len: u64,
        dirty_from: u64,
        fill: impl FnOnce(RangeInclusive<u64>, &mut [ChainingValue]),
    ) {
        let new_cells = if new_len == 0 {
            0
        } else {
            new_len.div_ceil(self.cell_bytes)
        };
        debug_assert!(dirty_from <= new_cells);
        self.file_len = new_len;
        self.cells.resize(new_cells as usize, [0u8; 32]);
        if dirty_from < new_cells {
            let (_, dirty) = self.cells.split_at_mut(dirty_from as usize);
            fill(dirty_from..=new_cells - 1, dirty);
        }
    }

    /// Recompute the stored chaining values of exactly the cells that
    /// intersect `changed` (a half-open byte range of an in-place edit).
    ///
    /// `content` must be the *current* full file content, and its length must
    /// match [`Self::file_len`]; only the bytes of the affected cells are
    /// read. The cost is `O(#dirty cells)` hashing instead of `O(file)`.
    ///
    /// See also [`Self::refresh_parallel`].
    ///
    /// # Panics
    ///
    /// Panics if the content length changed: use [`Self::resize`] for
    /// appends/truncations, [`Self::rebuild`] for wholesale changes. Also
    /// panics if `changed` lies outside the file.
    pub fn refresh(&mut self, content: &[u8], changed: Range<u64>) {
        self.refresh_parallel(content, changed, 1);
    }

    /// Like [`Self::refresh`], but recompute the dirty cells on `threads`
    /// worker threads (`0` = [`std::thread::available_parallelism`]). This is
    /// worthwhile when the edited range spans many cells (e.g. a bulk
    /// overwrite); a single-block edit touches one cell and is already as
    /// parallel as it gets.
    pub fn refresh_parallel(&mut self, content: &[u8], changed: Range<u64>, threads: usize) {
        assert_eq!(
            content.len() as u64,
            self.file_len,
            "content length changed ({} != {}): use resize() for \
             appends/truncations, rebuild() for wholesale changes, not refresh()",
            content.len(),
            self.file_len,
        );
        let cell_count = self.cells.len() as u64;
        if cell_count == 0 || changed.is_empty() {
            return;
        }
        assert!(
            changed.end <= self.file_len,
            "changed range {changed:?} is out of bounds for file of length {}",
            self.file_len,
        );
        let first = changed.start / self.cell_bytes;
        let last = ((changed.end - 1) / self.cell_bytes).min(cell_count - 1);
        let computed = compute_cells(
            content,
            self.cell_bytes,
            self.file_len,
            first..=last,
            threads,
        );
        for (i, cv) in computed.into_iter().enumerate() {
            self.cells[(first + i as u64) as usize] = cv;
        }
    }

    /// Compute the full 32-byte BLAKE3 hash of the file from the stored
    /// partial chaining values.
    ///
    /// `content` must be the current file content: when the file spans more
    /// than one cell only its length is checked and the bytes are not read,
    /// but files that fit in a single cell have no partial pieces to reuse
    /// and are hashed with a normal single pass. If you do not have the
    /// content at all (e.g. an aggregator in a distributed system), use
    /// [`Self::finalize`] instead, which works without content for every file
    /// larger than one cell.
    pub fn full_hash(&self, content: &[u8]) -> Hash {
        assert_eq!(
            content.len() as u64,
            self.file_len,
            "content length mismatch: the content passed to full_hash \
             must be the current file content",
        );
        // One cell or fewer (or an empty file): nothing to combine, hash the
        // content directly. Cell sizes are >= 1024 bytes, so this only happens
        // for files up to one cell in size.
        if self.cells.len() <= 1 {
            let mut hasher = Hasher::new();
            hasher.update(content);
            return hasher.finalize();
        }
        self.root_from_cells()
    }

    /// Combine the stored cell chaining values into the file's root hash.
    /// Requires at least two cells (the caller checks).
    fn root_from_cells(&self) -> Hash {
        // Process every cell except the last one, exactly like the reference
        // implementation's `add_chunk_chaining_value` processes chunks, with
        // cell chaining values in place of chunk chaining values and the cell
        // count in place of the chunk count. Every full cell of P chunks is a
        // perfect power-of-two subtree of the file tree, so merging them on
        // the carry of the cell count reproduces the file tree's left spine.
        // The last cell is the "open" one (it is the right edge of the tree
        // and may be short), just like the reference implementation keeps the
        // final chunk out of the stack.
        let last_cell = self.cells.len() - 1;
        let mut stack: Vec<ChainingValue> = Vec::with_capacity(64);
        for (i, &cell_cv) in self.cells[..last_cell].iter().enumerate() {
            let mut cv = cell_cv;
            let mut total = (i + 1) as u64;
            while total & 1 == 0 {
                let left = stack.pop().expect("subtree stack underflow");
                cv = merge_subtrees_non_root(&left, &cv, Mode::Hash);
                total >>= 1;
            }
            stack.push(cv);
        }

        // Combine the stack with the open last cell, from the top of the
        // stack down, mirroring `Hasher::finalize`.
        let mut right = self.cells[last_cell];
        for i in (1..stack.len()).rev() {
            right = merge_subtrees_non_root(&stack[i], &right, Mode::Hash);
        }
        // The root node is a parent combining the bottom stack subtree with
        // everything to its right.
        merge_subtrees_root(&stack[0], &right, Mode::Hash)
    }
}

/// Hash one contiguous piece of a file into its 32-byte BLAKE3 subtree
/// chaining value. This is the per-node operation in a distributed setup:
/// each storage node holds the bytes of its piece and its byte offset within
/// the file, and computes this CV once when the piece is written (and again
/// after a rewrite, to verify or to report the new value).
///
/// The piece's `byte_offset` must be chunk-aligned (a multiple of 1024), and
/// the piece must be a valid subtree of the file's tree: it must be an
/// aligned power-of-two-sized range, or the file's final (possibly short)
/// range. A cover of equal power-of-two cells satisfies this automatically
/// (see [`PartialBlake3`]). These rules are enforced by the underlying
/// `blake3::hazmat` code, which panics on violations.
///
/// The result depends only on the piece's bytes and its offset, never on the
/// total file length or on neighboring pieces, so nodes can compute it
/// without knowing the rest of the file. An aggregator that collected these
/// CVs later combines them with [`PartialBlake3::from_cell_cvs`] +
/// [`PartialBlake3::finalize`], without any file content.
pub fn subtree_cv(bytes: &[u8], byte_offset: u64) -> ChainingValue {
    let mut hasher = Hasher::new();
    hasher.set_input_offset(byte_offset);
    hasher.update(bytes);
    hasher.finalize_non_root()
}

/// Hash the single cell with index `m` and return its chaining value.
///
/// The cell covers bytes `[m * cell_bytes, min((m + 1) * cell_bytes, file_len))`
/// and is hashed as an official BLAKE3 subtree starting at its own byte
/// offset, so its chunks carry their global counters.
fn compute_cell_cv(content: &[u8], m: u64, cell_bytes: u64, file_len: u64) -> ChainingValue {
    let start = m * cell_bytes;
    let end = ((m + 1) * cell_bytes).min(file_len);
    subtree_cv(&content[start as usize..end as usize], start)
}

/// Compute the chaining values of the cells `first..=last`, returning them in
/// cell order. The work is split across `threads` worker threads (`0` =
/// available parallelism, `1` = single-threaded). Cell hashing is
/// independent, so results are identical regardless of the thread count.
fn compute_cells(
    content: &[u8],
    cell_bytes: u64,
    file_len: u64,
    cells: RangeInclusive<u64>,
    threads: usize,
) -> Vec<ChainingValue> {
    let first = *cells.start();
    let last = *cells.end();
    assert!(first <= last, "empty cell range");
    let count = (last - first + 1) as usize;
    let mut cvs = vec![[0u8; 32]; count];
    let workers = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
    .min(count)
    .max(1);
    if workers == 1 {
        for (i, cv) in cvs.iter_mut().enumerate() {
            *cv = compute_cell_cv(content, first + i as u64, cell_bytes, file_len);
        }
        return cvs;
    }
    // Each worker computes an owned share and returns it; the shares are
    // spliced back in order after all workers join.
    let shares: Vec<Vec<ChainingValue>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let share_first = first + (w * count / workers) as u64;
            let share_last = first
                + if w + 1 == workers {
                    count
                } else {
                    (w + 1) * count / workers
                } as u64
                - 1;
            handles.push(s.spawn(move || {
                (share_first..=share_last)
                    .map(|m| compute_cell_cv(content, m, cell_bytes, file_len))
                    .collect()
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("cell hashing worker panicked"))
            .collect()
    });
    let mut i = 0;
    for share in shares {
        cvs[i..i + share.len()].copy_from_slice(&share);
        i += share.len();
    }
    cvs
}

/// Index of the first cell whose stored chaining value is affected when the
/// file length changes from `old_len` to `new_len` (an end-only append or
/// truncation). Cells below this index keep their bytes, offsets and chunk
/// counters, so their chaining values stay valid. The return value is in
/// `0..=new_cells`, where `new_cells` means "no surviving cell changed".
fn resize_dirty_from(old_len: u64, new_len: u64, cell_bytes: u64) -> u64 {
    let new_cells = if new_len == 0 {
        0
    } else {
        new_len.div_ceil(cell_bytes)
    };
    if new_len == old_len {
        return new_cells;
    }
    if new_len < old_len {
        // Truncation: everything below the cut is untouched. Only the new
        // final cell can change, and only when it is short (a cell-aligned
        // truncation keeps a full final cell whose CV is still valid).
        if new_len % cell_bytes == 0 {
            new_cells
        } else {
            new_cells - 1
        }
    } else {
        // Append: only the old final cell can change (when it was short and
        // is now extended); every cell from there on is new.
        let old_cells = if old_len == 0 {
            0
        } else {
            old_len.div_ceil(cell_bytes)
        };
        if old_len % cell_bytes == 0 {
            old_cells
        } else {
            old_cells - 1
        }
    }
}
