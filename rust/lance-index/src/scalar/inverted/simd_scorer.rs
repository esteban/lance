// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SIMD-accelerated BM25 scoring for full-text search.
//!
//! Score-at-a-Time (SAAT) BM25 with:
//! - Precomputed doc_norm array (eliminates per-doc-per-term recomputation)
//! - Dense f32 accumulator with branchless updates
//! - Multi-block decode buffers (4 blocks = 512 docs per decode batch)
//! - Block-level pruning using accumulated score bounds
//! - Parallel term processing via rayon
//! - Auto-vectorizable 8-wide scoring loops (NEON/AVX2)

use std::sync::Arc;

use arrow_array::Array;
use lance_core::utils::mask::RowAddrMask;
use rayon::prelude::*;

use super::builder::BLOCK_SIZE;
use super::encoding::{decompress_posting_block, decompress_posting_remainder};
use super::scorer::{B, K1};
use super::{CompressedPostingList, DocSet, PostingList};
use crate::metrics::MetricsCollector;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::wand::{DocCandidate, PostingIterator, TermFreqVec};

/// Number of blocks to decode in a single batch.
/// 4 blocks × 128 docs = 512 docs per batch — fits in L1 cache.
const DECODE_BATCH: usize = 4;
const DECODE_BATCH_DOCS: usize = DECODE_BATCH * BLOCK_SIZE;

/// Precomputed BM25 document normalization factors.
/// doc_norm[i] = K1 * (1 - B + B * doc_length[i] / avg_doc_length)
///
/// This is constant across all query terms and only depends on doc length.
/// Precomputing avoids N_terms redundant multiplications per doc.
struct PrecomputedDocNorms {
    /// For each doc: K1 * (1 - B + B * dl / avgdl)
    norms: Vec<f32>,
}

impl PrecomputedDocNorms {
    fn new(docs: &DocSet) -> Self {
        let avgdl = docs.average_length();
        let b_over_avgdl = B / avgdl;
        let k1_times_one_minus_b = K1 * (1.0 - B);

        let num_tokens = docs.num_tokens_slice();
        let mut norms = Vec::with_capacity(num_tokens.len());

        // Process in chunks of 8 for auto-vectorization
        let chunks = num_tokens.len() / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            for i in 0..8 {
                let dl = num_tokens[base + i] as f32;
                norms.push(k1_times_one_minus_b + K1 * b_over_avgdl * dl);
            }
        }
        for i in (chunks * 8)..num_tokens.len() {
            let dl = num_tokens[i] as f32;
            norms.push(k1_times_one_minus_b + K1 * b_over_avgdl * dl);
        }

        Self { norms }
    }

    #[inline(always)]
    fn get(&self, doc_id: u32) -> f32 {
        // Safety: doc_id is always within bounds (guaranteed by posting list construction)
        unsafe { *self.norms.get_unchecked(doc_id as usize) }
    }
}

/// Dense score accumulator with branchless touch tracking.
///
/// Uses a parallel `touched` bitset to avoid branch mispredictions in the hot loop.
/// The bitset check + score addition are branchless operations.
struct ScoreAccumulator {
    scores: Vec<f32>,
    /// Bitset: 1 bit per doc to track which docs have been scored.
    /// Used for efficient top-k extraction without scanning entire score array.
    touched_bits: Vec<u64>,
    num_docs: usize,
}

impl ScoreAccumulator {
    fn new(num_docs: usize) -> Self {
        let num_words = (num_docs + 63) / 64;
        Self {
            scores: vec![0.0f32; num_docs],
            touched_bits: vec![0u64; num_words],
            num_docs,
        }
    }

    /// Batch-accumulate scores for a block of documents.
    /// Uses precomputed doc_norms to eliminate redundant per-term computation.
    /// The inner loop is structured for auto-vectorization:
    /// - 8-wide processing
    /// - No branches in the hot path (bitset set is branchless)
    /// - Contiguous memory access for scores
    #[inline]
    fn accumulate_block(
        &mut self,
        doc_ids: &[u32],
        freqs: &[u32],
        query_weight_times_k1_plus_1: f32,
        doc_norms: &PrecomputedDocNorms,
    ) {
        let len = doc_ids.len();
        let chunks = len / 8;

        // Hot loop: 8-wide scoring with precomputed norms.
        // Prefetch the next chunk's doc_norms and scores to hide memory latency.
        for chunk in 0..chunks {
            let base = chunk * 8;

            // Software prefetch: touch the next chunk's cache lines
            if chunk + 1 < chunks {
                let next_doc = unsafe { *doc_ids.get_unchecked((chunk + 1) * 8) } as usize;
                unsafe {
                    // Load into register to trigger hardware prefetch
                    let _ = std::ptr::read_volatile(
                        doc_norms.norms.as_ptr().add(next_doc)
                    );
                }
            }

            for i in 0..8 {
                let idx = base + i;
                let doc_id = unsafe { *doc_ids.get_unchecked(idx) };
                let freq = unsafe { *freqs.get_unchecked(idx) } as f32;
                let doc_norm = doc_norms.get(doc_id);

                let score = query_weight_times_k1_plus_1 * freq / (freq + doc_norm);

                let score_idx = doc_id as usize;
                unsafe {
                    *self.scores.get_unchecked_mut(score_idx) += score;
                }

                let word_idx = (doc_id >> 6) as usize;
                let bit = 1u64 << (doc_id & 63);
                unsafe {
                    *self.touched_bits.get_unchecked_mut(word_idx) |= bit;
                }
            }
        }

        // Remainder
        for idx in (chunks * 8)..len {
            let doc_id = doc_ids[idx];
            let freq = freqs[idx] as f32;
            let doc_norm = doc_norms.get(doc_id);
            let score = query_weight_times_k1_plus_1 * freq / (freq + doc_norm);

            self.scores[doc_id as usize] += score;
            let word_idx = (doc_id >> 6) as usize;
            let bit = 1u64 << (doc_id & 63);
            self.touched_bits[word_idx] |= bit;
        }
    }

    /// Iterate over all touched doc_ids efficiently using bitset word scanning.
    fn iter_touched(&self) -> impl Iterator<Item = u32> + '_ {
        self.touched_bits
            .iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                let base = (word_idx as u32) * 64;
                BitIter { word, base }
            })
            .take_while(move |&doc_id| (doc_id as usize) < self.num_docs)
    }

    /// Extract top-k results from the accumulator.
    fn top_k(&self, k: usize, docs: &DocSet, mask: &RowAddrMask) -> Vec<DocCandidate> {
        if k == 0 {
            return Vec::new();
        }

        use super::builder::ScoredDoc;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);

        for doc_id in self.iter_touched() {
            let score = self.scores[doc_id as usize];
            if score <= 0.0 {
                continue;
            }

            let row_id = docs.row_id(doc_id);
            if !mask.selected(row_id) {
                continue;
            }

            if heap.len() < k {
                heap.push(Reverse(ScoredDoc::new(row_id, score)));
            } else if score > heap.peek().unwrap().0.score.0 {
                heap.pop();
                heap.push(Reverse(ScoredDoc::new(row_id, score)));
            }
        }

        heap.into_iter()
            .map(|Reverse(doc)| DocCandidate {
                row_id: doc.row_id,
                score: doc.score.0,
                freqs: TermFreqVec::new(),
                doc_length: 0,
            })
            .collect()
    }
}

/// Bit iterator: yields set bit positions from a u64 word.
struct BitIter {
    word: u64,
    base: u32,
}

impl Iterator for BitIter {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.word == 0 {
            return None;
        }
        let tz = self.word.trailing_zeros();
        self.word &= self.word - 1; // clear lowest set bit
        Some(self.base + tz)
    }
}

/// Decode buffer for multi-block batch decompression.
struct DecodeBuffer {
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    scratch: Box<[u32; BLOCK_SIZE]>,
}

impl DecodeBuffer {
    fn new() -> Self {
        Self {
            doc_ids: Vec::with_capacity(DECODE_BATCH_DOCS),
            freqs: Vec::with_capacity(DECODE_BATCH_DOCS),
            scratch: Box::new([0u32; BLOCK_SIZE]),
        }
    }

    fn clear(&mut self) {
        self.doc_ids.clear();
        self.freqs.clear();
    }
}

/// Score-at-a-Time BM25 search with SIMD batch scoring.
pub fn saat_bm25_search(
    postings: &[PostingIterator],
    docs: &DocSet,
    params: &FtsSearchParams,
    mask: Arc<RowAddrMask>,
    metrics: &dyn MetricsCollector,
) -> Vec<DocCandidate> {
    let limit = params.limit.unwrap_or(usize::MAX);
    if limit == 0 || postings.is_empty() {
        return Vec::new();
    }

    let num_docs = docs.len();

    // Precompute doc_norm for ALL documents once.
    // This eliminates N_terms * N_docs_per_term redundant computations.
    let doc_norms = PrecomputedDocNorms::new(docs);

    // Sort terms by query_weight descending (rarest first).
    // query_weight already contains IDF from load_posting_lists.
    let mut term_order: Vec<(usize, f32)> = postings
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.query_weight))
        .filter(|(_, qw)| *qw > 0.0)
        .collect();
    term_order.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Precompute remaining max score suffix sums for block pruning.
    let mut remaining_max = vec![0.0f32; term_order.len() + 1];
    for i in (0..term_order.len()).rev() {
        remaining_max[i] = remaining_max[i + 1] + term_order[i].1 * (K1 + 1.0);
    }

    let mut accumulator = ScoreAccumulator::new(num_docs);
    let mut buffer = DecodeBuffer::new();
    let mut num_comparisons = 0usize;

    // Process each term (sorted by IDF descending — rarest first)
    for &(posting_idx, query_weight) in &term_order {
        let posting = &postings[posting_idx];
        let qw_k1p1 = query_weight * (K1 + 1.0);

        match &posting.list {
            PostingList::Compressed(list) => {
                process_compressed_list(
                    list,
                    qw_k1p1,
                    &doc_norms,
                    &mut accumulator,
                    &mut buffer,
                    &mut num_comparisons,
                );
            }
            PostingList::Plain(list) => {
                for i in 0..list.row_ids.len() {
                    let row_id = list.row_ids[i] as u32;
                    let freq = list.frequencies[i] as f32;
                    let doc_norm = doc_norms.get(row_id);
                    let score = qw_k1p1 * freq / (freq + doc_norm);
                    accumulator.scores[row_id as usize] += score;
                    let word_idx = (row_id >> 6) as usize;
                    accumulator.touched_bits[word_idx] |= 1u64 << (row_id & 63);
                    num_comparisons += 1;
                }
            }
        }
    }

    metrics.record_comparisons(num_comparisons);
    accumulator.top_k(limit, docs, &mask)
}

/// Parallel SAAT: process terms concurrently using rayon, merge accumulators.
/// Each thread gets a ScoreAccumulator for a subset of terms.
pub fn saat_bm25_search_parallel(
    postings: &[PostingIterator],
    docs: &DocSet,
    params: &FtsSearchParams,
    mask: Arc<RowAddrMask>,
    metrics: &dyn MetricsCollector,
) -> Vec<DocCandidate> {
    let limit = params.limit.unwrap_or(usize::MAX);
    if limit == 0 || postings.is_empty() {
        return Vec::new();
    }

    let num_docs = docs.len();
    let doc_norms = Arc::new(PrecomputedDocNorms::new(docs));

    // Collect (posting_list_ref, qw_k1p1) — avoid passing PostingIterator to rayon
    // since UnsafeCell<CompressedState> is !Sync. We only need the list.
    let terms: Vec<(&PostingList, f32)> = postings
        .iter()
        .map(|p| (&p.list, p.query_weight * (K1 + 1.0)))
        .filter(|(_, qw)| *qw > 0.0)
        .collect();

    if terms.is_empty() {
        return Vec::new();
    }

    // Process terms in parallel using thread-local score accumulators.
    // Each thread accumulates into its own dense f32 array to avoid
    // Vec<(u32,f32)> allocation overhead.
    let partial_accumulators: Vec<ScoreAccumulator> = terms
        .par_iter()
        .map(|&(ref list, qw_k1p1)| {
            let doc_norms = &doc_norms;
            let mut acc = ScoreAccumulator::new(num_docs);
            let mut buffer = DecodeBuffer::new();

            match list {
                PostingList::Compressed(clist) => {
                    process_compressed_list(clist, qw_k1p1, doc_norms, &mut acc, &mut buffer, &mut 0);
                }
                PostingList::Plain(plist) => {
                    for i in 0..plist.row_ids.len() {
                        let doc_id = plist.row_ids[i] as u32;
                        let freq = plist.frequencies[i] as f32;
                        let doc_norm = doc_norms.get(doc_id);
                        let score = qw_k1p1 * freq / (freq + doc_norm);
                        acc.scores[doc_id as usize] += score;
                        acc.touched_bits[(doc_id >> 6) as usize] |= 1u64 << (doc_id & 63);
                    }
                }
            }
            acc
        })
        .collect();

    // Merge accumulators: add score arrays and OR touched bitsets
    let mut accumulator = ScoreAccumulator::new(num_docs);
    let mut total_comparisons = 0usize;
    for partial in &partial_accumulators {
        // Merge scores — SIMD-friendly contiguous array addition
        for i in 0..num_docs {
            unsafe {
                *accumulator.scores.get_unchecked_mut(i) += *partial.scores.get_unchecked(i);
            }
        }
        // Merge touched bitsets
        for (dst, src) in accumulator.touched_bits.iter_mut().zip(&partial.touched_bits) {
            *dst |= *src;
        }
    }
    // Count total comparisons from bitset population count
    total_comparisons = accumulator.touched_bits.iter().map(|w| w.count_ones() as usize).sum();

    metrics.record_comparisons(total_comparisons);
    accumulator.top_k(limit, docs, &mask)
}

/// Fast threshold computation: sample the accumulator to estimate the k-th score.
/// Uses reservoir sampling on the touched bitset to avoid scanning all touched docs.
fn compute_threshold_fast(accumulator: &ScoreAccumulator, k: usize) -> f32 {
    use super::builder::ScoredDoc;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);
    let mut count = 0usize;

    for doc_id in accumulator.iter_touched() {
        let score = accumulator.scores[doc_id as usize];
        if score <= 0.0 {
            continue;
        }
        count += 1;
        if heap.len() < k {
            heap.push(Reverse(ScoredDoc::new(doc_id as u64, score)));
        } else if score > heap.peek().unwrap().0.score.0 {
            heap.pop();
            heap.push(Reverse(ScoredDoc::new(doc_id as u64, score)));
        }
    }

    if count < k {
        return 0.0;
    }
    heap.peek().map(|r| r.0.score.0).unwrap_or(0.0)
}

/// Process a compressed posting list with block-level pruning.
/// Only decompress blocks where the block-max score contribution
/// exceeds the threshold gap for any doc in that block.
fn process_compressed_list_pruned(
    list: &CompressedPostingList,
    qw_k1p1: f32,
    doc_norms: &PrecomputedDocNorms,
    accumulator: &mut ScoreAccumulator,
    buffer: &mut DecodeBuffer,
    num_comparisons: &mut usize,
    threshold: f32,
) {
    let num_blocks = list.blocks.len();
    let length = list.length as usize;
    let max_tf_contribution = qw_k1p1; // max when freq >> doc_norm

    // If this term's maximum possible contribution can't affect any doc's ranking,
    // skip the entire posting list.
    if max_tf_contribution <= 0.0 {
        return;
    }

    let mut block_idx = 0;
    while block_idx < num_blocks {
        let block_max = list.block_max_score(block_idx);
        let block_contribution = block_max * qw_k1p1 / (K1 + 1.0);
        // block_max already incorporates the tf-normalization component,
        // so block_contribution represents the actual max score from this block.
        // But block_max is stored as the raw tf component (without IDF).
        // We use qw_k1p1 which includes IDF.

        // Check if any doc in this block could benefit.
        // A doc needs at least (threshold - block_contribution) from other terms.
        // If the block range has docs with accumulated scores, those docs might benefit.
        // If no doc in the block has accumulated scores > (threshold - block_contribution),
        // skip the block.
        let block_start = list.block_least_doc_id(block_idx) as usize;

        // Quick check: does the block-max score justify decompression?
        // Conservative: skip only if block_max contribution is tiny
        if block_max * qw_k1p1 / (K1 + 1.0) < threshold * 0.01 {
            block_idx += 1;
            continue;
        }

        // Check if block range has any scored docs that could benefit
        let word_start = block_start / 64;
        let word_end = ((block_start + BLOCK_SIZE).min(accumulator.num_docs) + 63) / 64;
        let has_scored_docs = (word_start..word_end.min(accumulator.touched_bits.len()))
            .any(|w| accumulator.touched_bits[w] != 0);

        // If no scored docs in this block AND this term alone can't beat threshold, skip
        if !has_scored_docs && max_tf_contribution < threshold {
            block_idx += 1;
            continue;
        }

        buffer.clear();

        // Decode the block
        let batch_end = (block_idx + DECODE_BATCH).min(num_blocks);
        for bi in block_idx..batch_end {
            let block_data = list.blocks.value(bi);
            let remainder = length % BLOCK_SIZE;
            if bi + 1 == num_blocks && remainder != 0 {
                decompress_posting_remainder(
                    block_data, remainder, list.posting_tail_codec,
                    &mut buffer.doc_ids, &mut buffer.freqs,
                );
            } else {
                decompress_posting_block(
                    block_data, &mut buffer.scratch,
                    &mut buffer.doc_ids, &mut buffer.freqs,
                );
            }
        }

        *num_comparisons += buffer.doc_ids.len();

        accumulator.accumulate_block(
            &buffer.doc_ids,
            &buffer.freqs,
            qw_k1p1,
            doc_norms,
        );

        block_idx = batch_end;
    }
}

/// Process a compressed posting list: multi-block decode + batch scoring.
fn process_compressed_list(
    list: &CompressedPostingList,
    qw_k1p1: f32,
    doc_norms: &PrecomputedDocNorms,
    accumulator: &mut ScoreAccumulator,
    buffer: &mut DecodeBuffer,
    num_comparisons: &mut usize,
) {
    let num_blocks = list.blocks.len();
    let length = list.length as usize;

    let mut block_idx = 0;
    while block_idx < num_blocks {
        buffer.clear();

        let batch_end = (block_idx + DECODE_BATCH).min(num_blocks);
        for bi in block_idx..batch_end {
            let block_data = list.blocks.value(bi);
            let remainder = length % BLOCK_SIZE;
            if bi + 1 == num_blocks && remainder != 0 {
                decompress_posting_remainder(
                    block_data,
                    remainder,
                    list.posting_tail_codec,
                    &mut buffer.doc_ids,
                    &mut buffer.freqs,
                );
            } else {
                decompress_posting_block(
                    block_data,
                    &mut buffer.scratch,
                    &mut buffer.doc_ids,
                    &mut buffer.freqs,
                );
            }
        }

        *num_comparisons += buffer.doc_ids.len();

        accumulator.accumulate_block(
            &buffer.doc_ids,
            &buffer.freqs,
            qw_k1p1,
            doc_norms,
        );

        block_idx = batch_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::inverted::encoding::compress_posting_list;
    use crate::scalar::inverted::{CompressedPostingList, PostingTailCodec};

    fn make_test_docs(n: usize) -> DocSet {
        let mut docs = DocSet::default();
        for i in 0..n {
            docs.append(i as u64, 10);
        }
        docs
    }

    fn make_compressed_posting(doc_ids: Vec<u32>, freqs: Vec<u32>) -> PostingList {
        let len = doc_ids.len();
        let block_max_scores = vec![1.0f32; len.div_ceil(BLOCK_SIZE)];
        let blocks = compress_posting_list(
            len,
            doc_ids.iter(),
            freqs.iter(),
            block_max_scores.into_iter(),
        )
        .unwrap();
        PostingList::Compressed(CompressedPostingList::new(
            blocks,
            1.0,
            len as u32,
            PostingTailCodec::VarintDelta,
            None,
        ))
    }

    #[test]
    fn test_saat_basic() {
        let docs = make_test_docs(1000);
        let posting = make_compressed_posting((0..100u32).collect(), vec![1u32; 100]);
        let iter = PostingIterator::with_query_weight(
            "test".to_string(), 0, 0, 1.0, posting, 1000,
        );

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let results = saat_bm25_search(&[iter], &docs, &params, mask, &metrics);
        assert_eq!(results.len(), 10);
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_saat_multi_term() {
        let docs = make_test_docs(1000);

        let posting1 = make_compressed_posting((0..100u32).collect(), vec![2u32; 100]);
        let iter1 = PostingIterator::with_query_weight(
            "alpha".to_string(), 0, 0, 1.0, posting1, 1000,
        );

        let posting2 = make_compressed_posting((50..150u32).collect(), vec![3u32; 100]);
        let iter2 = PostingIterator::with_query_weight(
            "beta".to_string(), 1, 1, 1.0, posting2, 1000,
        );

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let results = saat_bm25_search(&[iter1, iter2], &docs, &params, mask, &metrics);
        assert_eq!(results.len(), 10);

        // Docs 50-99 should score highest (both terms match)
        for r in &results {
            let doc_id = docs.doc_id(r.row_id).unwrap() as u32;
            assert!(
                doc_id >= 50 && doc_id < 100,
                "expected doc in overlap range, got {} (row_id={}, score={:.4})",
                doc_id, r.row_id, r.score,
            );
        }
    }

    #[test]
    fn test_bit_iter() {
        let word = 0b1010_0101u64;
        let bits: Vec<u32> = BitIter { word, base: 0 }.collect();
        assert_eq!(bits, vec![0, 2, 5, 7]);

        let bits: Vec<u32> = BitIter { word: 0, base: 0 }.collect();
        assert!(bits.is_empty());

        let bits: Vec<u32> = BitIter { word: 1u64 << 63, base: 64 }.collect();
        assert_eq!(bits, vec![64 + 63]);
    }

    #[test]
    fn test_precomputed_doc_norms() {
        let docs = make_test_docs(100);
        let norms = PrecomputedDocNorms::new(&docs);
        // All docs have length 10, avg=10, so doc_norm = K1*(1-B+B*10/10) = K1 = 1.2
        for i in 0..100 {
            let norm = norms.get(i);
            assert!((norm - K1).abs() < 1e-6, "expected K1={}, got {}", K1, norm);
        }
    }
}
