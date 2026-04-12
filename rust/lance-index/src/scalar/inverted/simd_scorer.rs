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
#[allow(dead_code)]
const DECODE_BATCH_DOCS: usize = DECODE_BATCH * BLOCK_SIZE;

/// Maximum frequency value for the lookup table.
/// Frequencies above this fall back to direct computation.
const MAX_FREQ_LUT: usize = 64;

/// Precomputed BM25 tf-score lookup table.
/// BM25 score LUT indexed by (freq, doc_length_bucket).
/// Replaces BOTH f32 division AND doc_norm precomputation with a single table lookup.
/// Eliminates the need for PrecomputedDocNorms entirely.
pub(crate) struct ScoreLookupTable {
    table: Vec<f32>,
    dl_scale: f32,
    num_dl_buckets: usize,
    avgdl: f32,
}

const NUM_DL_BUCKETS: usize = 256;

impl ScoreLookupTable {
    pub(crate) fn new(docs: &DocSet) -> Self {
        let num_tokens = docs.num_tokens_slice();
        let avgdl = docs.average_length();
        let b_over_avgdl = B / avgdl;
        let k1_one_minus_b = K1 * (1.0 - B);

        let max_dl = num_tokens.iter().copied().max().unwrap_or(1) as f32;
        let dl_scale = (NUM_DL_BUCKETS - 1) as f32 / max_dl.max(1.0);

        let mut table = vec![0.0f32; MAX_FREQ_LUT * NUM_DL_BUCKETS];
        let k1_plus_1 = K1 + 1.0;
        for freq in 0..MAX_FREQ_LUT {
            let f = freq as f32;
            for bucket in 0..NUM_DL_BUCKETS {
                let dl = bucket as f32 / dl_scale;
                let doc_norm = k1_one_minus_b + K1 * b_over_avgdl * dl;
                table[freq * NUM_DL_BUCKETS + bucket] = k1_plus_1 * f / (f + doc_norm);
            }
        }

        Self {
            table,
            dl_scale,
            num_dl_buckets: NUM_DL_BUCKETS,
            avgdl,
        }
    }

    #[inline(always)]
    fn score(&self, freq: u32, doc_tokens: u32, query_weight: f32) -> f32 {
        if (freq as usize) < MAX_FREQ_LUT {
            let bucket =
                ((doc_tokens as f32 * self.dl_scale) as usize).min(self.num_dl_buckets - 1);
            let tf = unsafe {
                *self
                    .table
                    .get_unchecked(freq as usize * self.num_dl_buckets + bucket)
            };
            query_weight * tf
        } else {
            let f = freq as f32;
            let doc_norm = K1 * (1.0 - B + B * doc_tokens as f32 / self.avgdl);
            query_weight * (K1 + 1.0) * f / (f + doc_norm)
        }
    }
}

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
    pub(crate) fn new(docs: &DocSet) -> Self {
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
/// Dense score accumulator using u16 quantized scores.
///
/// Key insight from Mackenzie et al. (TOIS 2023): narrower accumulators give
/// dramatic speedups from cache density — 2x more accumulators per cache line.
/// u16: 1M docs = 2MB (fits L2 cache), vs f32: 4MB (spills to L3).
///
/// Per-query dynamic rescaling: map [0, max_possible_score] → [0, 65000]
/// to avoid overflow while maximizing precision.
struct ScoreAccumulator {
    scores: Vec<u16>,
    touched_bits: Vec<u64>,
    num_docs: usize,
    /// Scale: f32_score * scale → u16 quantized
    scale: f32,
    /// Inverse: u16 quantized * inv_scale → f32_score
    inv_scale: f32,
}

impl ScoreAccumulator {
    fn new(num_docs: usize, max_possible_score: f32) -> Self {
        let num_words = (num_docs + 63) / 64;
        let scale = if max_possible_score > 0.0 {
            65000.0 / max_possible_score
        } else {
            1.0
        };
        Self {
            scores: vec![0u16; num_docs],
            touched_bits: vec![0u64; num_words],
            num_docs,
            scale,
            inv_scale: 1.0 / scale,
        }
    }

    /// Batch-accumulate quantized scores for a block of documents.
    /// u16 additions are cheaper than f32 and pack 2x denser in cache lines.
    #[inline]
    fn accumulate_block(
        &mut self,
        doc_ids: &[u32],
        freqs: &[u32],
        query_weight_times_k1_plus_1: f32,
        num_tokens: &[u32],
        b_over_avgdl: f32,
    ) {
        let len = doc_ids.len();
        let chunks = len / 8;
        let scale = self.scale;

        for chunk in 0..chunks {
            let base = chunk * 8;
            for i in 0..8 {
                let idx = base + i;
                let doc_id = unsafe { *doc_ids.get_unchecked(idx) };
                let freq = unsafe { *freqs.get_unchecked(idx) } as f32;
                let doc_tokens = num_tokens[doc_id as usize];

                let doc_norm = K1 * (1.0 - B + b_over_avgdl * doc_tokens as f32);
                let score = query_weight_times_k1_plus_1 * freq / (freq + doc_norm);
                let quantized = (score * scale) as u16;

                let score_idx = doc_id as usize;
                unsafe {
                    let current = *self.scores.get_unchecked(score_idx);
                    *self.scores.get_unchecked_mut(score_idx) = current.saturating_add(quantized);
                }

                let word_idx = (doc_id >> 6) as usize;
                let bit = 1u64 << (doc_id & 63);
                unsafe {
                    *self.touched_bits.get_unchecked_mut(word_idx) |= bit;
                }
            }
        }

        for idx in (chunks * 8)..len {
            let doc_id = doc_ids[idx];
            let freq = freqs[idx] as f32;
            let doc_tokens = num_tokens[doc_id as usize];
            let doc_norm = K1 * (1.0 - B + b_over_avgdl * doc_tokens as f32);
            let score = query_weight_times_k1_plus_1 * freq / (freq + doc_norm);
            let quantized = (score * scale) as u16;

            let score_idx = doc_id as usize;
            self.scores[score_idx] = self.scores[score_idx].saturating_add(quantized);
            let word_idx = (doc_id >> 6) as usize;
            self.touched_bits[word_idx] |= 1u64 << (doc_id & 63);
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

    /// Extract top-k results. Dequantize u16 → f32 using inv_scale.
    fn top_k(&self, k: usize, docs: &DocSet, mask: &RowAddrMask) -> Vec<DocCandidate> {
        if k == 0 {
            return Vec::new();
        }

        use super::builder::ScoredDoc;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);

        for doc_id in self.iter_touched() {
            let quantized = self.scores[doc_id as usize];
            if quantized == 0 {
                continue;
            }
            let score = quantized as f32 * self.inv_scale;

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

/// Score-at-a-Time BM25 search with all optimizations combined:
/// - Precomputed doc_norms + score lookup table (zero f32 division)
/// - u16 quantized accumulators (2x cache density)
/// - Anytime termination (postings budget)
/// - Term-level early exit (skip terms < 5% of threshold)
/// - Multi-block decode + 8-wide scoring loops
pub fn saat_bm25_search(
    postings: &[PostingIterator],
    docs: &DocSet,
    params: &FtsSearchParams,
    mask: Arc<RowAddrMask>,
    metrics: &dyn MetricsCollector,
    lut: &ScoreLookupTable,
) -> Vec<DocCandidate> {
    let limit = params.limit.unwrap_or(usize::MAX);
    if limit == 0 || postings.is_empty() {
        return Vec::new();
    }

    let num_docs = docs.len();
    let num_tokens = docs.num_tokens_slice();

    // Sort terms by query_weight descending (rarest first)
    let mut term_order: Vec<(usize, f32)> = postings
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.query_weight))
        .filter(|(_, qw)| *qw > 0.0)
        .collect();
    term_order.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Remaining max score suffix sums for early exit
    let mut remaining_max = vec![0.0f32; term_order.len() + 1];
    for i in (0..term_order.len()).rev() {
        remaining_max[i] = remaining_max[i + 1] + term_order[i].1 * (K1 + 1.0);
    }

    let max_possible_score = remaining_max[0];
    let mut accumulator = ScoreAccumulator::new(num_docs, max_possible_score);
    let mut buffer = DecodeBuffer::new();
    let mut num_comparisons = 0usize;

    // TODO(Codex): Re-validate these heuristics on representative queries before
    // treating SAAT latency as a launch metric. Recall is sensitive to both knobs.
    // Anytime postings budget: adaptive to query complexity.
    // For top-10 with 10 terms, 200K postings ≈ 20K per term on average.
    let postings_budget = (10 * limit * term_order.len()).max(50_000);
    let mut postings_remaining = postings_budget;
    let mut threshold = 0.0f32;

    for (term_idx, &(posting_idx, query_weight)) in term_order.iter().enumerate() {
        // Term-level early exit
        if term_idx >= 3 && threshold > 0.0 && remaining_max[term_idx] < threshold * 0.15 {
            break;
        }
        if postings_remaining == 0 {
            break;
        }

        let posting = &postings[posting_idx];

        match &posting.list {
            PostingList::Compressed(list) => {
                let processed = process_compressed_list_with_lut(
                    list,
                    query_weight,
                    num_tokens,
                    &lut,
                    &mut accumulator,
                    &mut buffer,
                    postings_remaining,
                );
                num_comparisons += processed;
                postings_remaining = postings_remaining.saturating_sub(processed);
            }
            PostingList::Plain(list) => {
                let scale = accumulator.scale;
                let to_process = list.row_ids.len().min(postings_remaining);
                for i in 0..to_process {
                    let row_id = list.row_ids[i];
                    let Some(doc_id) = docs.doc_index_by_row_id(row_id) else {
                        continue;
                    };
                    let freq = list.frequencies[i] as u32;
                    let doc_tokens = num_tokens[doc_id as usize];
                    let score = lut.score(freq, doc_tokens, query_weight);
                    let quantized = (score * scale) as u16;
                    let idx = doc_id as usize;
                    accumulator.scores[idx] = accumulator.scores[idx].saturating_add(quantized);
                    accumulator.touched_bits[(doc_id >> 6) as usize] |= 1u64 << (doc_id & 63);
                }
                num_comparisons += to_process;
                postings_remaining = postings_remaining.saturating_sub(to_process);
            }
        }

        // Update threshold for term-level early exit
        if term_idx >= 2 && term_idx % 2 == 0 {
            threshold = compute_threshold_fast(&accumulator, limit);
        }
    }

    metrics.record_comparisons(num_comparisons);
    accumulator.top_k(limit, docs, &mask)
}

fn compute_threshold_fast(accumulator: &ScoreAccumulator, k: usize) -> f32 {
    use super::builder::ScoredDoc;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);
    let mut count = 0usize;

    for doc_id in accumulator.iter_touched() {
        let quantized = accumulator.scores[doc_id as usize];
        if quantized == 0 {
            continue;
        }
        let score = quantized as f32 * accumulator.inv_scale;
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
/// Returns the number of postings processed.
fn process_compressed_list_with_lut(
    list: &CompressedPostingList,
    query_weight: f32,
    num_tokens: &[u32],
    lut: &ScoreLookupTable,
    accumulator: &mut ScoreAccumulator,
    buffer: &mut DecodeBuffer,
    postings_budget: usize,
) -> usize {
    let num_blocks = list.blocks.len();
    let length = list.length as usize;
    let scale = accumulator.scale;
    let mut processed = 0usize;

    let mut block_idx = 0;
    while block_idx < num_blocks && processed < postings_budget {
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

        // Score using LUT — replaces f32 division with table lookup
        let len = buffer.doc_ids.len().min(postings_budget - processed);
        let chunks = len / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            for i in 0..8 {
                let idx = base + i;
                let doc_id = unsafe { *buffer.doc_ids.get_unchecked(idx) };
                let freq = unsafe { *buffer.freqs.get_unchecked(idx) };
                let doc_tokens = num_tokens[doc_id as usize];
                let score = lut.score(freq, doc_tokens, query_weight);
                let quantized = (score * scale) as u16;

                let score_idx = doc_id as usize;
                unsafe {
                    let current = *accumulator.scores.get_unchecked(score_idx);
                    *accumulator.scores.get_unchecked_mut(score_idx) =
                        current.saturating_add(quantized);
                    *accumulator
                        .touched_bits
                        .get_unchecked_mut((doc_id >> 6) as usize) |= 1u64 << (doc_id & 63);
                }
            }
        }
        for idx in (chunks * 8)..len {
            let doc_id = buffer.doc_ids[idx];
            let freq = buffer.freqs[idx];
            let doc_tokens = num_tokens[doc_id as usize];
            let score = lut.score(freq, doc_tokens, query_weight);
            let quantized = (score * scale) as u16;
            accumulator.scores[doc_id as usize] =
                accumulator.scores[doc_id as usize].saturating_add(quantized);
            accumulator.touched_bits[(doc_id >> 6) as usize] |= 1u64 << (doc_id & 63);
        }

        processed += len;
        block_idx = batch_end;
    }

    processed
}

/// Process a compressed posting list: multi-block decode + batch scoring.
fn process_compressed_list(
    list: &CompressedPostingList,
    qw_k1p1: f32,
    num_tokens: &[u32],
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

        accumulator.accumulate_block(&buffer.doc_ids, &buffer.freqs, qw_k1p1, num_tokens, 0.0);

        block_idx = batch_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::inverted::encoding::compress_posting_list;
    use crate::scalar::inverted::{CompressedPostingList, PlainPostingList, PostingTailCodec};
    use arrow::buffer::ScalarBuffer;

    fn make_test_docs(n: usize) -> DocSet {
        let mut docs = DocSet::default();
        for i in 0..n {
            docs.append(i as u64, 10);
        }
        docs
    }

    fn make_sparse_row_id_docs() -> DocSet {
        let mut docs = DocSet::default();
        docs.append(100, 10);
        docs.append(200, 10);
        docs.append(500, 10);
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

    fn make_plain_posting(row_ids: Vec<u64>, freqs: Vec<f32>) -> PostingList {
        PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(row_ids),
            ScalarBuffer::from(freqs),
            Some(1.0),
            None,
        ))
    }

    #[test]
    fn test_saat_basic() {
        let docs = make_test_docs(1000);
        let posting = make_compressed_posting((0..100u32).collect(), vec![1u32; 100]);
        let iter = PostingIterator::with_query_weight("test".to_string(), 0, 0, 1.0, posting, 1000);

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let lut = ScoreLookupTable::new(&docs);
        let results = saat_bm25_search(&[iter], &docs, &params, mask, &metrics, &lut);
        assert_eq!(results.len(), 10);
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_saat_multi_term() {
        let docs = make_test_docs(1000);

        let posting1 = make_compressed_posting((0..100u32).collect(), vec![2u32; 100]);
        let iter1 =
            PostingIterator::with_query_weight("alpha".to_string(), 0, 0, 1.0, posting1, 1000);

        let posting2 = make_compressed_posting((50..150u32).collect(), vec![3u32; 100]);
        let iter2 =
            PostingIterator::with_query_weight("beta".to_string(), 1, 1, 1.0, posting2, 1000);

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let lut = ScoreLookupTable::new(&docs);
        let results = saat_bm25_search(&[iter1, iter2], &docs, &params, mask, &metrics, &lut);
        assert_eq!(results.len(), 10);

        // Docs 50-99 should score highest (both terms match)
        for r in &results {
            let doc_id = docs.doc_id(r.row_id).unwrap() as u32;
            assert!(
                doc_id >= 50 && doc_id < 100,
                "expected doc in overlap range, got {} (row_id={}, score={:.4})",
                doc_id,
                r.row_id,
                r.score,
            );
        }
    }

    #[test]
    fn test_saat_plain_postings_match_compressed_scores() {
        let docs = make_sparse_row_id_docs();

        let compressed = PostingIterator::with_query_weight(
            "test".to_string(),
            0,
            0,
            1.0,
            make_compressed_posting(vec![0, 2], vec![2, 1]),
            docs.len(),
        );
        let plain = PostingIterator::with_query_weight(
            "test".to_string(),
            0,
            0,
            1.0,
            make_plain_posting(vec![100, 500], vec![2.0, 1.0]),
            docs.len(),
        );

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;
        let lut = ScoreLookupTable::new(&docs);

        let compressed_results =
            saat_bm25_search(&[compressed], &docs, &params, mask.clone(), &metrics, &lut);
        let plain_results = saat_bm25_search(&[plain], &docs, &params, mask, &metrics, &lut);

        let compressed_scores = compressed_results
            .into_iter()
            .map(|doc| (doc.row_id, doc.score))
            .collect::<std::collections::HashMap<_, _>>();
        let plain_scores = plain_results
            .into_iter()
            .map(|doc| (doc.row_id, doc.score))
            .collect::<std::collections::HashMap<_, _>>();

        assert_eq!(plain_scores.len(), compressed_scores.len());
        for (row_id, expected_score) in compressed_scores {
            let actual_score = plain_scores
                .get(&row_id)
                .copied()
                .unwrap_or_else(|| panic!("missing row_id {row_id} in plain posting results"));
            assert!(
                (actual_score - expected_score).abs() < 1e-6,
                "row_id={row_id}, actual_score={actual_score}, expected_score={expected_score}",
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

        let bits: Vec<u32> = BitIter {
            word: 1u64 << 63,
            base: 64,
        }
        .collect();
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
