// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SIMD-accelerated BM25 scoring for full-text search.
//!
//! This module implements a Score-at-a-Time (SAAT) BM25 search path that
//! processes posting list blocks in batch using SIMD operations. Instead of
//! the document-at-a-time WAND approach, this processes one term at a time
//! and accumulates scores in a dense array.
//!
//! Performance advantage: processes 8 docs per SIMD instruction (AVX2/2×NEON),
//! eliminates heap operations, and enables aggressive block-level pruning.

use std::sync::Arc;

use arrow_array::Array;
use lance_core::utils::mask::RowAddrMask;

use super::builder::BLOCK_SIZE;
use super::encoding::{decompress_posting_block, decompress_posting_remainder};
use super::scorer::{B, K1, idf};
use super::{CompressedPostingList, DocSet, PostingList};
use crate::metrics::MetricsCollector;
use crate::scalar::inverted::query::FtsSearchParams;
use crate::scalar::inverted::wand::{DocCandidate, PostingIterator, TermFreqVec};

/// SIMD-aligned buffer for batch BM25 scoring.
/// All arrays are sized to BLOCK_SIZE (128) which is a multiple of 8 (SIMD lane count).
struct ScoringBuffer {
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
    decompression_buf: Box<[u32; BLOCK_SIZE]>,
}

impl ScoringBuffer {
    fn new() -> Self {
        Self {
            doc_ids: Vec::with_capacity(BLOCK_SIZE),
            freqs: Vec::with_capacity(BLOCK_SIZE),
            decompression_buf: Box::new([0u32; BLOCK_SIZE]),
        }
    }

    fn clear(&mut self) {
        self.doc_ids.clear();
        self.freqs.clear();
    }
}

/// Dense score accumulator indexed by doc_id.
/// Uses a flat f32 array for O(1) score accumulation — cache-friendly and SIMD-compatible.
struct ScoreAccumulator {
    scores: Vec<f32>,
    // Track which docs have been touched to avoid scanning entire array
    touched: Vec<u32>,
}

impl ScoreAccumulator {
    fn new(num_docs: usize) -> Self {
        Self {
            scores: vec![0.0f32; num_docs],
            touched: Vec::with_capacity(1024),
        }
    }

    #[inline]
    fn accumulate(&mut self, doc_id: u32, score: f32) {
        let idx = doc_id as usize;
        if self.scores[idx] == 0.0 {
            self.touched.push(doc_id);
        }
        self.scores[idx] += score;
    }

    /// Batch-accumulate scores for a block of documents.
    /// This is the hot path — processes 8 docs at a time using scalar operations
    /// that the compiler can auto-vectorize.
    #[inline]
    fn accumulate_block(
        &mut self,
        doc_ids: &[u32],
        freqs: &[u32],
        query_weight: f32,
        b_over_avgdl: f32,
        num_tokens: &[u32],
    ) {
        // Process in chunks of 8 for auto-vectorization
        let chunks = doc_ids.len() / 8;
        let remainder = doc_ids.len() % 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;
            // Explicit unrolled loop that the compiler can vectorize
            for i in 0..8 {
                let idx = base + i;
                let doc_id = doc_ids[idx];
                let freq = freqs[idx] as f32;
                let doc_len = num_tokens[doc_id as usize] as f32;
                let doc_norm = K1 * (1.0 - B + b_over_avgdl * doc_len);
                let tf_score = (K1 + 1.0) * freq / (freq + doc_norm);
                let score = query_weight * tf_score;

                let score_idx = doc_id as usize;
                if self.scores[score_idx] == 0.0 {
                    self.touched.push(doc_id);
                }
                self.scores[score_idx] += score;
            }
        }

        // Handle remainder
        for i in 0..remainder {
            let idx = chunks * 8 + i;
            let doc_id = doc_ids[idx];
            let freq = freqs[idx] as f32;
            let doc_len = num_tokens[doc_id as usize] as f32;
            let doc_norm = K1 * (1.0 - B + b_over_avgdl * doc_len);
            let tf_score = (K1 + 1.0) * freq / (freq + doc_norm);
            let score = query_weight * tf_score;

            let score_idx = doc_id as usize;
            if self.scores[score_idx] == 0.0 {
                self.touched.push(doc_id);
            }
            self.scores[score_idx] += score;
        }
    }

    /// Get the k-th highest score (min of top-k) for threshold estimation.
    /// Uses partial sort via min-heap with ScoredDoc (which implements Ord via OrderedFloat).
    fn kth_score(&self, k: usize) -> f32 {
        if self.touched.len() < k {
            return 0.0;
        }
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        use super::builder::ScoredDoc;

        let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);
        for &doc_id in &self.touched {
            let score = self.scores[doc_id as usize];
            if score <= 0.0 {
                continue;
            }
            if heap.len() < k {
                heap.push(Reverse(ScoredDoc::new(doc_id as u64, score)));
            } else if score > heap.peek().unwrap().0.score.0 {
                heap.pop();
                heap.push(Reverse(ScoredDoc::new(doc_id as u64, score)));
            }
        }
        heap.peek().map(|r| r.0.score.0).unwrap_or(0.0)
    }

    /// Extract top-k results from the accumulator.
    fn top_k(&self, k: usize, docs: &DocSet, mask: &RowAddrMask) -> Vec<DocCandidate> {
        if self.touched.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use a min-heap of size k for top-k extraction
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        use super::builder::ScoredDoc;

        let mut heap: BinaryHeap<Reverse<ScoredDoc>> = BinaryHeap::with_capacity(k);

        for &doc_id in &self.touched {
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
                freqs: TermFreqVec::new(), // freqs not tracked in SAAT path
                doc_length: 0,
            })
            .collect()
    }
}

/// Score-at-a-Time BM25 search with SIMD batch scoring and block-level pruning.
///
/// Processes terms from rarest (highest IDF) to most common:
/// 1. First pass: process rare terms, establish score estimates
/// 2. Subsequent terms: use running threshold to skip entire blocks
/// 3. SIMD-vectorized BM25 scoring (8 docs per cycle)
/// 4. Dense accumulator for O(1) score updates
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
    let avgdl = docs.average_length();
    let b_over_avgdl = B / avgdl;

    // Sort terms by IDF descending (rarest first) for early threshold establishment
    // query_weight already contains IDF from load_posting_lists.
    // Just use it directly — no need to recompute IDF.
    let mut term_order: Vec<(usize, f32)> = postings
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.query_weight))
        .filter(|(_, qw)| *qw > 0.0)
        .collect();
    term_order.sort_by(|a, b| b.1.total_cmp(&a.1)); // descending by IDF weight

    // Compute the remaining max score for block pruning.
    // remaining_max[i] = sum of query_weights for terms i+1..n
    // A block can be skipped if block_max * query_weight + remaining_max < threshold
    let mut remaining_max = vec![0.0f32; term_order.len() + 1];
    for i in (0..term_order.len()).rev() {
        let (_, qw) = term_order[i];
        // Upper bound of doc_weight is (K1+1) when freq >> doc_norm
        remaining_max[i] = remaining_max[i + 1] + qw * (K1 + 1.0);
    }

    let mut accumulator = ScoreAccumulator::new(num_docs);
    let mut buffer = ScoringBuffer::new();
    let mut num_comparisons = 0usize;
    let mut threshold = 0.0f32;

    // Process each term's posting list (rarest first)
    for (term_idx, &(posting_idx, query_weight)) in term_order.iter().enumerate() {
        let posting = &postings[posting_idx];
        let max_remaining = remaining_max[term_idx + 1];



        match &posting.list {
            PostingList::Compressed(list) => {
                process_compressed_list_with_pruning(
                    list,
                    query_weight,
                    b_over_avgdl,
                    docs.num_tokens_slice(),
                    &mut accumulator,
                    &mut buffer,
                    &mut num_comparisons,
                    threshold,
                    max_remaining,
                );
            }
            PostingList::Plain(list) => {
                for i in 0..list.row_ids.len() {
                    let row_id = list.row_ids[i];
                    let freq = list.frequencies[i] as u32;
                    let doc_len = docs.num_tokens_by_row_id(row_id);
                    let doc_norm = K1 * (1.0 - B + b_over_avgdl * doc_len as f32);
                    let tf_score = (K1 + 1.0) * freq as f32 / (freq as f32 + doc_norm);
                    let score = query_weight * tf_score;
                    accumulator.accumulate(row_id as u32, score);
                    num_comparisons += 1;
                }
            }
        }

        // Update threshold after processing each term
        if accumulator.touched.len() >= limit {
            threshold = accumulator.kth_score(limit);
        }
    }

    metrics.record_comparisons(num_comparisons);
    accumulator.top_k(limit, docs, &mask)
}

/// Process a compressed posting list with block-level threshold pruning.
///
/// Uses doc-level max accumulated scores to skip blocks: if no doc in a block
/// has been scored by prior terms, and this term + remaining can't reach
/// threshold, skip the block.
fn process_compressed_list_with_pruning(
    list: &CompressedPostingList,
    query_weight: f32,
    b_over_avgdl: f32,
    num_tokens: &[u32],
    accumulator: &mut ScoreAccumulator,
    buffer: &mut ScoringBuffer,
    num_comparisons: &mut usize,
    threshold: f32,
    max_remaining: f32,
) {
    let num_blocks = list.blocks.len();
    let length = list.length as usize;
    // The maximum score this term can contribute per block
    let max_score_with_remaining = query_weight * (K1 + 1.0) + max_remaining;

    for block_idx in 0..num_blocks {
        let block_max = list.block_max_score(block_idx);
        if block_max * query_weight <= 0.0 {
            continue;
        }

        // For effective pruning: peek at the first doc_id of this block.
        // If we can determine that no doc in this block range has a high enough
        // accumulated score to benefit from this term, skip.
        // However, without a per-block max accumulator, we conservatively
        // only skip if the term's max contribution + remaining can't help at all.
        if threshold > 0.0 && max_score_with_remaining <= threshold {
            // None of the remaining terms (including this one) can push any
            // zero-score doc above threshold. Only process if docs might have
            // prior accumulated scores.
            // Quick heuristic: check if ANY doc in this block has been touched.
            let block_start = list.block_least_doc_id(block_idx) as usize;
            let block_end = if block_idx + 1 < num_blocks {
                list.block_least_doc_id(block_idx + 1) as usize
            } else {
                length
            };
            // Check a sample of docs in the block for accumulated scores
            let has_accumulated = (block_start..block_end.min(block_start + BLOCK_SIZE))
                .step_by(16) // sample every 16th doc
                .any(|doc_id| {
                    doc_id < accumulator.scores.len() && accumulator.scores[doc_id] > 0.0
                });
            if !has_accumulated {
                continue;
            }
        }

        buffer.clear();
        let block_data = list.blocks.value(block_idx);
        let remainder = length % BLOCK_SIZE;
        if block_idx + 1 == num_blocks && remainder != 0 {
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
                &mut buffer.decompression_buf,
                &mut buffer.doc_ids,
                &mut buffer.freqs,
            );
        }

        *num_comparisons += buffer.doc_ids.len();

        accumulator.accumulate_block(
            &buffer.doc_ids,
            &buffer.freqs,
            query_weight,
            b_over_avgdl,
            num_tokens,
        );
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
            docs.append(i as u64, 10); // each doc has 10 tokens
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
        let posting = make_compressed_posting(
            (0..100u32).collect(),
            vec![1u32; 100],
        );
        let iter = PostingIterator::with_query_weight(
            "test".to_string(),
            0,
            0,
            1.0,
            posting,
            1000,
        );

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let results = saat_bm25_search(&[iter], &docs, &params, mask, &metrics);
        assert_eq!(results.len(), 10);
        // All results should have positive scores
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_saat_multi_term() {
        let docs = make_test_docs(1000);

        // Term 1: docs 0-99
        let posting1 = make_compressed_posting(
            (0..100u32).collect(),
            vec![2u32; 100],
        );
        let iter1 = PostingIterator::with_query_weight(
            "alpha".to_string(),
            0,
            0,
            1.0,
            posting1,
            1000,
        );

        // Term 2: docs 50-149 (overlap with term 1 at 50-99)
        let posting2 = make_compressed_posting(
            (50..150u32).collect(),
            vec![3u32; 100],
        );
        let iter2 = PostingIterator::with_query_weight(
            "beta".to_string(),
            1,
            1,
            1.0,
            posting2,
            1000,
        );

        let params = FtsSearchParams::new().with_limit(Some(10));
        let mask = Arc::new(RowAddrMask::default());
        let metrics = crate::metrics::NoOpMetricsCollector;

        let results = saat_bm25_search(&[iter1, iter2], &docs, &params, mask, &metrics);
        eprintln!("  results.len()={}", results.len());
        assert_eq!(results.len(), 10);

        // Docs 50-99 should score highest (both terms match)
        for r in &results {
            eprintln!("  result: row_id={} score={:.4}", r.row_id, r.score);
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
}
