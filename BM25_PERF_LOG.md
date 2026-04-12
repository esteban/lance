# BM25 Search Pipeline Performance Optimization Log

## Baseline (branch: `perf/bm25-optimization-baseline`)

**System**: macOS Darwin 25.3.0, Apple Silicon
**Dataset**: 1M docs, Zipf(1.1) distribution, 100k vocab, 1-100 words/doc
**Queries**: 1024 pre-generated, 15 tokens each from first doc's vocabulary

| Benchmark | Time | Notes |
|-----------|------|-------|
| invert_search(1000000) | **7.17 ms** | OR search, top-10, 15-token query |
| invert_phrase_search(1000000) | **12.84 ms** | AND + phrase_slop=0, 2-token |
| invert_indexing(1000000) | 10.51 s | Build, no positions |
| invert_indexing_with_positions(1000000) | 20.11 s | Build, with positions |

**Primary target**: `invert_search` (7.17ms) — goal is 100x → ~70us

## Optimization Log

| # | Branch | Change | Search (ms) | Phrase (ms) | Delta | Cumulative |
|---|--------|--------|-------------|-------------|-------|------------|
| 0 | baseline | -- | 7.17 | 12.84 | -- | 1.00x |
| 1 | opt-01-smallvec | SmallVec<16> for DocCandidate.freqs | 7.31 | -- | ~0% | 1.00x |
| 2 | opt-01-smallvec | Cache doc_id in PostingIterator | 5.50 | 12.23 | -25% | 1.30x |
| 3 | opt-01-smallvec | Cache doc_id in HeadPosting | 5.39 | -- | -2% | 1.33x |
| 4 | opt-01-smallvec | Single-partition fast path (skip re-score) | 5.39 | 12.22 | ~0% | 1.33x |
| 5 | opt-01-smallvec | Precompute B/avgdl reciprocal | 5.37 | -- | ~0% | 1.34x |

## Phase 2 Summary

Achieved **1.34x** speedup through:
- **Dominant win**: Caching `DocInfo` in `PostingIterator` eliminates repeated decompression lookups (-25%)
- **Supporting wins**: HeadPosting doc_id cache, single-partition fast path, reciprocal precompute
- **No impact**: SmallVec for freqs (allocation not dominant at this scale)

Remaining 5.37ms is dominated by: WAND traversal (heap ops + block decompression + scoring)

## Phase 3: Algorithmic Analysis

### WAND Inner Loop Profile (15-token OR, top-10, 1M docs)

| Metric | Average | Notes |
|--------|---------|-------|
| Inner loop iterations | 73,000 | The real work count |
| update_max_scores calls | 591 | One per block boundary (~128-doc window) |
| Threshold prunes | 103,000 | Docs rejected by score check |
| Final candidates (comparisons) | 114 | Docs that pass all filters |
| us per inner iteration | 0.073 | 5.3ms / 73K iterations |

### Root Cause Analysis

The WAND evaluates **73K docs per query** despite only returning 10 results.
The "114 comparisons" metric only counts docs that pass the mask filter.

With 15 Zipf-distributed terms:
- Each term's block-max score ~1.0
- Combined block-max across all terms ~15.0
- Top-10 threshold ~13.0
- Block-max sum > threshold for **almost every block**
- Block-level pruning rarely triggers
- Result: every doc in the union of posting lists is checked individually

### Optimization Path Forward

Block-level skip (implemented + tested): **no impact** because combined block-max of 15 terms always exceeds threshold.

**Required approach**: MaxScore essential/non-essential term partitioning:
- Sort terms by IDF * max_doc_weight
- Only iterate through "essential" terms whose individual max score > threshold gap
- Non-essential terms checked lazily only for candidates from essential terms
- Expected reduction: 73K iterations -> ~5-10K (only essential term posting lists)

Alternative: Score-at-a-Time (SAAT) with hash accumulator
- Process one term at a time, accumulate in HashMap<u32, f32>
- Avoids heap overhead entirely
- Memory: O(|union of posting lists|) which can be large for Zipf

### Attempted Optimizations (Phase 3, ineffective)

| Attempt | Why Ineffective |
|---------|----------------|
| Block-level skip on prune | Combined block-max of 15 terms (~15) > threshold (~13) |
| MaxScore term repartitioning | Only 1-3 terms moved; existing tail overflow logic conflicts |
| BinaryHeap::from(vec) rebuild | O(n) vs O(n log n) but n=15 is too small to matter |

### Per-Iteration Analysis

At 73K iterations / 5.3ms = **73ns per iteration**. This includes:
- BinaryHeap peek/pop/push: O(log 15) ~ 4 operations
- Score computation: O(|lead|) ~ 1-3 multiplications
- Threshold check + push_back_leads: O(|lead| x log 15)

**73ns for ~60 operations = ~1.2ns per operation** — this is near-optimal for modern CPUs.
The WAND is already well-optimized per-iteration. The 73K iteration count is inherent
to the DAAT + block-max approach with 15 terms on Zipf data.

### True 100x Path

To achieve 100x (5ms -> 50us), need fundamentally different approaches:
1. **Precomputed impact-ordered indices**: Store docs sorted by BM25 impact per term, enabling early termination after scanning top-impact blocks
2. **WAND with larger blocks**: 1024 instead of 128 would reduce update_max_scores from 600 to 75 calls (format change)
3. **Query-time term elimination**: For top-10 queries, terms beyond the top-3 by IDF contribute <10% of final score; can be deferred to re-scoring
4. **SIMD batch scoring**: Process 8 docs simultaneously using AVX2
5. **Tiered index**: Pre-cluster docs by quality, search high-quality tier first

## Current State

| Benchmark | Baseline | Current | Speedup |
|-----------|----------|---------|---------|
| invert_search(1M) | 7.17 ms | 5.32 ms | **1.35x** |
| invert_phrase_search(1M) | 12.84 ms | 12.22 ms | **1.05x** |

## Changes Shipped (on perf/bm25-opt-01-smallvec + perf/bm25-opt-phase3-algo)

### Phase 2 — Rust-Level Optimizations (1.35x)
1. **Cache DocInfo in PostingIterator** — eliminates repeated decompression lookups on `doc()` calls. Dominant win at -25%.
2. **Cache doc_id in HeadPosting** — avoids Box pointer chase in heap comparisons
3. **SmallVec<16> for DocCandidate.freqs** — eliminates heap allocation for term frequencies
4. **Single-partition fast path** — skip IDF re-scoring when partition-local IDF == global IDF
5. **Precompute B/avg_doc_length** — replace division with multiplication in BM25 scoring
6. **Add score field to DocCandidate** — carry WAND-computed scores through pipeline

### Phase 3 — Profiling Infrastructure
7. **WAND profiling counters** — inner_loop_iters, update_max_calls, threshold_prunes (gated by `WAND_PROFILE` env var)
8. **Benchmark instrumentation** — comparison count + timing breakdown in benches/inverted.rs
9. **BinaryHeap::from(vec) rebuild** — O(n) heapify instead of O(n log n) individual pushes

## Phase 4: SAAT with SIMD Batch Scoring

Implemented Score-at-a-Time (SAAT) alternative search path with:
- Dense f32 score accumulator indexed by doc_id
- Block-by-block decompression + 8-doc batch scoring (auto-vectorizable)
- Terms processed rarest-first for early threshold establishment
- Block-level pruning with sampling-based accumulated score check

| Path | Time | Per-Doc Cost | Docs Processed | Notes |
|------|------|-------------|----------------|-------|
| WAND (baseline) | 5.32 ms | 73 ns | ~73K | Block-max pruning effective |
| SAAT v1 (initial) | 11.3 ms | 23 ns | ~500K | No norms precompute, Vec touch |
| SAAT v2 (optimized) | 7.44 ms | 15 ns | ~500K | Precomputed norms, bitset, unsafe |
| SAAT parallel (rayon) | 7.12 ms | -- | ~500K | Merge phase (40MB) limits gains |
| SAAT + block pruning | 7.79 ms | -- | ~500K | Pruning overhead > benefit |
| SAAT + u16 quantized | 7.30 ms | 15 ns | ~500K | u16 cache density gain (Mackenzie et al.) |

### SAAT Optimization History

| Optimization | Impact | Notes |
|-------------|--------|-------|
| Precomputed doc_norm array | -34% | Eliminates N_terms redundant computations |
| Bitset instead of Vec for touch tracking | -5% | Branchless, no allocation pressure |
| unsafe get_unchecked in hot loop | -3% | Eliminate bounds checks |
| Multi-block decode (4 blocks/batch) | -2% | Better L1 cache utilization |
| Software prefetch of doc_norms | -1% | Hardware prefetcher already good |
| Block-level pruning | +4% (worse) | Overhead exceeds pruning benefit |
| Parallel rayon term processing | +5% (worse) | 10x4MB merge dominates |
| Two-phase threshold + pruned path | +5% (worse) | threshold_fast scan too expensive |
| u16 quantized accumulators | -2% | Cache density benefit limited at 1M docs |
| Parallel rayon (thread-local accs) | -4% | Merge of 10x2MB arrays negates gains |

### Reference: Mackenzie et al. (TOIS 2023)

Key techniques from "Efficient DaaT and SaaT Query Evaluation for Learned Sparse Representations":
- **u8/u16 accumulators**: 1.3-1.9x speedup on 8.8M docs from cache density
- **Query-specific impact rescaling**: dynamic [0, Mq] → [0, 255] mapping per query
- **Heap score caching**: avoid tie-breaking comparisons (1.1x)
- **Anytime DaaT (clustered)**: 1.3-1.8x via document reordering + cluster-priority traversal
- **Impact-ordered indices**: organize posting lists by score (high→low) for SaaT early termination

**Finding**: SAAT's 3x per-doc advantage from batch scoring is negated by processing
7x more data (full union of posting lists vs WAND's pruned candidates).
The WAND's block-max pruning is the dominant factor — it eliminates 93% of the work.

SAAT would win when:
- Number of terms is small (1-3) → union is similar size to WAND candidates
- Block-max scores are uniformly high → WAND can't prune effectively
- Scoring is the bottleneck → SIMD batch scoring dominates

## Architecture Notes

The BM25 search pipeline in Lance uses **Block-Max WAND (BMW)** with:
- BitPacker4x compressed posting lists (128-element blocks)
- Block-max scores stored per block for early termination
- Head/tail/lead three-partition posting iterator management
- UnsafeCell-based decompression cache for block reuse

The WAND implementation at `rust/lance-index/src/scalar/inverted/wand.rs` (~1700 lines) is already
well-optimized for its algorithmic class. The **73ns per-iteration cost** is near CPU-optimal
for the heap + score + advance operations at query term counts of 10-15.
