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
