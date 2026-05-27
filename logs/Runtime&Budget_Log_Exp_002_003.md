# Runtime and Budget Log: Exp_002 & Exp_003
**Date:** 2026-04-28

---

## Exp_002 — Ridge Regression on Engineered Features

### 1. Measured Runtime
* **Total Time:** ~1.0s wall (sklearn fit + 5-fold OOF + write)
* **Average Time per Lead:** ~0.016 seconds
* **Components:** CSV parsing + featurization + 5× Ridge fit + cross_val_predict + TSV write. No network calls.

### 2. Estimated API / Data Cost
* **Data Source:** Apollo.io firmographics (already pulled, same as baseline)
* **Cost per Credit:** $0.02
* **Total Leads Processed:** 62
* **Total Estimated Cost:** $1.24
* **Marginal cost vs. V0 baseline:** $0.00 — same input data, only the modeling layer changed

### 3. Scalability Projection
* 1,000 leads: ~16 seconds runtime, ~$20 data cost
* 10,000 leads: ~3 minutes runtime, ~$200 data cost
* Bottleneck for scaling is data acquisition, not compute. Ridge fit is O(n × p²) ≈ trivially small at p=9.

---

## Exp_003 — Management Depth Scraper Signal

### 1. Measured Runtime
| Mode | Wall Time | Per-Firm | Notes |
|---|---|---|---|
| **Cold scrape (empty cache)** | 259.7s (~4.3 min) | ~4.2s / firm | 61 successful + 1 timeout-bound failure |
| **Warm cache (re-run)** | 1.2s | ~0.02s / firm | All depths read from `logs/scrape_cache.json` |

* **Per-firm runtime budget compliance:** 4.2s avg / firm < 10s budget from `program.md`. ✓
* **Worst case:** the single failure consumed the full 4s timeout × ~3 path attempts before being cached as `homepage_fetch_failed`. Subsequent runs skip it.

### 2. Estimated API / Data Cost
* **Data Source:** direct HTTP scraping of company-owned websites (no third-party API)
* **Cost per Request:** $0.00 (own bandwidth only; no rate-limited or paid endpoint)
* **Total HTTP Requests (cold run):** 62 homepages + ~62 team-page attempts ≈ 124 requests
* **Marginal cost vs. exp_002:** $0.00
* **Cumulative cost through exp_003:** still $1.24 (Apollo data, unchanged)

### 3. Scalability Projection
* **Cold scrape, 1,000 leads:** ~70 minutes serial. Concurrent fetching with `requests` + a `ThreadPoolExecutor` (10 workers) brings this to ~7 min. Cached re-runs would remain ~10 seconds.
* **Cold scrape, 10,000 leads:** ~12 hours serial; ~1.2 hours at 10× concurrency. Practical for a one-time enrichment, not for per-request inference.
* **Operational risk:** scraping introduces external dependencies — sites can return 4xx/5xx, change their HTML, or rate-limit. The disk cache mitigates this for *re-runs*, but the cold path remains the brittle step. Mitigation strategies for production: respect `robots.txt`, add exponential backoff on transient errors, persist a cache TTL to refresh stale entries.

### 4. Cumulative Budget Through Exp_003
| Item | Amount |
|---|---|
| Apollo firmographics (62 firms × $0.02) | $1.24 |
| Web scraping (own bandwidth) | $0.00 |
| Compute (local sklearn) | $0.00 |
| **Total** | **$1.24** |

### 5. Scraper-Specific Diagnostics
* **Cache file:** `logs/scrape_cache.json` (5,954 bytes, 62 entries)
* **Reachability:** 61 / 62 firms (98.4%)
* **mgmt_depth distribution:** min 0, mean 2.9, max 9
* **Failure modes observed:** 1 × `homepage_fetch_failed` (DNS/timeout)
