# Personalized PageRank Movie Recommendations — A06 (LLM4Rec, HSE)

**Live demo:** https://konstantinpatrushev.github.io/recsys_hw6/

Solution of [dryjins/RecSys-LLMs/week6](https://github.com/dryjins/RecSys-LLMs/tree/main/week6):
global PageRank and user-seeded Personalized PageRank (PPR) on the user–movie graph of positive
MovieLens-100k training ratings, computed with TensorFlow.js, and compared with popularity and
matrix factorization under one retrieval protocol.

## Run

The data layer uses `fetch`, so serve the folder over HTTP:

```bash
python3 -m http.server 8000
# open http://localhost:8000/index.html
```

Then: **Build graph** → **Run tests** → **Run alpha sweep (validation)** (and set the alpha slider
to the selected value) → **Run PageRank** / **Run PPR** → **Evaluate Recall@10 — all four methods**
(trains the MF reference on first use) → **Recommend Top-10** / **Run two-user demo** → the three
export buttons.

Dependencies actually used: TensorFlow.js **4.22.0** (CDN, as in the starter), d3 v7.9.0 (vendored,
starter's copy), Google Chrome 155.0.8059.85 on macOS 26.5.1, MacBook Air (Apple M5), WebGL backend. The checks in `tests/` need Node.js
(tested with v26.12.0); no npm packages.

## What was implemented

The seven `TODO(hw6)` stubs. `index.html`, `style.css`, `data.js`, `u.data`, `u.item` and
`vendor/d3.min.js` are the starter's files unchanged; the rest of `script.js` / `graph.js` is the
starter's scaffolding unchanged.

| File | Function | Notes |
|---|---|---|
| `graph.js` | `buildGraph` | node universe = users and movies observed in the training split, users first; `R[u][i] = 1` for a training rating ≥ 4; no padding node |
| `graph.js` | `rowNormalize` | dense `Q` with rows divided by degree, zero rows + `danglingMask` for zero-degree nodes |
| `graph.js` | `pageRank` / `personalizedPageRank` | power iteration `p ← α(Qᵀp + d·v) + (1−α)v` with `tf.matMul` on the two off-diagonal blocks (`R` scaled by user degree, `R` scaled by movie degree); stops at L1 < 1e-6 or 500 iterations |
| `script.js` | `movieScores`, `recommendTopK`, `recallAtK` | movie-node scores; candidate pool = training movies; test excludes train + validation ratings, validation excludes training; ties by ascending id; macro Recall@10 on the shared cohort |

**Restart argument of `personalizedPageRank`.** The starter's JSDoc calls the first argument a node
index, but the starter's own adapter (`hw6AdapterPPR`) passes the raw user id. The implementation
follows the callers: a raw user id on the MovieLens path, a node index when `opts.transition` is
given (the reference-graph checks). Treating the user id as a node index would restart at the
next user's node, because ids start at 1 and node indices at 0.

## Results (test, shared cohort of 861 users)

| Method | Recall@10 | Mean list popularity |
|---|---|---|
| Popularity | 0.0760 | 272 |
| Global PageRank (α = 0.50) | 0.0797 | 258 |
| **PPR (α = 0.50, selected on validation)** | **0.1122** | 244 |
| MF, starter's reference (collapsed to the mean, see report) | 0.0211 | 97 |
| MF, my assignment-5 checkpoint (epoch 2) | 0.0175 | 44 |

Exports: `verification.json`, `results.csv`, `recommendations.csv` (from the page's own exporters);
`results_supplement.csv` (assignment-5 MF and the independent recomputation).

## Checks

| Path | What |
|---|---|
| `tests/runner.html`, `tests/run_browser.sh` | drive the real page (`index.html` in an iframe) headless in Chrome: self-checks, convergence, alpha sweep, comparison, lists, demo, exports |
| `tests/independent.mjs` | float64 sparse re-implementation sharing no code with `graph.js` / `script.js`; reproduces every recall value and both users' lists |
| `tests/list_stats.mjs` | overlaps, popularity, training histories of the report users |
| `evidence/` | all outputs |
