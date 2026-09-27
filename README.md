# Business Entity Resolution — Amazon ML Challenge 2026

For each Source 1 business, find all matching Source 2 / Source 3 records. CPU-only
pipeline: text normalisation → multi-channel sparse TF-IDF blocking → ~60 pairwise
features → LightGBM → "each S2/S3 record goes to at most one S1" decision rule with a
threshold tuned directly on macro F0.5.

Current result: **2-fold out-of-fold macro F0.5 ≈ 0.942** on train (threshold 0.35).

## Setup

```bash
pip install -r requirements.txt        # Python 3.10+ (developed on 3.13)
unzip dataset.zip                      # gives student_resource/ (dataset/, utils/)
```

## Run everything (one command)

```bash
bash run_all.sh path/to/student_resource work output
```

or step by step (same thing):

```bash
python src/convert.py  --data path/to/student_resource/dataset --work work   # TSV -> parquet (~1 min)
python src/prep.py     --work work                        # normalise all 6 files (~5 min)
python src/pipeline.py build   --work work --split train  # blocking + features, 30% of train queries
python src/pipeline.py build   --work work --split test   # blocking + features, all test queries
python src/pipeline.py train   --work work --sample 0.25  # 2-fold OOF LightGBM + threshold search
python src/pipeline.py predict --work work --out output   # writes the two TSVs
python path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir path/to/student_resource/dataset/test
```

### GPU training (optional)

Training can run on an NVIDIA GPU with XGBoost (Apache-2.0, `pip install xgboost`):

```bash
python src/pipeline.py train   --work work --model xgb --device cuda --sample 1.0 --rounds 800
python src/pipeline.py predict --work work --out output
```

With a GPU (and enough RAM) use `--sample 1.0` to train on every train query instead of 25%,
and more rounds. The CV macro F0.5 is printed per threshold. Compare it with the LightGBM run
(0.942) before switching. Blocking and feature building stay on the CPU (sparse top-k search +
rapidfuzz); they are the long steps, and a machine with more RAM/cores speeds them up.

`predict --reuse` rewrites the TSVs from saved predictions without rescoring.

`train --final_thr 0.35` skips the OOF pass (saves ~15 min) and trains only the final model.
`build` resumes per country: if a run dies, delete the partial `work/feat_<split>/<Country>_*.parquet`
files of the country that was in progress and rerun.

### Timings / memory (8 GB RAM laptop, 10 cores)

| step | time |
|---|---|
| convert + prep | ~6 min |
| build train (3.1M queries) | ~50 min |
| build test (10M queries) | ~2–2.5 h |
| train (OOF + final) | ~25 min |

On 8 GB RAM **never run two steps at the same time** (it OOMs). With 32 GB+ RAM you can run
`build --split train` and `build --split test` in parallel (separate work dirs are not needed).

## Knobs worth trying on a bigger machine

* `src/block.py` → `DF_CAP`: tokens more frequent than this fraction of the index are dropped
  for speed. 0.003 gives ~92% India blocking recall, 0.01 gives ~94.5% but ~3× slower.
* `src/block_v2.py` (experimental): adds a 4th "cross" channel of conjunction tokens
  (name skeleton / address code × address word, e.g. `lksm|udaipur`) to recover truncated
  addresses and transliterated names. Evaluate with
  `python src/eval_block_v2.py --work work --frac 0.05 --country India` vs `src/eval_block.py`;
  to use it, change `from block import` → `from block_v2 import` in `src/pipeline.py`
  (then rebuild train + test and retrain).
* `pipeline.py` → `TRAIN_QUERY_FRAC` (0.3) and `train --sample` (share of train queries used).

## Source layout

| file | purpose |
|---|---|
| `src/convert.py` | streaming TSV → parquet |
| `src/normalize.py` | transliteration (anyascii), accent / leetspeak / punctuation cleanup, legal-form and address-abbreviation canonicalisation, consonant skeletons, address numbers |
| `src/prep.py` | applies normalisation to every source file (chunked) |
| `src/block.py` | hashed TF-IDF, 3 retrieval channels (full / name / address), per-country sparse top-k search, streamed to disk |
| `src/features.py` | query-side context features + pairwise name / address / house-number similarity features |
| `src/metric.py` | exact competition metric (per-S1 F0.5, macro, singletons included) |
| `src/pipeline.py` | build / train / predict orchestration, decision rule, TSV writers |
| `src/eval_block.py` | blocking recall diagnostics |

No external data, APIs or pretrained models; the only model is LightGBM (MIT licence).
