# Task: produce an improved submission on the GPU machine

> You can paste this whole file as a prompt into a coding agent (e.g. Claude Code) running in
> this repo on the GPU machine, or follow it by hand.

## Hard deadline

The challenge closes **27 Sep 2026, 23:59 IST**. The team must have `matching_results.tsv`
uploaded well before that. **Send the finished output files back by 22:30 IST at the latest.**
If a step is running late, skip the optional steps. A baseline submission already exists, so
a late or broken improvement is worth nothing.

## Context

Amazon ML Challenge 2026: **business entity resolution**. Three sources of business records
(`entity_id`, `business_name`, `business_address`, `country`). Source 1 (S1) is deduplicated.
For every S1 record, find all S2/S3 records that are the same business.

* Metric: **F0.5 per S1 entity, macro-averaged** (precision weighted 2× over recall). An S1
  with no true match scores 1.0 only if we predict nothing for it.
* Train covers US + India. **Test also has France** (never seen in training). Nothing may be
  hard-coded per country.
* Data is in `dataset.zip` → `student_resource/dataset/{train,test}` (TSV, tab-separated).
  The validator is in `student_resource/utils/validate_submission.py`.

### Rules (breaking these disqualifies the team)

* **No external data, APIs, geocoding, or web lookups.** Only the provided training data.
* Any model must be **MIT or Apache-2.0 licensed and ≤ 8B parameters**. LightGBM (MIT) and
  XGBoost (Apache-2.0) are fine. Don't add pretrained models or other libraries without
  checking their licence.
* Output files must pass the validator.

### What the pipeline does (see README.md for details)

1. `convert.py` / `prep.py`: TSV → parquet, text normalisation (transliteration of Indic scripts,
   accents, leetspeak, legal forms, address abbreviations, consonant skeletons).
2. `pipeline.py build`: **blocking** (hashed TF-IDF, 3 channels: full / name / address,
   top-k S1 per S2/S3 record, per country) + ~58 pairwise features. Written in chunks to
   `work/feat_<split>/`. Train uses a 30% sample of queries.
3. `pipeline.py train`: 2-fold out-of-fold model, threshold tuned on macro F0.5, then a final model.
4. `pipeline.py predict`: scores test pairs. Decision rule: each S2/S3 record goes to its single
   best S1 if probability ≥ threshold. Writes `output/matching_results.tsv` and
   `output/candidate_pairs.tsv`.

### UPDATE 17:15 IST: v2 blocking (pushed; use this version)

The first leaderboard submission scored **0.939** (the CV said 0.942, so the CV is reliable).
Error analysis: even a perfect matcher on the old candidates would only reach ~0.959, so the
main loss was blocking. Cause: the generator draws names/streets from a small vocabulary, so
single words are common and got dropped by the DF cap, leaving only house numbers, which
the noise corrupts. v2 adds a **"pair" channel** of conjunction tokens (name×name,
name×address word, address×address word):
* US blocking recall 94% → **98.1%**, India 92% → **95.5%** (the rest is mostly transliteration).
* Candidates are capped at 10 per query; features gained `s_pair*` columns.

The laptop is rebuilding with v2 now (train `--frac 0.2`), but it takes ~4–5 h there, which
is very tight for the deadline. **On the GPU machine, run v2 with `--frac 0.3` for train (and pass
`--frac 0.3` to `train`).** Step 1 (blocking config choice) is optional now; v2 is the default.

### Old baseline (laptop, 8 GB RAM)

* LightGBM, CPU, 25% of train queries, 400 rounds → **CV macro F0.5 = 0.9424** at threshold 0.35.
* Main weakness: **blocking recall**. Only ~92% of true India pairs reach the candidate set,
  because `DF_CAP = 0.003` in `src/block.py` drops mid-frequency tokens (city names, common
  name words) to save time and RAM. At `0.01`, recall was ~94.5% but ~3× slower. That was too
  slow for the laptop but should be fine on a big machine.

## What to do, in order

Work in this repo. Keep the laptop's baseline untouched; this is a separate, improved run.
Log every step's duration and key numbers. You'll send them back at the end.

### Step 0: setup (≈10 min)

```bash
pip install -r requirements.txt xgboost
unzip dataset.zip                         # -> student_resource/
nvidia-smi                                # confirm the GPU is visible
python -c "import xgboost; print(xgboost.__version__)"
python src/convert.py --data student_resource/dataset --work work
python src/prep.py --work work
```

Note the machine's RAM and CPU cores. With **≥ 32 GB RAM**, the steps below may run in
parallel where noted. With less, run everything strictly one after another. Running two heavy
steps at once on 8 GB crashed the laptop.

### Step 1: pick the blocking configuration (≈15–20 min), optional but high value

Measure blocking recall on 5% of India queries. Try each configuration below and record the
printed `recall@10` and time:

```bash
# A: current setting (baseline)
python src/eval_block.py    --work work --frac 0.05 --country India
# B: higher DF cap: in src/block.py set every word family in DF_CAP to 0.01 (keep "c": 0.002)
python src/eval_block.py    --work work --frac 0.05 --country India
# C: B + experimental cross-token channel: apply the same DF_CAP edit in src/block_v2.py
python src/eval_block_v2.py --work work --frac 0.05 --country India
```

Choose the configuration with the **highest recall whose full test build fits your time
budget**. The test build is ~10M queries; scale the 5%-India timing by roughly ×50 to estimate.
If you pick C, change `from block import topk_candidates` to `from block_v2 import
topk_candidates` in `src/pipeline.py`. Train and test must be built with the **same**
configuration.

If short on time, skip Step 1 and use configuration A (identical to the baseline blocking).

### Step 2: build features (the long step)

```bash
python src/pipeline.py build --work work --split train      # optionally --frac 0.5 for more training data
python src/pipeline.py build --work work --split test
```

* With ≥ 32 GB RAM these two may run in parallel (two terminals).
* If a build crashes mid-country, delete that country's partial `work/feat_<split>/<Country>_*.parquet`
  files and rerun. Completed countries are skipped.
* Don't change `block.py`, `features.py` or `normalize.py` between the train and test builds.
* **If you build train with `--frac 0.5`, pass the same `--frac 0.5` to every `train` command
  below.** The CV score is computed on exactly the sampled queries; a mismatch gives a wrong score.

### Step 3: train on the GPU (≈10–30 min)

```bash
python src/pipeline.py train --work work --model xgb --device cuda --sample 1.0 --rounds 800
```

It prints macro F0.5 for thresholds 0.20–0.95 and the best one (saved in `work/model_meta.json`).

* If the best CV macro F0.5 is **below 0.942**, also try LightGBM as a check:
  `python src/pipeline.py train --work work --sample 0.5` (CPU). Keep whichever scores higher.
  The last `train` run is the one `predict` uses, so rerun the winner last.
* If `--device cuda` errors, the GPU driver or CUDA setup is the problem. Fall back to
  `--device cpu` rather than losing time.

### Step 4: predict and validate (≈15 min)

```bash
python src/pipeline.py predict --work work --out output
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

The validator must print **PASS**.

Sanity checks before sending:
* 1,732,544 rows in each file.
* Share of S1 rows with a non-empty match list should be roughly 90–95%. The baseline had 93.3%.
* Look at ~20 France rows by joining with `test_source1/2/3.tsv`, and check the matches look right.

### Step 5: send back

1. `output/matching_results.tsv` (this is what gets uploaded; zip it if it's too large to send).
2. `output/candidate_pairs.tsv` (needed for the final submission zip).
3. A short note with:
   * the blocking configuration used (A/B/C) and its India recall@10
   * the model type, `--sample` and `--rounds`, and the **best CV macro F0.5 and threshold**
   * the S1 match rate from the sanity check
   * the machine's CPU, RAM and GPU, and how long each step took
4. Push code changes to a **new branch** (not `main`), e.g. `git checkout -b gpu-run && git commit -am "gpu run" && git push -u origin gpu-run`,
   so the exact code behind the submitted files is on record.

## Don'ts

* Don't use any external data, web APIs or pretrained models that weren't checked for licence.
* Don't hard-code, filter or one-hot on country values.
* Don't tune the threshold by hand on test output. Only use the CV search printed by `train`.
* Don't hand over output that failed the validator.
