#!/usr/bin/env bash
# End-to-end run: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# Usage: bash run_all.sh <path to student_resource> [work dir] [output dir]
set -euo pipefail

SR="${1:?path to student_resource (the folder containing dataset/ and utils/)}"
WORK="${2:-work}"
OUT="${3:-output}"
SRC="$(cd "$(dirname "$0")" && pwd)/src"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

python -u "$SRC/convert.py"  --data "$SR/dataset" --work "$WORK"
python -u "$SRC/prep.py"     --work "$WORK"
python -u "$SRC/pipeline.py" build --work "$WORK" --split train
python -u "$SRC/pipeline.py" build --work "$WORK" --split test
# full 2-fold OOF training + threshold search (prints CV macro F0.5).
# To skip the OOF pass and reuse the known threshold: add  --final_thr 0.35
python -u "$SRC/pipeline.py" train --work "$WORK" --sample 0.25
python -u "$SRC/pipeline.py" predict --work "$WORK" --out "$OUT"

python "$SR/utils/validate_submission.py" \
    --matching "$OUT/matching_results.tsv" \
    --candidate "$OUT/candidate_pairs.tsv" \
    --test-dir "$SR/dataset/test"
