"""Step 0: convert the raw TSV files to Parquet (streaming, low memory).

Usage: python src/convert.py --data <student_resource/dataset> --work <work dir>
"""
import argparse
import os

import polars as pl

FILES = [
    ("train", "train_source1"), ("train", "train_source2"), ("train", "train_source3"),
    ("train", "train_ground_truth"),
    ("test", "test_source1"), ("test", "test_source2"), ("test", "test_source3"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    for split, name in FILES:
        src = os.path.join(args.data, split, name + ".tsv")
        dst = os.path.join(args.work, name + ".parquet")
        if os.path.exists(dst):
            continue
        # quote_char=None: fields are raw text, a stray '"' must not swallow rows
        lf = pl.scan_csv(src, separator="\t", quote_char=None, infer_schema=False,
                         missing_utf8_is_empty_string=True)
        lf.sink_parquet(dst)
        n = pl.scan_parquet(dst).select(pl.len()).collect().item()
        print(f"{name}: {n} rows -> {dst}", flush=True)


if __name__ == "__main__":
    main()
