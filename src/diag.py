"""Quick diagnostics of a finished run: train build fraction, blocking recall ceiling per
country (share of true pairs that reached the candidate set), candidates per query, and the
stack CV scores.   python src/diag.py --work work
"""
import argparse
import glob
import json
import os

import polars as pl

from metric import truth_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    W = ap.parse_args().work
    feat = pl.scan_parquet(os.path.join(W, "feat_train", "*.parquet")).select("s1", "q", "label").collect()
    q = pl.concat([pl.scan_parquet(os.path.join(W, f"norm_train_source{s}.parquet")).select("entity_id", "country").collect()
                   for s in (2, 3)])
    frac = round(feat["q"].n_unique() / q.height, 2)
    t = truth_pairs(pl.read_parquet(os.path.join(W, "train_ground_truth.parquet"))).filter(pl.col("q").is_not_null())
    t = t.filter((pl.col("q").hash(seed=3) % 1000) < int(frac * 1000)).join(q, left_on="q", right_on="entity_id")
    hit = t.join(feat.filter(pl.col("label") == 1).select("s1", "q"), on=["s1", "q"], how="semi")
    print(f"train build frac: {frac}; candidates per query: {feat.height / feat['q'].n_unique():.2f}")
    r = t.group_by("country").len().join(hit.group_by("country").len(), on="country", suffix="_hit")
    for row in r.sort("country").iter_rows(named=True):
        print(f"blocking recall {row['country']}: {row['len_hit'] / row['len']:.4f}")
    print(f"blocking recall overall: {hit.height / t.height:.4f}")
    for f in sorted(glob.glob(os.path.join(W, "stack_meta*.json"))):
        print(os.path.basename(f), json.load(open(f)))


if __name__ == "__main__":
    main()
