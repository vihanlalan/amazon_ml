"""Blocking diagnostics on train: recall of the true S1 within each query's top-k."""
import argparse
import os
import time

import polars as pl

from block import topk_candidates
from metric import truth_pairs

COLS = ["entity_id", "country", "core_n", "skel", "addr_n", "business_address"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_df", type=float, default=1.0)
    ap.add_argument("--frac", type=float, default=0.1, help="fraction of queries to test")
    ap.add_argument("--country", default=None)
    args = ap.parse_args()
    W = args.work
    s1 = pl.read_parquet(os.path.join(W, "norm_train_source1.parquet"), columns=COLS)
    q = pl.concat([pl.read_parquet(os.path.join(W, f"norm_train_source{s}.parquet"), columns=COLS)
                   for s in (2, 3)]).sample(fraction=args.frac, seed=0)
    if args.country:
        s1 = s1.filter(pl.col("country") == args.country)
        q = q.filter(pl.col("country") == args.country)
    t = time.time()
    cand = topk_candidates(s1, q, max_df_frac=args.max_df)
    print(f"blocking {q.height} queries: {time.time() - t:.0f}s, {cand.height} pairs")
    tp = truth_pairs(pl.read_parquet(os.path.join(W, "train_ground_truth.parquet")))
    tp = tp.filter(pl.col("q").is_in(q["entity_id"].implode()))
    j = tp.join(cand, on=["s1", "q"], how="left")
    print("true pairs:", tp.height)
    for k in (1, 2, 3, 5, 10, 20):
        if k <= args.k:
            print(f"  recall@{k}: {(j['rank'] < k).sum() / tp.height:.4f}")
    j = j.with_columns(pl.col("q").str.slice(0, 2).alias("src"))
    print(j.join(q.select(pl.col("entity_id").alias("q"), "country"), on="q")
          .group_by("src", "country").agg((pl.col("rank").is_not_null()).mean().alias("recall")))
    j.filter(pl.col("rank").is_null()).write_parquet(os.path.join(W, "block_misses.parquet"))


if __name__ == "__main__":
    main()
