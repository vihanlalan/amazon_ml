"""End-to-end pipeline.

  python src/pipeline.py build    --work W --split train|test   # blocking + pair features
  python src/pipeline.py train    --work W                      # 2-fold OOF LightGBM + threshold
  python src/pipeline.py predict  --work W --out OUT            # test predictions + TSVs

Decision rule: every S2/S3 record is assigned to at most one S1 entity (ground truth never
links one record to two S1 entities) - its highest-probability candidate, if that
probability clears a threshold tuned for macro F0.5 on out-of-fold train predictions.
"""
import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from block import topk_candidates
from features import NCOLS, context_features, feature_names, pair_features
from metric import macro_f05, truth_pairs

COLS = sorted(set(["entity_id", "country", "core_n", "skel", "addr_n", "business_address"]) | set(NCOLS))
TRAIN_QUERY_FRAC = 0.3   # train queries are subsampled (each still searches the full S1 index)


def _load(W, split, sources, country, frac=None):
    lf = pl.concat([pl.scan_parquet(os.path.join(W, f"norm_{split}_source{s}.parquet"))
                    for s in sources]).filter(pl.col("country") == country).select(COLS)
    if frac:
        lf = lf.filter((pl.col("entity_id").hash(seed=3) % 1000) < int(frac * 1000))
    return lf.collect()


def cmd_build(W, split, frac):
    """Blocking + context features + pair features, streamed chunk by chunk to feat_<split>/."""
    t = time.time()
    out_dir = os.path.join(W, f"feat_{split}")
    os.makedirs(out_dir, exist_ok=True)
    tp = None
    if split == "train":
        tp = truth_pairs(pl.read_parquet(os.path.join(W, "train_ground_truth.parquet")))
        tp = tp.filter(pl.col("q").is_not_null()).with_columns(pl.lit(1, pl.Int8).alias("label"))
    countries = (pl.scan_parquet(os.path.join(W, f"norm_{split}_source1.parquet"))
                 .select(pl.col("country").unique()).collect()["country"].to_list())
    for country in sorted(countries):
        if glob.glob(os.path.join(out_dir, f"{country}_*.parquet")):
            continue  # resume support
        s1 = _load(W, split, (1,), country)
        q = _load(W, split, (2, 3), country, frac if split == "train" else None)
        print(f"{country}: {s1.height} S1, {q.height} queries", flush=True)

        def post(cand, s1=s1, q=q):
            cand = context_features(cand)
            f = pair_features(cand, s1.filter(pl.col("entity_id").is_in(cand["s1"].unique().implode())),
                              q.filter(pl.col("entity_id").is_in(cand["q"].unique().implode())))
            if tp is not None:
                f = f.join(tp, on=["s1", "q"], how="left").with_columns(pl.col("label").fill_null(0))
            return f

        topk_candidates(s1, q, out_dir=out_dir, post=post)
        print(f"{country} done ({time.time() - t:.0f}s)", flush=True)
        del s1, q


PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=max(os.cpu_count() - 2, 1), verbose=-1)


def _fold(col="q"):
    return (pl.col(col).hash(seed=7) % 2).cast(pl.Int8)


def _load_feats(W, split, sample=None, fold=None):
    files = sorted(glob.glob(os.path.join(W, f"feat_{split}", "*.parquet")))
    lf = pl.scan_parquet(files)
    if fold is not None:
        lf = lf.filter(_fold() == fold)
    if sample:
        lf = lf.filter((pl.col("q").hash(seed=11) % 1000) < int(sample * 1000))
    return lf.collect()


def decide(pred: pl.DataFrame, thr: float) -> pl.DataFrame:
    """Assign each query to its best S1 if prob >= thr."""
    best = pred.sort("p", descending=True).unique("q", keep="first")
    return best.filter(pl.col("p") >= thr).select("s1", "q")


def eval_truth(W, frac):
    """Ground truth restricted to the sampled train queries. S1 entities whose true matches
    were all left out of the sample are dropped; true singletons stay."""
    t = truth_pairs(pl.read_parquet(os.path.join(W, "train_ground_truth.parquet")))
    return t.filter(pl.col("q").is_null()
                    | ((pl.col("q").hash(seed=3) % 1000) < int(frac * 1000)))


def cmd_train(W, sample, rounds, frac, final_thr=None):
    t = time.time()
    feats = None
    oof = []
    res = {}
    # final_thr given: skip the (slow) 2-fold OOF pass and reuse a known threshold
    for fold in (() if final_thr else (0, 1)):
        tr = _load_feats(W, "train", sample=sample, fold=1 - fold)
        cols = feature_names(tr)
        print(f"fold {fold}: train rows {tr.height}, pos {tr['label'].mean():.3f}", flush=True)
        ds = lgb.Dataset(tr.select(cols).to_numpy().astype(np.float32), tr["label"].to_numpy())
        del tr
        m = lgb.train(PARAMS, ds, num_boost_round=rounds)
        del ds
        for f in sorted(glob.glob(os.path.join(W, "feat_train", "*.parquet"))):
            ch = pl.read_parquet(f).filter(_fold() == fold)
            if ch.height == 0:
                continue
            p = ch.select("s1", "q", "label").with_columns(
                pl.Series("p", m.predict(ch.select(cols).to_numpy().astype(np.float32))))
            # a query's candidates all live in one chunk: keep only its best S1 (saves memory)
            oof.append(p.sort("p", descending=True).unique("q", keep="first"))
        feats = cols
        print(f"  fold {fold} done ({time.time() - t:.0f}s)", flush=True)
    if final_thr:
        best = final_thr
    else:
        oof = pl.concat(oof)
        oof.write_parquet(os.path.join(W, "oof_train.parquet"))
        truth = eval_truth(W, frac)
        for thr in np.arange(0.2, 0.96, 0.05):
            res[round(float(thr), 2)] = macro_f05(truth, decide(oof, thr))
            print(f"  thr {thr:.2f}: macro F0.5 {res[round(float(thr), 2)]:.5f}", flush=True)
        best = max(res, key=res.get)
        print(f"best thr {best}: {res[best]:.5f}")
        del oof
    # final model on both folds
    tr = _load_feats(W, "train", sample=sample)
    feats = feature_names(tr)
    m = lgb.train(PARAMS, lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["label"].to_numpy()),
                  num_boost_round=rounds)
    m.save_model(os.path.join(W, "model.txt"))
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(a, int(b)) for a, b in imp[:20]])
    json.dump({"thr": best, "features": feats, "cv": res}, open(os.path.join(W, "model_meta.json"), "w"), indent=1)


def cmd_predict(W, out):
    meta = json.load(open(os.path.join(W, "model_meta.json")))
    m = lgb.Booster(model_file=os.path.join(W, "model.txt"))
    preds = []
    for f in sorted(glob.glob(os.path.join(W, "feat_test", "*.parquet"))):
        ch = pl.read_parquet(f)
        preds.append(ch.select("s1", "q").with_columns(
            pl.Series("p", m.predict(ch.select(meta["features"]).to_numpy().astype(np.float32)))))
    pred = pl.concat(preds)
    pred.write_parquet(os.path.join(W, "pred_test.parquet"))
    write_outputs(W, pred, decide(pred, meta["thr"]), out)


def write_outputs(W, cand, match, out):
    """Write candidate_pairs.tsv and matching_results.tsv (one row per test S1 entity)."""
    os.makedirs(out, exist_ok=True)
    s1 = pl.read_parquet(os.path.join(W, "test_source1.parquet"), columns=["entity_id"])
    for df, col, name in ((cand, "candidate_entity_ids", "candidate_pairs.tsv"),
                          (match, "matched_entity_ids", "matching_results.tsv")):
        g = df.select("s1", "q").unique().sort("q").group_by("s1").agg(pl.col("q").str.join(",").alias(col))
        res = (s1.join(g, left_on="entity_id", right_on="s1", how="left")
               .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))
        res.write_csv(os.path.join(out, name), separator="\t", quote_style="never")
        print(f"wrote {name}: {res.height} rows, {(res[col] != '').sum()} non-empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "predict"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--frac", type=float, default=TRAIN_QUERY_FRAC)
    ap.add_argument("--sample", type=float, default=0.25)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--final_thr", type=float, default=None)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a.work, a.split, a.frac)
    elif a.cmd == "train":
        cmd_train(a.work, a.sample, a.rounds, a.frac, a.final_thr)
    else:
        cmd_predict(a.work, a.out)


if __name__ == "__main__":
    main()
