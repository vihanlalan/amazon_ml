"""Stage 2 (stacking): S1-group consistency features on top of stage-1 probabilities.

Why: stage 1 scores every (S1, query) pair in isolation. Distractors are near-copies of an S1
with a shifted house number, while the S1's real S2/S3 records mostly agree with it. Looking at
*all* candidates of an S1 together (how many confident candidates it has, whether they agree
on the house number, how many queries picked this S1 as their best, ...) separates them.

Leakage / shift control:
  * All folds split by S1 (hash), so the stage-1 probabilities feeding an S1's group features
    come from a model that never saw that S1's labels.
  * Train queries are a `frac` sample, so an S1 sees ~frac of its candidates in train. On test,
    queries are split into round(1/frac) random groups and group features are computed within
    each group, so the features mean the same thing in train and test.

Runs only on existing feature files (no rebuild):
  python src/stack.py --work work --frac 0.2 --out output [--model xgb --device cuda] [--sample 0.5]
--frac must equal the --frac used for `pipeline.py build --split train`.
Prints stage-1 and stage-2 CV (macro F0.5); writes the outputs of whichever is better.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import polars as pl

from features import feature_names
from metric import macro_f05
from pipeline import PARAMS, XGB_PARAMS, eval_truth, fit_model, predict_model, write_outputs

SRC = ["s1", "q", "nummax_eq"]
HN = ["_shn", "_qhn"]            # hidden main house numbers (augment.py), used for group features
HN_CTX = ["hn_cluster_oth", "hn_cluster_conf_oth", "q_is_s1hn", "s1_conf_s1hn", "hn_rival"]
SUFFIX = ""                      # "_aug" with --aug
CTX = ["p1", "p1_qmax", "p1_qsecond", "p1_gap", "p1_margin", "p1_qrank", "q_nconf",
       "s1_n", "s1_nconf", "s1_psum", "s1_pmax", "p1_s1rank", "s1_conf_numeq", "s1_conf_numneq",
       "s1_nbestof", "s1_oth_conf", "s1_oth_psum", "s1_oth_conf_numeq", "s1_oth_bestof",
       "num_conflict"]
THRS = np.arange(0.2, 0.96, 0.05)


def s1fold():
    return (pl.col("s1").hash(seed=7) % 2).cast(pl.Int8)


def in_sample(sample):
    return (pl.col("q").hash(seed=11) % 1000) < int(sample * 1000) if sample < 1 else pl.lit(True)


def by_country(W, split):
    out = {}
    for f in sorted(glob.glob(os.path.join(W, f"feat_{split}{SUFFIX}", "*.parquet"))):
        out.setdefault(os.path.basename(f).split("_")[0], []).append(f)
    return out


def load_xy(W, feats, filt, ctx_dir=None):
    """Pre-sized float32 matrix over all train files (optionally joined with group features)."""
    groups = by_country(W, "train")
    n = sum(pl.scan_parquet(f).filter(filt).select(pl.len()).collect().item()
            for fs in groups.values() for f in fs)
    X = np.empty((n, len(feats)), np.float32)
    y = np.empty(n, np.int8)
    o = 0
    for c, fs in groups.items():
        cx = pl.read_parquet(os.path.join(ctx_dir, f"{c}.parquet")) if ctx_dir else None
        for f in fs:
            ch = pl.scan_parquet(f).filter(filt).collect()
            if cx is not None:
                ch = ch.join(cx, on=["s1", "q"], how="left")
            k = ch.height
            X[o:o + k] = ch.select(feats).to_numpy().astype(np.float32)
            y[o:o + k] = ch["label"].to_numpy()
            o += k
    return X, y


def group_features(d: pl.DataFrame) -> pl.DataFrame:
    """d: s1, q, g, p1, nummax_eq -> s1, q + CTX. S1 statistics are computed within (s1, g)."""
    conf = pl.col("p1") > 0.5
    numeq = pl.col("nummax_eq") == 1
    G = ["s1", "g"]
    d = d.with_columns(
        pl.col("p1").max().over("q").alias("p1_qmax"),
        pl.when(pl.len().over("q") > 1).then(pl.col("p1").top_k(2).min().over("q"))
          .otherwise(0.0).alias("p1_qsecond"),
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Int32).alias("p1_qrank"),
        conf.cast(pl.Int32).sum().over("q").alias("q_nconf"),
        pl.len().over(G).cast(pl.Int32).alias("s1_n"),
        conf.cast(pl.Int32).sum().over(G).alias("s1_nconf"),
        pl.col("p1").sum().over(G).alias("s1_psum"),
        pl.col("p1").max().over(G).alias("s1_pmax"),
        pl.col("p1").rank("ordinal", descending=True).over(G).cast(pl.Int32).alias("p1_s1rank"),
        (conf & numeq).cast(pl.Int32).sum().over(G).alias("s1_conf_numeq"),
        (conf & (pl.col("nummax_eq") == 0)).cast(pl.Int32).sum().over(G).alias("s1_conf_numneq"),
    )
    d = d.with_columns((pl.col("p1") >= pl.col("p1_qmax")).cast(pl.Int32).alias("_best"))
    d = d.with_columns(pl.col("_best").sum().over(G).alias("s1_nbestof"))
    d = d.with_columns(
        (pl.col("p1_qmax") - pl.col("p1")).alias("p1_gap"),
        (pl.col("p1") - pl.col("p1_qsecond")).alias("p1_margin"),
        (pl.col("s1_nconf") - conf.cast(pl.Int32)).alias("s1_oth_conf"),
        (pl.col("s1_psum") - pl.col("p1")).alias("s1_oth_psum"),
        (pl.col("s1_conf_numeq") - (conf & numeq).cast(pl.Int32)).alias("s1_oth_conf_numeq"),
        (pl.col("s1_nbestof") - pl.col("_best")).alias("s1_oth_bestof"),
    )
    # a candidate whose house number disagrees while other confident candidates agree with S1
    d = d.with_columns(((pl.col("nummax_eq") == 0) & (pl.col("s1_oth_conf_numeq") > 0))
                       .cast(pl.Int8).alias("num_conflict"))
    extra = []
    if "_qhn" in d.columns:
        has = pl.col("_qhn") != ""
        C = ["s1", "g", "_qhn"]
        d = d.with_columns(
            pl.when(has).then(pl.len().over(C) - 1).otherwise(0).cast(pl.Int32).alias("hn_cluster_oth"),
            pl.when(has).then(conf.cast(pl.Int32).sum().over(C) - conf.cast(pl.Int32)).otherwise(0)
              .cast(pl.Int32).alias("hn_cluster_conf_oth"),
            (has & (pl.col("_qhn") == pl.col("_shn"))).cast(pl.Int8).alias("q_is_s1hn"),
        )
        d = d.with_columns(
            (conf & (pl.col("q_is_s1hn") == 1)).cast(pl.Int32).sum().over(G).alias("s1_conf_s1hn"),
            # this record's number differs from S1's but is shared by other candidates: a decoy group
            (has & (pl.col("q_is_s1hn") == 0) & (pl.col("hn_cluster_oth") > 0)).cast(pl.Int8).alias("hn_rival"),
        )
        extra = HN_CTX
    return d.select("s1", "q", *CTX, *extra)


def sweep(truth, pred, name):
    best_q = pred.sort("p", descending=True).unique("q", keep="first")  # decision rule, sorted once
    res = {round(float(t), 2): macro_f05(truth, best_q.filter(pl.col("p") >= t).select("s1", "q"))
           for t in THRS}
    best = max(res, key=res.get)
    print(f"{name}: " + ", ".join(f"{k}:{v:.5f}" for k, v in res.items()), flush=True)
    print(f"{name} best thr {best}: {res[best]:.5f}", flush=True)
    return best, res[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--frac", type=float, default=None,
                    help="same --frac as the train build (auto-detected from the feature files if omitted)")
    ap.add_argument("--sample", type=float, default=0.5, help="share of train queries used to fit")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out", default="output")
    ap.add_argument("--aug", action="store_true", help="use feat_*_aug from augment.py")
    a = ap.parse_args()
    if a.lr:
        PARAMS["learning_rate"] = a.lr
        XGB_PARAMS["learning_rate"] = a.lr
    global SUFFIX
    SUFFIX = "_aug" if a.aug else ""
    W, t0 = a.work, time.time()
    if a.frac is None:  # share of train queries present in the train build
        nq = pl.scan_parquet(os.path.join(W, "feat_train", "*.parquet")).select(pl.col("q").n_unique()).collect().item()
        tot = sum(pl.scan_parquet(os.path.join(W, f"norm_train_source{s}.parquet")).select(pl.len()).collect().item()
                  for s in (2, 3))
        a.frac = round(nq / tot, 2)
        print(f"auto-detected --frac {a.frac} ({nq} of {tot} train queries in the build)", flush=True)
    meta = {"model": a.model, "device": a.device}
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
    tr_files = by_country(W, "train")
    te_files = by_country(W, "test")
    head = pl.scan_parquet(next(iter(tr_files.values()))[0]).head(1).collect()
    feats = [c for c in feature_names(head) if not c.startswith("_")]
    src = SRC + [c for c in HN if c in head.columns]
    tag = SUFFIX
    for d in ("p1_train", "p1_test", "ctx_train", "ctx_test", "pred2_test", "pred1_test"):
        os.makedirs(os.path.join(W, d + tag), exist_ok=True)

    # ---- stage 1: S1-grouped 2-fold OOF on train, final model on test -------------------------
    models = []
    for fold in (0, 1):
        X, y = load_xy(W, feats, (s1fold() != fold) & in_sample(a.sample))
        log(f"stage1 fold {fold}: {len(y)} rows")
        models.append(fit_model(X, y, a.rounds, a.model, a.device))
        del X, y
    for c, fs in tr_files.items():
        for f in fs:
            ch = pl.read_parquet(f)
            X = ch.select(feats).to_numpy().astype(np.float32)
            fo = ch.select(s1fold()).to_series().to_numpy()
            p = np.zeros(ch.height, np.float32)
            for k in (0, 1):
                m = fo == k
                if m.any():
                    p[m] = predict_model(models[k], meta, X[m])
            ch.select(*src, "label").with_columns(pl.Series("p1", p)).write_parquet(
                os.path.join(W, "p1_train" + tag, os.path.basename(f)))
    del models
    X, y = load_xy(W, feats, in_sample(a.sample))
    m1 = fit_model(X, y, a.rounds, a.model, a.device)
    del X, y
    for c, fs in te_files.items():
        for f in fs:
            ch = pl.read_parquet(f)
            p = predict_model(m1, meta, ch.select(feats).to_numpy().astype(np.float32))
            ch.select(*src).with_columns(pl.Series("p1", p, dtype=pl.Float32)).write_parquet(
                os.path.join(W, "p1_test" + tag, os.path.basename(f)))
    del m1
    log("stage1 done")

    truth = eval_truth(W, a.frac)
    oof1 = pl.scan_parquet(os.path.join(W, "p1_train" + tag, "*.parquet")).select("s1", "q", pl.col("p1").alias("p")).collect()
    thr1, cv1 = sweep(truth, oof1, "stage1 CV")
    del oof1

    # ---- group features (train: one group = the sample; test: round(1/frac) random groups) ----
    G = max(1, round(1 / a.frac))
    for split, files in (("train", tr_files), ("test", te_files)):
        for c in files:
            d = pl.scan_parquet(os.path.join(W, f"p1_{split}" + tag, f"{c}_*.parquet")).collect()
            g = pl.lit(0) if split == "train" else (pl.col("q").hash(seed=17) % G)
            group_features(d.with_columns(g.alias("g"))).write_parquet(os.path.join(W, f"ctx_{split}" + tag, f"{c}.parquet"))
    log(f"group features done (test groups: {G})")

    # ---- stage 2: same S1 folds ------------------------------------------------------------------
    ctx_cols = pl.read_parquet_schema(glob.glob(os.path.join(W, "ctx_train" + tag, "*.parquet"))[0])
    feats2 = feats + [c for c in ctx_cols if c not in ("s1", "q")]
    ctxd = os.path.join(W, "ctx_train" + tag)
    oof2 = []
    for fold in (0, 1):
        X, y = load_xy(W, feats2, (s1fold() != fold) & in_sample(a.sample), ctxd)
        m = fit_model(X, y, a.rounds, a.model, a.device)
        del X, y
        for c, fs in tr_files.items():
            cx = pl.read_parquet(os.path.join(ctxd, f"{c}.parquet"))
            for f in fs:
                ch = pl.scan_parquet(f).filter(s1fold() == fold).collect().join(cx, on=["s1", "q"], how="left")
                if ch.height:
                    oof2.append(ch.select("s1", "q").with_columns(
                        pl.Series("p", predict_model(m, meta, ch.select(feats2).to_numpy().astype(np.float32)))))
        del m
        log(f"stage2 fold {fold} done")
    thr2, cv2 = sweep(truth, pl.concat(oof2), "stage2 CV")
    del oof2

    # ---- final: write outputs of the better stage -------------------------------------------------
    if cv2 > cv1:
        X, y = load_xy(W, feats2, in_sample(a.sample), ctxd)
        m2 = fit_model(X, y, a.rounds, a.model, a.device)
        del X, y
        for c, fs in te_files.items():
            cx = pl.read_parquet(os.path.join(W, "ctx_test" + tag, f"{c}.parquet"))
            for f in fs:
                ch = pl.read_parquet(f).join(cx, on=["s1", "q"], how="left")
                ch.select("s1", "q").with_columns(pl.Series(
                    "p", predict_model(m2, meta, ch.select(feats2).to_numpy().astype(np.float32)), dtype=pl.Float32)
                ).write_parquet(os.path.join(W, "pred2_test" + tag, os.path.basename(f)))
        pred_dir, thr, used = "pred2_test" + tag, thr2, "stage2"
    else:
        for f in glob.glob(os.path.join(W, "p1_test" + tag, "*.parquet")):
            pl.read_parquet(f).select("s1", "q", pl.col("p1").alias("p")).write_parquet(
                os.path.join(W, "pred1_test" + tag, os.path.basename(f)))
        pred_dir, thr, used = "pred1_test" + tag, thr1, "stage1"
    write_outputs(W, pl.scan_parquet(os.path.join(W, pred_dir, "*.parquet")), thr, a.out)
    json.dump({"used": used, "cv_stage1": cv1, "thr_stage1": thr1, "cv_stage2": cv2, "thr_stage2": thr2,
               "frac": a.frac, "test_groups": G, "model": a.model},
              open(os.path.join(W, f"stack_meta{tag}.json"), "w"), indent=1)
    log(f"DONE: used {used} (stage1 CV {cv1:.5f}, stage2 CV {cv2:.5f}), outputs in {a.out}")


if __name__ == "__main__":
    main()
