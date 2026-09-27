"""Add generator-aware features to existing feature files (no blocking rebuild).

House numbers: true records corrupt the number in characteristic ways (dropped digit
15201->5201, leading zeros, changed first digit 8759->1759), while distractors are copies of an
S1 with a small additive shift of the same length (1543->1547, 22644->22649). Name: true variants
tend to add legal forms, distractors add business words. `_shn` / `_qhn` (S1 / query main house
number) are kept as hidden columns so stack.py can find groups of candidates sharing a number.

  python src/augment.py --work work --split train     # writes work/feat_train_aug/
  python src/augment.py --work work --split test      # writes work/feat_test_aug/
Then run stack.py with --aug.
"""
import argparse
import glob
import os
import time

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from normalize import FILLER, LEGAL

LEGAL_TOK = sorted(set(LEGAL.values()))
FILLER_TOK = sorted(FILLER)


def _longest(e: pl.Expr) -> pl.Expr:
    """Longest string of a list column (first on ties); null for empty lists."""
    return e.list.get(e.list.eval(pl.element().str.len_chars()).list.arg_max(), null_on_oob=True)


def lookup(W, split, sources):
    """entity_id -> main house number (longest digit string) + name tokens."""
    lf = pl.concat([pl.scan_parquet(os.path.join(W, f"norm_{split}_source{s}.parquet")) for s in sources])
    return lf.select(
        "entity_id",
        _longest(pl.col("nums").list.eval(
            pl.element().str.replace_all(r"\D", "").str.replace(r"^0+", ""))).fill_null("").alias("hn"),
        pl.col("name_n").fill_null("").str.split(" ").alias("nt"),
    ).collect()


def add_features(ch: pl.DataFrame, a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    d = ch.join(a.rename({"entity_id": "s1", "hn": "_shn", "nt": "_snt"}), on="s1", how="left") \
          .join(b.rename({"entity_id": "q", "hn": "_qhn", "nt": "_qnt"}), on="q", how="left")
    d = d.with_columns(pl.col("_shn").fill_null(""), pl.col("_qhn").fill_null(""))
    sa, sb = d["_shn"].to_list(), d["_qhn"].to_list()
    lev = cpdist(sa, sb, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    both = (pl.col("_shn") != "") & (pl.col("_qhn") != "")
    la, lb = pl.col("_shn").str.len_chars().cast(pl.Int32), pl.col("_qhn").str.len_chars().cast(pl.Int32)
    na = pl.col("_shn").str.slice(0, 15).cast(pl.Float64, strict=False)
    nb = pl.col("_qhn").str.slice(0, 15).cast(pl.Float64, strict=False)
    absdiff = (na - nb).abs()
    b_only = pl.col("_qnt").list.set_difference("_snt")
    a_only = pl.col("_snt").list.set_difference("_qnt")
    d = d.with_columns(
        pl.Series("_lev", lev),
        both.cast(pl.Int8).alias("hn_both"),
        la.cast(pl.Int16).alias("hn_a_len"),
        lb.cast(pl.Int16).alias("hn_b_len"),
    ).with_columns(
        pl.when(both).then(pl.col("_shn") == pl.col("_qhn")).cast(pl.Int8).fill_null(-1).alias("hn_eq"),
        pl.when(both).then(la - lb).fill_null(-99).cast(pl.Int16).alias("hn_len_diff"),
        pl.when(both).then(pl.col("_lev")).fill_null(-1).cast(pl.Int16).alias("hn_lev"),
        # one number contained in the other (digit dropped) but not equal
        pl.when(both).then((pl.col("_shn") != pl.col("_qhn")) & (
            pl.col("_shn").str.contains(pl.col("_qhn"), literal=True)
            | pl.col("_qhn").str.contains(pl.col("_shn"), literal=True))).cast(pl.Int8).fill_null(-1).alias("hn_sub"),
        pl.when(both).then(pl.col("_shn").str.slice(0, 1) == pl.col("_qhn").str.slice(0, 1)).cast(pl.Int8).fill_null(-1).alias("hn_first_eq"),
        pl.when(both).then(pl.col("_shn").str.slice(-1, 1) == pl.col("_qhn").str.slice(-1, 1)).cast(pl.Int8).fill_null(-1).alias("hn_last_eq"),
        pl.when(both).then(absdiff).fill_null(-1).alias("hn_absdiff"),
        # the distractor signature: same length, small positive/negative shift
        pl.when(both).then((la == lb) & (absdiff >= 1) & (absdiff <= 12)).cast(pl.Int8).fill_null(-1).alias("hn_shift_small"),
        b_only.list.eval(pl.element().is_in(LEGAL_TOK)).list.sum().fill_null(0).cast(pl.Int16).alias("nb_extra_legal"),
        b_only.list.eval(pl.element().is_in(FILLER_TOK)).list.sum().fill_null(0).cast(pl.Int16).alias("nb_extra_filler"),
        b_only.list.eval(~pl.element().is_in(LEGAL_TOK + FILLER_TOK) & (pl.element() != "")).list.sum().fill_null(0).cast(pl.Int16).alias("nb_extra_other"),
        a_only.list.eval(~pl.element().is_in(LEGAL_TOK + FILLER_TOK) & (pl.element() != "")).list.sum().fill_null(0).cast(pl.Int16).alias("na_extra_other"),
    )
    return d.drop("_snt", "_qnt", "_lev")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    a = ap.parse_args()
    W, t0 = a.work, time.time()
    out = os.path.join(W, f"feat_{a.split}_aug")
    os.makedirs(out, exist_ok=True)
    s1 = lookup(W, a.split, (1,))
    q = lookup(W, a.split, (2, 3))
    print(f"lookups ready ({time.time() - t0:.0f}s)", flush=True)
    files = sorted(glob.glob(os.path.join(W, f"feat_{a.split}", "*.parquet")))
    for i, f in enumerate(files):
        dst = os.path.join(out, os.path.basename(f))
        if os.path.exists(dst):
            continue
        ch = pl.read_parquet(f)
        ids1, idsq = ch["s1"].unique().implode(), ch["q"].unique().implode()
        add_features(ch, s1.filter(pl.col("entity_id").is_in(ids1)),
                     q.filter(pl.col("entity_id").is_in(idsq))).write_parquet(dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        print(f"  {i + 1}/{len(files)} {os.path.basename(f)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"augment {a.split} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
