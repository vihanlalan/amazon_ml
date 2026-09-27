"""Step 3: pairwise features for (Source 1, Source 2/3) candidate pairs.

Feature groups
  * name:    rapidfuzz similarities on normalised / core / skeleton names, token overlap,
             extra/missing tokens (distractors often add a word), legal-form agreement
  * address: fuzzy similarities on full address and on its alphabetic part, house-number
             agreement (distractors copy the address with a slightly shifted number)
  * context: blocking score, rank, gap to the query's best S1, candidate counts, source
Country is never used as a categorical feature, so unseen countries (France) are handled
by the same country-agnostic signals.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

NCOLS = ["entity_id", "name_n", "core_n", "skel", "addr_n", "nums", "legal"]


def context_features(cand: pl.DataFrame) -> pl.DataFrame:
    """Features that depend on the whole candidate set (computed once, before chunking)."""
    # only query-side context: every query sees the full S1 index in train and test alike
    # (S1-side counts would shift because train queries are subsampled)
    ex = []
    for c in ("s_full", "s_name", "s_addr", "s_pair"):
        ex += [pl.col(c).max().over("q").alias(f"{c}_qbest"),
               pl.when(pl.len().over("q") > 1).then(pl.col(c).top_k(2).min().over("q"))
                 .otherwise(0.0).alias(f"{c}_qsecond")]
    return cand.with_columns(pl.len().over("q").alias("q_ncand"), *ex).with_columns(
        *[(pl.col(f"{c}_qbest") - pl.col(c)).alias(f"{c}_gap") for c in ("s_full", "s_name", "s_addr", "s_pair")],
        *[(pl.col(c) - pl.col(f"{c}_qsecond")).alias(f"{c}_margin") for c in ("s_full", "s_name", "s_addr", "s_pair")],
    )


def _sim(a, b, scorer) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def pair_features(pairs: pl.DataFrame, s1n: pl.DataFrame, qn: pl.DataFrame) -> pl.DataFrame:
    """pairs: output of context_features (chunk). s1n/qn: normalised frames with NCOLS."""
    d = (pairs.join(s1n.select(NCOLS), left_on="s1", right_on="entity_id", how="left")
              .join(qn.select(NCOLS), left_on="q", right_on="entity_id", how="left", suffix="_b"))
    L = lambda c: d[c].fill_null("").to_list()
    na, nb = L("name_n"), L("name_n_b")
    ca, cb = L("core_n"), L("core_n_b")
    sa, sb = L("skel"), L("skel_b")
    aa, ab = L("addr_n"), L("addr_n_b")
    alpha = lambda xs: [" ".join(t for t in x.split() if not any(ch.isdigit() for ch in t)) for x in xs]
    aaa, aab = alpha(aa), alpha(ab)
    nospace = lambda xs: [x.replace(" ", "") for x in xs]
    f = {
        "name_ratio": _sim(na, nb, fuzz.ratio),
        "name_tset": _sim(na, nb, fuzz.token_set_ratio),
        "core_tsort": _sim(ca, cb, fuzz.token_sort_ratio),
        "core_tset": _sim(ca, cb, fuzz.token_set_ratio),
        "core_partial": _sim(ca, cb, fuzz.partial_ratio),
        "core_jw": _sim(ca, cb, JaroWinkler.normalized_similarity),
        "core_nospace": _sim(nospace(ca), nospace(cb), fuzz.ratio),
        "core_nospace_partial": _sim(nospace(ca), nospace(cb), fuzz.partial_ratio),
        "skel_ratio": _sim(sa, sb, fuzz.ratio),
        "skel_tset": _sim(sa, sb, fuzz.token_set_ratio),
        "addr_ratio": _sim(aa, ab, fuzz.ratio),
        "addr_tset": _sim(aa, ab, fuzz.token_set_ratio),
        "addr_tsort": _sim(aa, ab, fuzz.token_sort_ratio),
        "addr_alpha_tset": _sim(aaa, aab, fuzz.token_set_ratio),
        "addr_alpha_partial": _sim(aaa, aab, fuzz.partial_ratio),
    }
    out = d.select(
        "s1", "q", "rank", "q_ncand", pl.col("^s_(full|name|addr|pair).*$"),
        pl.col("q").str.starts_with("S2").cast(pl.Int8).alias("is_s2"),
        # token set arithmetic on names
        _tok_feats("core_n", "core_n_b", "core"),
        _tok_feats("name_n", "name_n_b", "name"),
        # legal forms
        (pl.col("legal").list.len() > 0).cast(pl.Int8).alias("legal_a"),
        (pl.col("legal_b").list.len() > 0).cast(pl.Int8).alias("legal_b_has"),
        pl.col("legal").list.set_intersection("legal_b").list.len().alias("legal_inter"),
        pl.col("legal").list.set_symmetric_difference("legal_b").list.len().alias("legal_symdiff"),
        # address numbers
        pl.col("nums").list.len().alias("nums_a"),
        pl.col("nums_b").list.len().alias("nums_b_n"),
        pl.col("nums").list.set_intersection("nums_b").list.len().alias("nums_inter"),
        pl.col("nums").list.set_difference("nums_b").list.len().alias("nums_a_only"),
        pl.col("nums_b").list.set_difference("nums").list.len().alias("nums_b_only"),
        (pl.col("nums").list.first() == pl.col("nums_b").list.first()).cast(pl.Int8).fill_null(-1).alias("num1_eq"),
        _num_max("nums").alias("_ma"), _num_max("nums_b").alias("_mb"),
        pl.col("nums").list.first().count().over(["q", "nums"]).cast(pl.Int8).alias("num_consensus"),
        (pl.col("addr_n").fill_null("").str.len_chars() == 0).cast(pl.Int8).alias("addr_a_empty"),

        (pl.col("addr_n_b").fill_null("").str.len_chars() == 0).cast(pl.Int8).alias("addr_b_empty"),
        pl.col("core_n").fill_null("").str.len_chars().alias("core_len_a"),
        pl.col("core_n_b").fill_null("").str.len_chars().alias("core_len_b"),
    ).unnest("core", "name")
    out = out.with_columns(
        (pl.col("_ma") == pl.col("_mb")).cast(pl.Int8).fill_null(-1).alias("nummax_eq"),
        ((pl.col("_ma") - pl.col("_mb")).abs() / pl.max_horizontal("_ma", "_mb", pl.lit(1.0)))
        .fill_null(-1).alias("nummax_reldiff"),
        (pl.col("_ma") - pl.col("_mb")).abs().fill_null(-1).alias("nummax_absdiff"),
    ).drop("_ma", "_mb")
    return out.with_columns(**{k: pl.Series(v) for k, v in f.items()})


def _num_max(c: str) -> pl.Expr:
    return (pl.col(c).list.eval(pl.element().str.extract(r"^(\d{1,9})").cast(pl.Float64, strict=False))
            .list.max())


def _tok_feats(a: str, b: str, p: str) -> pl.Expr:
    ta = pl.col(a).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tb = pl.col(b).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    inter = ta.list.set_intersection(tb).list.len()
    union = ta.list.set_union(tb).list.len()
    return pl.struct(
        inter.alias(f"{p}_inter"),
        (inter / pl.max_horizontal(union, pl.lit(1))).alias(f"{p}_jacc"),
        ta.list.set_difference(tb).list.len().alias(f"{p}_a_only"),
        tb.list.set_difference(ta).list.len().alias(f"{p}_b_only"),
    ).alias(p)


FEATURES = None  # filled lazily: all numeric columns except ids / label


def feature_names(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in ("s1", "q", "label")]
