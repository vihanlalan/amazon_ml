"""Competition metric: F_0.5 per Source 1 entity, macro-averaged over all S1 entities.
Singletons score 1.0 for an empty prediction and 0.0 for any prediction."""
import polars as pl


def macro_f05(truth: pl.DataFrame, pred: pl.DataFrame) -> float:
    """truth / pred: (s1, q) pair frames. `s1_all` of truth defines the evaluated entities
    when passed as column 's1' with q == None for singletons."""
    t = truth.filter(pl.col("q").is_not_null())
    all_s1 = truth.select("s1").unique()
    tp = t.join(pred, on=["s1", "q"], how="inner").group_by("s1").len().rename({"len": "tp"})
    nt = t.group_by("s1").len().rename({"len": "nt"})
    npred = pred.group_by("s1").len().rename({"len": "np"})
    d = (all_s1.join(nt, on="s1", how="left").join(npred, on="s1", how="left")
         .join(tp, on="s1", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0)
    d = d.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
        .when((pl.col("nt") == 0) | (pl.col("np") == 0) | (pl.col("tp") == 0)).then(0.0)
        .otherwise(f).alias("f"))
    return d["f"].mean()


def truth_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Ground-truth file -> (s1, q) with q null for singletons."""
    return (gt.select(pl.col("source1_entity_id").alias("s1"),
                      pl.col("matched_entity_ids").str.split(",").alias("q"))
            .explode("q")
            .with_columns(pl.when(pl.col("q") == "").then(None).otherwise(pl.col("q")).alias("q")))
