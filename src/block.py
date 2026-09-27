"""Step 2: candidate generation (blocking).

Every record is turned into sparse, L2-normalised TF-IDF vectors over hashed tokens:
    n|<core name word>   k|<consonant-skeleton word>   c|<name char 4-gram>
    a|<address word>     d|<compound address code, e.g. 't39', '1314', 'e1002'>
IDF comes from the Source 1 side (the index); tokens more common than `max_df_frac` of the
index are dropped (they cost a lot in the sparse product and discriminate little).

Three retrieval channels, each with its own top-k, are unioned:
    full  - all token families          (the main channel)
    name  - n, k, c only                (records with empty / truncated addresses)
    addr  - a, d only                   (records renamed at the same address)
Queries run in the *reverse* direction: each Source 2/3 record retrieves the most similar
Source 1 records of the same country (multi-threaded sparse top-n matmul). Every S2/S3
record belongs to at most one S1 entity, so this is both cheaper and more precise.
For all unioned pairs the cosine of every channel is computed exactly and kept as features.
"""
import os
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

DIM = 1 << 23
CHANNELS = {
    "full": {"n": 1.0, "k": 0.6, "a": 1.0, "d": 1.0},
    "name": {"n": 1.0, "k": 0.6, "c": 0.5},
    "addr": {"a": 1.0, "d": 1.0},
    "pair": {"x": 1.0},
}
TOPK = {"full": 5, "name": 3, "addr": 3, "pair": 8}
MAX_CAND = 10


def token_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Explode records into unique (row, token hash, family) triples."""
    raw_addr = pl.col("business_address").fill_null("").str.to_lowercase()
    base = df.select(
        pl.int_range(pl.len(), dtype=pl.Int32).alias("row"),
        pl.col("core_n").str.split(" ").alias("n"),
        pl.col("skel").str.split(" ").alias("k"),
        pl.col("addr_n").str.split(" ").alias("a"),
        # compound codes: letters/number groups joined across '-', '/', '#', '.'
        raw_addr.str.extract_all(r"[a-z]{0,2}[-#/ .]{0,3}\d[a-z0-9/#.-]*")
        .list.eval(pl.element().str.replace_all(r"[^a-z0-9]", "").str.replace(r"^0+", ""))
        .alias("d"),
        _ngrams(pl.col("core_n").str.replace_all(" ", ""), 4).alias("c"),
    )
    # pair tokens: the generator draws names/streets from a small vocabulary, so single words
    # are common (and dropped by the DF cap) while their *combinations* are distinctive.
    # name x name, name x address-word, address-word x address-word (order-free, typo-robust
    # via skeletons for names; numbers excluded because the noise corrupts them).
    kw = pl.col("k").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()
    aw = pl.col("a").list.eval(pl.element().filter(
        (pl.element().str.len_chars() >= 3) & ~pl.element().str.contains(r"\d"))).list.unique()
    # cap address words: long (Indian) addresses would otherwise explode quadratically
    pb = base.select("row", kw.list.head(5).alias("kw"), aw.list.head(8).alias("aw"),
                     aw.list.head(6).alias("aw6"))
    parts = [_pairs(pb, "kw", "kw", True), _pairs(pb, "kw", "aw", False), _pairs(pb, "aw6", "aw6", True)]
    for f, minlen in (("n", 2), ("k", 2), ("a", 1), ("d", 3), ("c", 4)):
        t = base.select("row", pl.col(f).alias("tok")).explode("tok")
        t = t.filter(pl.col("tok").str.len_chars() >= minlen)
        parts.append(t.with_columns(pl.lit(f).alias("f")))
    tf = pl.concat(parts)
    return tf.with_columns(
        ((pl.col("f") + "|" + pl.col("tok")).hash(seed=42) % DIM).cast(pl.Int32).alias("h")
    ).select("row", "h", "f").unique(["row", "h"])


def _pairs(pb: pl.DataFrame, lc: str, rc: str, same: bool) -> pl.DataFrame:
    t = pb.select("row", pl.col(lc).alias("_l"), pl.col(rc).alias("_r")).explode("_l").explode("_r").drop_nulls()
    if same:
        t = t.filter(pl.col("_l") < pl.col("_r"))
    tag = {"kw": "nn", "aw6": "aa"}[lc] if lc == rc else "na"
    return t.select("row", (pl.lit(tag + ":") + pl.col("_l") + "|" + pl.col("_r")).alias("tok"),
                    pl.lit("x").alias("f"))


def _ngrams(e: pl.Expr, n: int, max_len: int = 40) -> pl.Expr:
    """Vectorised character n-grams (strings shorter than n give themselves)."""
    grams = pl.concat_list([e.str.slice(i, n) for i in range(max_len - n + 1)])
    return pl.when(e.str.len_chars() < n).then(pl.concat_list([e])).otherwise(
        grams.list.eval(pl.element().filter(pl.element().str.len_chars() == n)))


def build_matrix(tf: pl.DataFrame, n_rows: int, idf: np.ndarray, weights: dict) -> sp.csr_matrix:
    tf = tf.filter(pl.col("f").is_in(list(weights)))
    h = tf["h"].to_numpy()
    w = idf[h] * tf["f"].replace_strict(weights, return_dtype=pl.Float32).to_numpy()
    keep = w > 0
    m = sp.csr_matrix((w[keep].astype(np.float32), (tf["row"].to_numpy()[keep], h[keep])),
                      shape=(n_rows, DIM))
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return (sp.diags((1.0 / norms).astype(np.float32)) @ m).tocsr()


# per-family document-frequency caps (fraction of index size): char n-grams are the most
# numerous and most expensive, so they are capped hardest
DF_CAP = {"n": 0.003, "k": 0.003, "a": 0.003, "d": 0.003, "c": 0.002, "x": 0.003}


def make_idf(tf: pl.DataFrame, n_docs: int, scale: float = 1.0) -> np.ndarray:
    dfreq = np.bincount(tf["h"].to_numpy(), minlength=DIM).astype(np.float32)
    idf = np.log((n_docs + 1) / (dfreq + 1)).astype(np.float32)
    for f, cap in DF_CAP.items():
        hs = tf.filter(pl.col("f") == f)["h"].unique().to_numpy()
        hs = hs[dfreq[hs] > max(cap * scale * n_docs, 50)]
        idf[hs] = 0.0                                    # too common: drop
    idf[dfreq == 0] = 0.0                                # never in index: useless
    return idf


PRUNE = {"full": 10, "name": 6, "addr": 6, "pair": 20}


def prune_rows(m: sp.csr_matrix, keep: int) -> sp.csr_matrix:
    """Keep only the `keep` heaviest (rarest) tokens of every query row. Retrieval cost is
    dominated by long posting lists of common tokens; the rare tokens identify the record."""
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    d = pl.DataFrame({"r": rows, "c": m.indices, "w": m.data})
    d = d.filter(pl.col("w").rank("ordinal", descending=True).over("r") <= keep)
    return sp.csr_matrix((d["w"].to_numpy(), (d["r"].to_numpy(), d["c"].to_numpy())), shape=m.shape)


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Exact cosine for given (row of A, row of B) pairs."""
    out = np.empty(len(ia), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(ia), step):
        out[s:s + step] = np.asarray(A[ia[s:s + step]].multiply(B[ib[s:s + step]]).sum(axis=1)).ravel()
    return out


def topk_candidates(s1: pl.DataFrame, q: pl.DataFrame, max_df_frac: float = 1.0,
                    chunk: int = 250_000, n_threads: int = max(os.cpu_count() - 2, 1), topk: dict = None,
                    out_dir: str = None, post=None) -> pl.DataFrame:
    """For each query row, the union of per-channel top-k S1 rows of the same country.
    Returns (s1, q, s_full, s_name, s_addr, score, rank)."""
    topk = topk or TOPK
    out = []
    for country in sorted(q["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        qc = q.filter(pl.col("country") == country)
        if s1c.height == 0 or qc.height == 0:
            continue
        # token table of the index built in slices (memory), row ids shifted back to global
        tf1 = pl.concat([token_frame(s1c.slice(o, 200_000)).with_columns(pl.col("row") + o)
                         for o in range(0, s1c.height, 200_000)])
        idf = make_idf(tf1, s1c.height, max_df_frac)  # max_df_frac scales DF_CAP
        M1 = {c: build_matrix(tf1, s1c.height, idf, w) for c, w in CHANNELS.items()}
        B = {c: M1[c].T.tocsr() for c in CHANNELS}
        del tf1
        s1_ids = s1c["entity_id"].to_numpy()
        for off in range(0, qc.height, chunk):
            qq = qc.slice(off, chunk)
            t0 = time.time()
            tfq = token_frame(qq)
            t1 = time.time()
            MQ = {c: build_matrix(tfq, qq.height, idf, w) for c, w in CHANNELS.items()}
            if off == 0:
                print(f"    token_frame {t1 - t0:.1f}s, build {time.time() - t1:.1f}s", flush=True)
            rows, cols = [], []
            for c in CHANNELS:
                t0 = time.time()
                C = sp_matmul_topn(prune_rows(MQ[c], PRUNE[c]), B[c], top_n=topk[c], threshold=0.02,
                                   n_threads=n_threads).tocsr()
                rows.append(np.repeat(np.arange(C.shape[0]), np.diff(C.indptr)))
                cols.append(C.indices)
                if off == 0:
                    print(f"    {c}: {time.time() - t0:.1f}s", flush=True)
            pairs = np.unique(np.stack([np.concatenate(rows), np.concatenate(cols)]), axis=1)
            iq, i1 = pairs[0], pairs[1]
            t0 = time.time()
            d = {f"s_{c}": _rowdot(MQ[c], M1[c], iq, i1) for c in CHANNELS}
            if off == 0:
                print(f"    rowdot {time.time() - t0:.1f}s for {len(iq)} pairs", flush=True)
            res = pl.DataFrame({"s1": s1_ids[i1], "q": qq["entity_id"].to_numpy()[iq], **d})
            res = res.with_columns(pl.col("s_full").alias("score")).with_columns(
                (pl.col("score").rank("ordinal", descending=True).over("q") - 1).cast(pl.Int16).alias("rank"))
            # cap candidates per query: keep the MAX_CAND best by their strongest channel score
            res = res.filter(pl.max_horizontal([f"s_{c}" for c in CHANNELS])
                             .rank("ordinal", descending=True).over("q") <= MAX_CAND)
            if post is not None:
                res = post(res)
            if out_dir:  # stream chunks to disk: the full candidate set does not fit in RAM
                res.write_parquet(f"{out_dir}/{country}_{off // chunk:04d}.parquet")
            else:
                out.append(res)
            print(f"  {country}: {off + qq.height}/{qc.height} queries", flush=True)
    return pl.concat(out) if out else None
