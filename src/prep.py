"""Step 1: normalise every source file once and cache it as parquet (chunked, low memory).

Usage: python src/prep.py --work <work dir>
"""
import argparse
import os

import polars as pl

from normalize import normalize

CHUNK = 1_000_000


def prep_file(src: str, dst: str) -> None:
    n = pl.scan_parquet(src).select(pl.len()).collect().item()
    parts = []
    for off in range(0, n, CHUNK):
        lf = pl.scan_parquet(src).slice(off, CHUNK)
        part = normalize(lf).collect()
        tmp = f"{dst}.part{off // CHUNK}"
        part.write_parquet(tmp)
        parts.append(tmp)
    # write to a temp name and rename, so readers never see a half-written file
    pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    for p in parts:
        try:
            os.remove(p)
        except PermissionError:  # Windows: file briefly locked (e.g. antivirus scan)
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    for split in ("train", "test"):
        for s in (1, 2, 3):
            src = os.path.join(args.work, f"{split}_source{s}.parquet")
            dst = os.path.join(args.work, f"norm_{split}_source{s}.parquet")
            if not os.path.exists(dst):
                prep_file(src, dst)
                print("done", dst, flush=True)


if __name__ == "__main__":
    main()
