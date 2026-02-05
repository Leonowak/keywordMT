#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd


def process_file(
    path,
    top_per_source=5,
    max_kws=None,
    sort_by="volume"
):
    df = pd.read_csv(path)

    df["target_norm"] = (
        df["target"]
        .str.lower()
        .str.strip()
        .str.rstrip(".,;!?")
    )

    df = df.drop_duplicates(subset=["source", "target_norm"])
    df = df.sort_values(sort_by, ascending=False)

    df = (
        df.groupby("source", as_index=False)
        .head(top_per_source)
        .reset_index(drop=True)
    )

    if max_kws is not None:
        df = df.head(max_kws)

    df = df[["source", "src_lang", "target", "tgt_lang", "volume"]]

    base, ext = os.path.splitext(path)
    out_path = f"{base}_clean{ext}"
    df.to_csv(out_path, index=False)

    print(f"Processed {path} → {out_path} ({len(df)} rows)")


def main(input_glob, top_per_source, max_kws):
    files = glob.glob(input_glob)
    if not files:
        print(f"No files matched: {input_glob}")
        return

    for path in files:
        process_file(
            path,
            top_per_source=top_per_source,
            max_kws=max_kws,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_glob",
        default="*_cands.csv",
    )
    p.add_argument(
        "--top_per_source",
        type=int,
        default=5,
    )
    p.add_argument(
        "--max_kws",
        type=int,
        default=None,
    )

    args = p.parse_args()
    main(args.input_glob, args.top_per_source, args.max_kws)
