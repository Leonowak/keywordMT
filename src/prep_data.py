#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.rstrip(".,;!?")
    )

def is_valid_candidate(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 2:             
        return False
    if t in {"news news", "webs", "nltv"}:  
        return False
    if "  " in t:               
        return False
    return True

def main():
    ap = argparse.ArgumentParser(description="Clean & prepare keyword pairs for training")
    ap.add_argument("--input_csv",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--topk_per_source", type=int, default=1,
                    help="Keep top-K targets per source by actual_volume (default: 1)")
    ap.add_argument("--min_target_sv", type=int, default=1,
                    help="Minimum target actual_volume to keep (default: 1)")
    ap.add_argument("--min_chars", type=int, default=2,
                    help="Drop rows where input or target shorter than this")
    ap.add_argument("--max_chars", type=int, default=60,
                    help="Drop rows where input or target longer than this")
    ap.add_argument("--train_frac", type=float, default=0.9,
                    help="Train split fraction (by unique source)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_upweighted", action="store_true",
                    help="Also write train_upweighted.csv using simple SV buckets")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)


    needed = {"source","src_lang","target","tgt_lang","actual_volume"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in input: {missing}")

    df["target_norm"] = normalize_text(df["target"])

    df = df[df["actual_volume"].fillna(0).astype(float) >= args.min_target_sv].copy()

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    df = df[
        (df["source"].str.len().between(args.min_chars, args.max_chars)) &
        (df["target"].str.len().between(args.min_chars, args.max_chars))
    ].copy()

    df = df[df["target_norm"].apply(is_valid_candidate)].copy()

    df = df.drop_duplicates(subset=["source","target_norm"]).copy()

    df = (
        df.sort_values(["source","actual_volume"], ascending=[True, False])
          .groupby("source", as_index=False)
          .head(args.topk_per_source)
          .reset_index(drop=True)
    )

    clean_full_path = os.path.join(args.output_dir, "clean_full.csv")
    df.to_csv(clean_full_path, index=False)

    final = df.rename(columns={"source":"input"})[["input","target"]].copy()

    rng = np.random.default_rng(args.seed)
    sources = df["source"].unique()
    rng.shuffle(sources)
    cut = int(len(sources) * args.train_frac)
    train_src = set(sources[:cut])

    train = final[final["input"].isin(train_src)].reset_index(drop=True)
    eval_  = final[~final["input"].isin(train_src)].reset_index(drop=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    eval_path  = os.path.join(args.output_dir, "eval.csv")
    train.to_csv(train_path, index=False)
    eval_.to_csv(eval_path, index=False)

    if args.make_upweighted:
        df_w = df.copy()
        # Buckets tweak thresholds if needed
        bins   = [-1, 100, 1_000, 10_000, 100_000, np.inf]
        labels = [1,   2,    3,      4,       5]   # replication factor
        df_w["sv_bucket"] = pd.cut(df_w["actual_volume"], bins=bins, labels=labels).astype(int)

        train_wide = df_w[df_w["source"].isin(train_src)].copy()
        train_up = train_wide.loc[train_wide.index.repeat(train_wide["sv_bucket"])]
        train_up = train_up.rename(columns={"source":"input"})[["input","target"]].reset_index(drop=True)

        train_up_path = os.path.join(args.output_dir, "train_upweighted.csv")
        train_up.to_csv(train_up_path, index=False)
        print(f"Wrote upweighted train to {train_up_path}")

    print(f"Rows → clean_full: {len(df)} | train: {len(train)} | eval: {len(eval_)}")
    print(f"Saved:\n- {clean_full_path}\n- {train_path}\n- {eval_path}")

if __name__ == "__main__":
    main()
