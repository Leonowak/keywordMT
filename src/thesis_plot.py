#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LANG_MAP = {
    "de_DE": "DE",
    "en_XX": "EN",
    "nl_XX": "NL",
}

PLOT_LANG_ORDER = ["DE", "NL", "EN"]


def load_and_stack(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)

        # src-side SV: volume
        src = df[["source", "src_lang", "volume"]].copy()
        src.rename(columns={"source": "term", "src_lang": "language", "volume": "SV"}, inplace=True)
        src["role"] = "source"

        # tgt-side SV: actual_volume
        if "target_norm" in df.columns:
            tgt_term_col = "target_norm"
        else:
            tgt_term_col = "target"

        tgt = df[[tgt_term_col, "tgt_lang", "actual_volume"]].copy()
        tgt.rename(columns={tgt_term_col: "term", "tgt_lang": "language", "actual_volume": "SV"}, inplace=True)
        tgt["role"] = "target"

        stacked = pd.concat([src, tgt], ignore_index=True)
        stacked["file"] = Path(p).name
        frames.append(stacked)

    long_df = pd.concat(frames, ignore_index=True)

    long_df["language"] = long_df["language"].map(LANG_MAP)
    long_df = long_df[long_df["language"].isin(PLOT_LANG_ORDER)]

    long_df["SV"] = pd.to_numeric(long_df["SV"], errors="coerce")
    long_df["term"] = long_df["term"].astype(str).str.strip()

    long_df = long_df.dropna(subset=["SV"])
    long_df = long_df[long_df["SV"] > 0]

    return long_df


def deduplicate(long_df: pd.DataFrame) -> pd.DataFrame:
    dedup = (
        long_df.groupby(["role", "language", "term"], as_index=False)["SV"]
        .max()
        .sort_values(["role", "language", "SV"], ascending=[True, True, False])
    )
    return dedup


def make_violin(df: pd.DataFrame, outpath: Path, title: str) -> None:
    data = [df.loc[df["language"] == lang, "SV"].values for lang in PLOT_LANG_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(PLOT_LANG_ORDER) + 1))
    ax.set_xticklabels(PLOT_LANG_ORDER)
    ax.set_yscale("log")
    ax.set_xlabel("Language")
    ax.set_ylabel("Search Volume (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def make_box(df: pd.DataFrame, outpath: Path, title: str) -> None:
    data = [df.loc[df["language"] == lang, "SV"].values for lang in PLOT_LANG_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, labels=PLOT_LANG_ORDER, showfliers=False)  # hide extreme outliers for readability
    ax.set_yscale("log")
    ax.set_xlabel("Language")
    ax.set_ylabel("Search Volume (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "mbart50_de_en_cands_enriched_clean.csv",
            "mbart50_de_nl_cands_enriched_clean.csv",
            "mbart50_en_nl_cands_enriched_clean.csv",
        ],
        help="Input CSV paths",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "--title",
        default="SV distribution by language (DE/NL/EN)",
        help="Plot title",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_df = load_and_stack(args.inputs)
    dedup_df = deduplicate(long_df)

    print(f"Rows (raw, positive SV only): {len(long_df)}")
    print(f"Rows (deduplicated by role+language+term): {len(dedup_df)}")

    stats = dedup_df.groupby("language")["SV"].describe(percentiles=[0.5])[["count", "min", "50%", "max"]]
    print("\nSV summary (deduplicated):")
    print(stats)

    violin_path = outdir / "sv_violin_de_nl_en.png"
    box_path = outdir / "sv_box_de_nl_en.png"

    make_violin(dedup_df, violin_path, args.title)
    make_box(dedup_df, box_path, args.title)

    print(f"\nSaved violin plot: {violin_path.resolve()}")
    print(f"Saved box plot:   {box_path.resolve()}")


if __name__ == "__main__":
    main()