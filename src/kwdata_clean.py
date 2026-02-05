import argparse
import pandas as pd


def prune_keywords(csv_path, output_path, include_intents=True):
    df = pd.read_csv(
        csv_path,
        sep="\t",
        encoding="utf-16-le",
        engine="python"
    )

    cleaned = []
    for col in df.columns:
        c = col.replace("\x00", "").replace("\ufeff", "").strip()
        if c.startswith('"') and c.endswith('"'):
            c = c[1:-1]
        if c == "#":
            c = "ID"
        cleaned.append(c)
    df.columns = cleaned
    print("Cleaned columns:", df.columns.tolist())

    keep = ["ID", "Keyword", "Volume", "SERP Features"]
    if include_intents:
        keep.append("Intents")

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found after cleaning: {missing}")

    clean = df[keep].copy()
    clean.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Pruned {csv_path} → {output_path} ({', '.join(keep)})")


def build_parallel(
    src_clean_csv: str,
    tgt_clean_csv: str,
    output_path: str,
    include_intents: bool = True
):

    src = pd.read_csv(src_clean_csv)
    tgt = pd.read_csv(tgt_clean_csv)
    assert len(src) == len(tgt), "Source and target must have same row count"

    def make_input(row):
        parts = [
            f"Keyword: {row.Keyword}",
            f"Volume: {row.Volume}",
            f"SERP: {row['SERP Features']}"
        ]
        if include_intents:
            parts.append(f"Intents: {row.Intents}")
        return " | ".join(parts)

    inputs = src.apply(make_input, axis=1)
    targets = tgt["Keyword"]
    out = pd.DataFrame({"input": inputs, "target": targets})
    out.to_csv(output_path, index=False)
    print(f"Built parallel {src_clean_csv} + {tgt_clean_csv} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune Ahrefs keyword CSVs and (optionally) build parallel training files"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prune", help="Drop unwanted columns")
    p1.add_argument("in_csv")
    p1.add_argument("out_csv")
    p1.add_argument(
        "--no-intents", action="store_true",
        help="Omit the Intents column"
    )

    p2 = sub.add_parser("parallel", help="Zip two pruned CSVs into input/target pairs")
    p2.add_argument("src_csv", help="Pruned source-language CSV")
    p2.add_argument("tgt_csv", help="Pruned target-language CSV")
    p2.add_argument("out_csv")
    p2.add_argument(
        "--no-intents", action="store_true",
        help="Omit Intents from the input string"
    )

    args = parser.parse_args()
    if args.cmd == "prune":
        prune_keywords(
            args.in_csv,
            args.out_csv,
            include_intents=not args.no_intents
        )
    else:
        build_parallel(
            args.src_csv,
            args.tgt_csv,
            args.out_csv,
            include_intents=not args.no_intents
        )
