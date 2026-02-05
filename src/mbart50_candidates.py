#!/usr/bin/env python3

import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

def main():
    parser = argparse.ArgumentParser(description="Batch candidate generation with mBART-50")
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--batch_size",           type=int, default=32)
    parser.add_argument("--num_beams",            type=int, default=8)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens",       type=int, default=32)
    parser.add_argument("--max_src_length",       type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    for col in ("source", "src_lang", "tgt_lang", "volume"):
        if col not in df.columns:
            raise KeyError(f"Input CSV must contain '{col}' column")

    #clean inputs
    df = df.dropna(subset=["source"])
    df["source"] = df["source"].astype(str)
    df = df[df["source"].str.lower() != "source"]
    df = df[df["source"].str.strip() != ""].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_name)
    model     = MBartForConditionalGeneration.from_pretrained(args.model_name)\
                                              .to(device).eval()
    if device.type == "cuda":
        model.half()

    rows = []
    for (sl, tl), group in tqdm(df.groupby(["src_lang", "tgt_lang"]),
                               desc="Language groups"):
        sources = group["source"].tolist()
        volumes = group["volume"].tolist()

        tokenizer.src_lang = sl
        tokenizer.tgt_lang = tl
        forced_bos = tokenizer.lang_code_to_id[tl]

        for i in range(0, len(sources), args.batch_size):
            batch_src = sources[i : i + args.batch_size]
            batch_vol = volumes[i : i + args.batch_size]

            # sanity check ensure all entries are strings
            for j, s in enumerate(batch_src):
                if not isinstance(s, str):
                    print(f"Skipping non-string source at index {i+j}: {s!r}")
                    batch_src[j] = str(s)

            # tokenize batch
            enc = tokenizer(
                batch_src,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src_length
            ).to(device)

            # generate n-best candidates
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                max_new_tokens=args.max_new_tokens,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

            # reshape and dedupe per example
            for idx, src in enumerate(batch_src):
                start = idx * args.num_return_sequences
                stop  = start + args.num_return_sequences
                seen = set()
                for text in decoded[start:stop]:
                    t = text.strip()
                    if t and t not in seen:
                        seen.add(t)
                        rows.append({
                            "source":   src,
                            "src_lang": sl,
                            "target":   t,
                            "tgt_lang": tl,
                            "volume":   batch_vol[idx],
                        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()