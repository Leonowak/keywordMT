#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

nltk.download("wordnet", quiet=True)


def translate_batch(
    sources,
    tokenizer,
    model,
    device,
    batch_size,
    max_input_length,
    max_new_tokens,
    num_beams,
    model_type,
    src_lang,
    tgt_lang,
    forced_bos_token_id,
):
    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()

    preds = []
    for i in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch = sources[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend([p.strip() for p in decoded])
    return preds


def compute_metrics(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score
    ter = sacrebleu.metrics.TER().corpus_score(hyps, [refs]).score
    mets = [meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)]
    meteor = sum(mets) / len(mets)
    return bleu, ter, meteor


def main():
    ap = argparse.ArgumentParser(description="Unified MT evaluation")
    ap.add_argument("--model_dir", required=True,
                    help="Path to fine-tuned model or HF model name")
    ap.add_argument("--model_type", required=True, choices=["mbart", "m2m100", "helsinki"])
    ap.add_argument("--input_csv", required=True,
                    help="CSV with either (id,source,target) or (input,target)")
    ap.add_argument("--output_csv", required=True,
                    help="Where to save CSV with added")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_input_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--num_beams", type=int, default=8)
    ap.add_argument("--src_lang", type=str, default=None,
                    help="mBART: de_DE, en_XX, nl_XX; M2M100: de, en, nl")
    ap.add_argument("--tgt_lang", type=str, default=None,
                    help="mBART: de_DE, en_XX, nl_XX; M2M100: de, en, nl")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.input_csv)

    if {"source", "target"}.issubset(df.columns):
        pass
    elif {"input", "target"}.issubset(df.columns):
        df = df.rename(columns={"input": "source"})
    else:
        missing = {"source", "target"} - set(df.columns)
        raise KeyError(
            f"Input CSV must have either (id,source,target) or (input,target). "
            f"Missing (source/target) columns: {missing}"
        )

    if "id" not in df.columns:
        df["id"] = range(len(df))

    sources = df["source"].astype(str).tolist()
    refs    = df["target"].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    forced_bos_token_id = None

    if args.model_type == "mbart":
        if hasattr(tokenizer, "src_lang") and args.src_lang is not None:
            tokenizer.src_lang = args.src_lang
        if hasattr(tokenizer, "tgt_lang") and args.tgt_lang is not None:
            tokenizer.tgt_lang = args.tgt_lang
        if hasattr(tokenizer, "lang_code_to_id") and args.tgt_lang is not None:
            forced_bos_token_id = tokenizer.lang_code_to_id.get(args.tgt_lang, None)
        if model.config.decoder_start_token_id is None and forced_bos_token_id is not None:
            model.config.decoder_start_token_id = forced_bos_token_id

    elif args.model_type == "m2m100":
        if args.src_lang is None or args.tgt_lang is None:
            raise ValueError("For m2m100, please specify --src_lang and --tgt_lang (e.g. de, en, nl).")
        tokenizer.src_lang = args.src_lang
        tokenizer.tgt_lang = args.tgt_lang
        if hasattr(tokenizer, "get_lang_id"):
            forced_bos_token_id = tokenizer.get_lang_id(args.tgt_lang)

    elif args.model_type == "helsinki":
        forced_bos_token_id = None  # single-direction Marian models

    preds = translate_batch(
        sources,
        tokenizer,
        model,
        device,
        args.batch_size,
        args.max_input_length,
        args.max_new_tokens,
        args.num_beams,
        args.model_type,
        args.src_lang,
        args.tgt_lang,
        forced_bos_token_id,
    )

    bleu, ter, meteor = compute_metrics(refs, preds)
    print(f"BLEU:   {bleu:.2f}")
    print(f"TER:    {ter:.2f}")
    print(f"METEOR: {meteor:.2f}")

    df["prediction"] = preds
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions → {args.output_csv}")


if __name__ == "__main__":
    main()
