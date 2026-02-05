#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Tuple, Dict

import pandas as pd
import torch
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

nltk.download("wordnet", quiet=True)

FLORES_CODE: Dict[str, str] = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "nl": "nld_Latn",
}

MBART50_CODE: Dict[str, str] = {
    "deu_Latn": "de_DE",
    "eng_Latn": "en_XX",
    "nld_Latn": "nl_XX",
}

M2M100_CODE: Dict[str, str] = {
    "deu_Latn": "de",
    "eng_Latn": "en",
    "nld_Latn": "nl",
}


def parse_pair(pair: str) -> Tuple[str, str]:
    if "-" not in pair:
        raise ValueError(f"Invalid pair '{pair}'. Use e.g. de-en or deu_Latn-eng_Latn")

    a, b = pair.split("-", 1)

    if "_" in a and "_" in b:
        return a, b

    if a not in FLORES_CODE or b not in FLORES_CODE:
        raise ValueError(
            f"Unsupported pair '{pair}'. Supported short codes: {sorted(FLORES_CODE.keys())}"
        )
    return FLORES_CODE[a], FLORES_CODE[b]


def load_data_from_csv(input_csv: str):
    df = pd.read_csv(input_csv)
    if {"source", "target"}.issubset(df.columns):
        pass
    elif {"input", "target"}.issubset(df.columns):
        df = df.rename(columns={"input": "source"})
    else:
        raise KeyError("CSV must have (source,target) or (input,target)")
    sources = df["source"].astype(str).tolist()
    refs = df["target"].astype(str).tolist()
    return df, sources, refs


def load_data_from_flores(
    flores_pair: str, split: str, max_eval_samples: int = 0
):

    from datasets import load_dataset

    src, tgt = parse_pair(flores_pair)
    config = f"{src}-{tgt}"

    ds = load_dataset("facebook/flores", config, split=split)

    src_field = f"sentence_{src}"
    tgt_field = f"sentence_{tgt}"

    sources = [ex[src_field] for ex in ds]
    refs = [ex[tgt_field] for ex in ds]

    if max_eval_samples and max_eval_samples > 0:
        sources = sources[:max_eval_samples]
        refs = refs[:max_eval_samples]

    df = pd.DataFrame({"source": sources, "target": refs})
    return df, sources, refs, src, tgt

def compute_metrics(refs, hyps):
    bleu_metric = sacrebleu.metrics.BLEU(tokenize="13a")
    bleu = bleu_metric.corpus_score(hyps, [refs]).score
    ter = sacrebleu.metrics.TER().corpus_score(hyps, [refs]).score

    mets = [meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)]
    meteor = sum(mets) / len(mets) if mets else 0.0

    sig = bleu_metric.get_signature()
    return bleu, ter, meteor, sig

def _forced_bos(tokenizer, model_family: str, tgt_lang_code: Optional[str]):

    if not tgt_lang_code:
        return None

    if model_family == "mbart50":
        if not hasattr(tokenizer, "lang_code_to_id"):
            raise ValueError("Tokenizer has no lang_code_to_id; is this really an mBART-50 tokenizer?")
        if tgt_lang_code not in tokenizer.lang_code_to_id:
            raise ValueError(f"Unknown mBART-50 target lang code: {tgt_lang_code}")
        return tokenizer.lang_code_to_id[tgt_lang_code]

    if model_family == "m2m100":
        if not hasattr(tokenizer, "get_lang_id"):
            raise ValueError("Tokenizer has no get_lang_id; is this really an M2M100 tokenizer?")
        return tokenizer.get_lang_id(tgt_lang_code)

    return None


def translate_seq2seq(
    sources,
    tokenizer,
    model,
    device,
    batch_size,
    max_input_length,
    max_new_tokens,
    num_beams,
    model_family: str,
    flores_src: Optional[str] = None,
    flores_tgt: Optional[str] = None,
):

    model.eval()
    preds = []

    src_lang_code: Optional[str] = None
    tgt_lang_code: Optional[str] = None

    if model_family in {"mbart50", "m2m100"}:
        if not flores_src or not flores_tgt:
            raise ValueError(f"{model_family} requires FLORES src/tgt codes (internal error).")

        if model_family == "mbart50":
            src_lang_code = MBART50_CODE.get(flores_src)
            tgt_lang_code = MBART50_CODE.get(flores_tgt)
        else:
            src_lang_code = M2M100_CODE.get(flores_src)
            tgt_lang_code = M2M100_CODE.get(flores_tgt)

        if not src_lang_code or not tgt_lang_code:
            raise ValueError(
                f"Missing mapping for FLORES codes: {flores_src}->{src_lang_code}, {flores_tgt}->{tgt_lang_code}"
            )

        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang_code
        else:
            raise ValueError(f"Tokenizer does not expose src_lang; cannot set source language for {model_family}.")

    forced_bos_token_id = _forced_bos(tokenizer, model_family, tgt_lang_code)

    for i in tqdm(range(0, len(sources), batch_size), desc=f"Translating ({model_family})"):
        batch_src = sources[i : i + batch_size]

        enc = tokenizer(
            batch_src,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True if num_beams > 1 else False,
            forced_bos_token_id=forced_bos_token_id,
        )
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        batch_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend([p.strip() for p in batch_preds])

    return preds

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate MBART-50  M2M100 & Helsinki-NLP models on FLORES-200)."
    )

    ap.add_argument("--model_family", required=True, choices=["mbart50", "m2m100", "helsinki"])
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--use_flores", action="store_true", help="Use FLORES-200 from HF.")
    ap.add_argument("--flores_pair", type=str, default="de-en", help="de-en / en-nl / de-nl or FLORES-coded.")
    ap.add_argument("--flores_split", type=str, default="devtest", choices=["dev", "devtest"])
    ap.add_argument("--input_csv", default="")
    ap.add_argument("--max_eval_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_input_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--compute_comet", action="store_true")
    ap.add_argument("--comet_model", type=str, default="Unbabel/wmt22-comet-da")
    ap.add_argument("--comet_batch_size", type=int, default=8)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    flores_src: Optional[str] = None
    flores_tgt: Optional[str] = None

    if args.use_flores:
        df, sources, refs, flores_src, flores_tgt = load_data_from_flores(
            args.flores_pair, args.flores_split, args.max_eval_samples
        )
        print(f"Loaded FLORES pair: {flores_src} -> {flores_tgt} | split={args.flores_split} | n={len(sources)}")
    else:
        if not args.input_csv:
            raise ValueError("Provide --use_flores or --input_csv.")
        df, sources, refs = load_data_from_csv(args.input_csv)
        if args.max_eval_samples and args.max_eval_samples > 0:
            df = df.iloc[: args.max_eval_samples].copy()
            sources = sources[: args.max_eval_samples]
            refs = refs[: args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    if device.type == "cuda":
        try:
            model.half()
        except Exception:
            pass

    preds = translate_seq2seq(
        sources=sources,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        model_family=args.model_family,
        flores_src=flores_src,
        flores_tgt=flores_tgt,
    )

    bleu, ter, meteor, sig = compute_metrics(refs, preds)
    print(f"BLEU:   {bleu:.2f}")
    print(f"TER:    {ter:.2f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"SacreBLEU signature: {sig}")

    # COMET
    comet_sys: Optional[float] = None
    comet_seg = None
    if args.compute_comet:
        from comet import download_model, load_from_checkpoint

        ckpt = download_model(args.comet_model)
        comet_model = load_from_checkpoint(ckpt)
        comet_model.to(device)

        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, preds, refs)]
        out = comet_model.predict(
            data,
            batch_size=args.comet_batch_size,
            gpus=1 if device.type == "cuda" else 0,
            progress_bar=True,
        )

        # comet returns either a dict or a Prediction object depending on version
        if isinstance(out, dict):
            comet_sys = float(out["system_score"])
            comet_seg = out.get("scores", None)
        else:
            comet_sys = float(out.system_score)
            comet_seg = getattr(out, "scores", None)

        print(f"COMET:  {comet_sys:.4f}")

    df["prediction"] = preds
    if comet_sys is not None:
        df["comet_score_system"] = comet_sys
    if comet_seg is not None:
        df["comet_score_segment"] = comet_seg

    df.to_csv(args.output_csv, index=False)
    print(f"Saved → {args.output_csv}")


if __name__ == "__main__":
    main()
