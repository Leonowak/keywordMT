#!/usr/bin/env python3
import argparse, math, os, random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if not {"input", "target"}.issubset(df.columns):
        raise KeyError(f"CSV must contain columns: input,target. Got: {df.columns.tolist()}")
    return Dataset.from_pandas(df[["input", "target"]])


def preprocess_dataset(ds, tokenizer, max_src_len, max_tgt_len, num_proc=1):
    def map_fn(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_src_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_tgt_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(
        map_fn,
        batched=True,
        num_proc=max(1, num_proc),
        remove_columns=ds.column_names,
    )


def main():
    ap = argparse.ArgumentParser(description="Unified fine-tuning for mBART, M2M100, Helsinki-NLP")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--eval_csv", required=True)
    ap.add_argument("--model_name", required=True,
                    help="HF model name or checkpoint path")
    ap.add_argument("--model_type", required=True, choices=["mbart", "m2m100", "helsinki"],
                    help="Model family: mbart | m2m100 | helsinki")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_src", type=int, default=64)
    ap.add_argument("--max_tgt", type=int, default=32)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--num_proc", type=int, default=4)


    ap.add_argument("--src_lang", type=str, default=None,
                    help="Source language code (mBART: de_DE, M2M100: de,)")
    ap.add_argument("--tgt_lang", type=str, default=None,
                    help="Target language code (mBART: en_XX, M2M100: en,)")

    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data loading
    train_ds = load_dataset(args.train_csv)
    eval_ds = load_dataset(args.eval_csv)

    # Models + Tokens
    # different models require different lang tokens!
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    forced_bos_token_id = None

    if args.model_type == "mbart":
        if args.src_lang is not None and hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = args.src_lang
        if args.tgt_lang is not None and hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = args.tgt_lang
        if hasattr(tokenizer, "lang_code_to_id") and args.tgt_lang is not None:
            forced_bos_token_id = tokenizer.lang_code_to_id.get(args.tgt_lang, None)
        if model.config.decoder_start_token_id is None and forced_bos_token_id is not None:
            model.config.decoder_start_token_id = forced_bos_token_id

    elif args.model_type == "m2m100":
        if args.src_lang is None or args.tgt_lang is None:
            raise ValueError(
                "For model_type=m2m100, please set --src_lang and --tgt_lang "
                "(e.g., de, en, nl)."
            )

        tokenizer.src_lang = args.src_lang
        tokenizer.tgt_lang = args.tgt_lang 

        if hasattr(tokenizer, "get_lang_id"):
            forced_bos_token_id = tokenizer.get_lang_id(args.tgt_lang)
        else:
            forced_bos_token_id = None

    elif args.model_type == "helsinki":
        forced_bos_token_id = None

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(device)

    train_ds_enc = preprocess_dataset(train_ds, tokenizer, args.max_src, args.max_tgt, args.num_proc)
    eval_ds_enc = preprocess_dataset(eval_ds, tokenizer, args.max_src, args.max_tgt, args.num_proc)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )

    train_loader = DataLoader(
        train_ds_enc,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds_enc,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if step % max(1, args.grad_accum_steps) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss += loss.item()

        avg_train = (total_loss * max(1, args.grad_accum_steps)) / len(train_loader)
        print(f"[Epoch {epoch}] train loss: {avg_train:.4f}")

        model.eval()
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                total_eval_loss += loss.item()

        avg_eval = total_eval_loss / len(eval_loader)
        print(f"[Epoch {epoch}] eval loss:  {avg_eval:.4f}")

        #Save predictions
        model.eval()
        print("\nSample predictions:")
        n_samples = min(10, len(eval_ds))
        if n_samples > 0:
            idx = np.random.choice(len(eval_ds), size=n_samples, replace=False).tolist()
            sample_inputs = [eval_ds[i]["input"] for i in idx]

            enc = tokenizer(
                text=sample_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_src,
            ).to(device)

            gen_kwargs = {
                "max_new_tokens": args.max_tgt,
                "num_beams": 4,
            }
            if forced_bos_token_id is not None:
                gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

            with torch.no_grad():
                gen_tokens = model.generate(**enc, **gen_kwargs)

            preds = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            for inp, pred in zip(sample_inputs, preds):
                print(f"- [IN]: {inp}\n  [OUT]: {pred}")
        print("-" * 60)


    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)


if __name__ == "__main__":
    main()
