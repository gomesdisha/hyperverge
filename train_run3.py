import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from datasets import Dataset

MODEL     = "jhu-clsp/ettin-encoder-32m"
START_CKPT = "ckpt_best/checkpoint-237"
OUT_DIR   = "ckpt_run3"
MAX_LEN   = 64
BATCH_SIZE = 128
LR        = 8e-5  # Gentle fine-tuning LR
EPOCHS    = 3
SEED      = 42
TRAIN_PATH = "data/train.jsonl"
VAL_PATH   = "data/val.jsonl"
TEST_PATH  = "data/test.jsonl"

LABELS = [
    "O","I-COUNTERPARTY_NAME","I-PROCESSOR","I-TRANSACTION_METHOD",
    "I-BANK_SERVICE_EVENT","I-RECURRING_FLAG","I-FILLER_WORD","I-SEPARATOR_PUNCTUATION",
]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
FIELD_MAP = {
    "COUNTERPARTY_NAME": "counterparty",
    "PROCESSOR": "processor",
    "TRANSACTION_METHOD": "transaction_method",
    "RECURRING_FLAG": "recurring_flag",
    "BANK_SERVICE_EVENT": "bank_service_event",
    "FILLER_WORD": "filler_word",
    "SEPARATOR_PUNCTUATION": "separator_punctuation",
}
SCORED_FIELDS = ["counterparty", "transaction_method", "processor", "recurring_flag"]

# Deliberate loss weights: Boost the 3 scored fields, downweight unscored BANK_SERVICE_EVENT
CUSTOM_WEIGHTS = {
    "O": 0.25,
    "I-COUNTERPARTY_NAME": 3.5,     # Strong boost to fix counterparty recall
    "I-PROCESSOR": 5.0,             # Keep processor recall high
    "I-TRANSACTION_METHOD": 2.5,    # Already strong, maintain
    "I-BANK_SERVICE_EVENT": 0.35,   # Downweight so model prefers COUNTERPARTY
    "I-RECURRING_FLAG": 1.0,
    "I-FILLER_WORD": 0.5,
    "I-SEPARATOR_PUNCTUATION": 0.4,
}
weights_tensor = torch.tensor([CUSTOM_WEIGHTS[l] for l in LABELS], dtype=torch.float)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Custom weights:", {l: CUSTOM_WEIGHTS[l] for l in LABELS})

tok = AutoTokenizer.from_pretrained(MODEL)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_consensus(rows):
    id_to_rows = defaultdict(list)
    for r in rows: id_to_rows[r["id"]].append(r)
    merged = []
    for rid, rlist in id_to_rows.items():
        if len(rlist) == 1:
            merged.append(rlist[0])
            continue
        r0, r1 = rlist[0], rlist[1]
        tags0, tags1 = r0["ner_tags"], r1["ner_tags"]
        n = len(tags0)
        consensus = []
        for i in range(n):
            t0 = tags0[i] if i < len(tags0) else "O"
            t1 = tags1[i] if i < len(tags1) else "O"
            if t0 == t1: consensus.append(t0)
            elif t0 == "O": consensus.append(t1)
            elif t1 == "O": consensus.append(t0)
            else: consensus.append(t0)
        merged_row = dict(r0)
        merged_row["ner_tags"] = consensus
        merged.append(merged_row)
    return merged

def encode(batch, has_labels=True):
    enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
    if has_labels:
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(i)
            prev, labels = None, []
            for w in word_ids:
                if w is None: labels.append(-100)
                elif w != prev: labels.append(label2id.get(tags[w], 0))
                else: labels.append(-100)
                prev = w
            all_labels.append(labels)
        enc["labels"] = all_labels
    return enc

class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

@torch.no_grad()
def tag_words_batch(model, batch_tokens):
    enc = tok(batch_tokens, is_split_into_words=True, return_tensors="pt",
              truncation=True, padding=True, max_length=MAX_LEN).to(device)
    preds = model(**enc).logits.argmax(-1).tolist()
    batch_out = []
    for i, tokens in enumerate(batch_tokens):
        out, seen = ["O"] * len(tokens), set()
        for idx, w in enumerate(enc.word_ids(i)):
            if w is not None and w not in seen:
                out[w] = id2label[preds[i][idx]]
                seen.add(w)
        batch_out.append(out)
    return batch_out

def extract_fields(tokens, tags):
    rec = {v: [] for v in FIELD_MAP.values()}
    for token, tag in zip(tokens, tags):
        if tag.startswith("I-"):
            k = tag[2:]
            if k in FIELD_MAP: rec[FIELD_MAP[k]].append(token)
    return {k: " ".join(v) for k, v in rec.items()}

def run_inference(model, records, batch_size=64):
    model.eval()
    results = []
    for i in range(0, len(records), batch_size):
        chunk = records[i: i + batch_size]
        tags_batch = tag_words_batch(model, [r["tokens"] for r in chunk])
        for r, tags in zip(chunk, tags_batch):
            results.append({"id": r["id"], **extract_fields(r["tokens"], tags)})
    return results

def main():
    print("Loading data for Run 3 (Counterparty & Processor Boost)...")
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    test_rows = load_jsonl(TEST_PATH)

    train_c = build_consensus(train_rows)
    val_c = build_consensus(val_rows)
    train_ds = Dataset.from_list(train_c)
    val_ds = Dataset.from_list(val_c)

    def enc_lbl(b): return encode(b, True)
    train_enc = train_ds.map(enc_lbl, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ["input_ids","attention_mask","labels"]])
    val_enc = val_ds.map(enc_lbl, batched=True, remove_columns=[c for c in val_ds.column_names if c not in ["input_ids","attention_mask","labels"]])

    print(f"Loading initial model from: {START_CKPT}")
    model = AutoModelForTokenClassification.from_pretrained(START_CKPT)
    model.to(device)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        warmup_steps=50,
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = WeightedNERTrainer(
        class_weights=weights_tensor,
        model=model, args=args,
        train_dataset=train_enc, eval_dataset=val_enc,
        data_collator=DataCollatorForTokenClassification(tok),
    )

    print("Fine-tuning Run 3...")
    trainer.train()

    print("Running test inference with Run 3 model...")
    test_results = run_inference(model, test_rows, batch_size=64)

    # Apply recurring flag rules (which gave us 100% recall and 0.9052 F1)
    import re
    RECURRING_PATTERNS = [
        re.compile(r"\b(subscription|subscr)\b", re.IGNORECASE),
        re.compile(r"\b(recurring)\b", re.IGNORECASE),
        re.compile(r"\b(auto[\s-]?pay)\b", re.IGNORECASE),
        re.compile(r"\b(auto[\s-]?debit)\b", re.IGNORECASE),
        re.compile(r"\b(pre[\s-]?auth\w*)\b", re.IGNORECASE),
        re.compile(r"\b(auto[\s-]?approv\w*)\b", re.IGNORECASE),
    ]

    test_map = {r["id"]: r for r in test_rows}
    cleaned = []
    for r in test_results:
        raw_tokens = test_map[r["id"]]["tokens"]
        matched_words = []
        for token in raw_tokens:
            for pat in RECURRING_PATTERNS:
                if pat.search(token):
                    matched_words.append(token)
                    break
        rec_flag = " ".join(matched_words) if matched_words else ""
        cleaned.append({
            "id": r["id"],
            "counterparty": r.get("counterparty", ""),
            "transaction_method": r.get("transaction_method", ""),
            "processor": r.get("processor", ""),
            "recurring_flag": rec_flag,
            "bank_service_event": r.get("bank_service_event", ""),
        })

    json.dump(cleaned, open("predictions_run3.json", "w"), indent=2)
    # Update active predictions.json
    json.dump(cleaned, open("predictions.json", "w"), indent=2)
    print("Run 3 complete! Predictions written to predictions.json and predictions_run3.json")

if __name__ == "__main__":
    main()
