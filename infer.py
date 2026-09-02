"""
Standalone inference script - loads saved checkpoint and generates predictions.json
Usage: python infer.py [checkpoint_dir] [--apply-recurring]
"""
import sys
import os
import json
import re
import torch
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL     = "jhu-clsp/ettin-encoder-32m"
MAX_LEN   = 64
PRED_PATH = "predictions.json"
TEST_PATH = "data/test.jsonl"
VAL_PATH  = "data/val.jsonl"

LABELS = [
    "O","I-COUNTERPARTY_NAME","I-PROCESSOR","I-TRANSACTION_METHOD",
    "I-BANK_SERVICE_EVENT","I-RECURRING_FLAG","I-FILLER_WORD","I-SEPARATOR_PUNCTUATION",
]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
FIELD_MAP = {
    "COUNTERPARTY_NAME": "counterparty", "PROCESSOR": "processor",
    "TRANSACTION_METHOD": "transaction_method", "RECURRING_FLAG": "recurring_flag",
    "BANK_SERVICE_EVENT": "bank_service_event", "FILLER_WORD": "filler_word",
    "SEPARATOR_PUNCTUATION": "separator_punctuation",
}
SCORED_FIELDS = ["counterparty", "transaction_method", "processor", "recurring_flag"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_consensus(rows):
    id_to_rows = defaultdict(list)
    for r in rows:
        id_to_rows[r["id"]].append(r)
    merged = []
    for rid, rlist in id_to_rows.items():
        if len(rlist) == 1:
            merged.append(rlist[0])
            continue
        r0, r1 = rlist[0], rlist[1]
        tags0, tags1 = r0["ner_tags"], r1["ner_tags"]
        consensus = []
        for i in range(len(tags0)):
            t0 = tags0[i] if i < len(tags0) else "O"
            t1 = tags1[i] if i < len(tags1) else "O"
            if t0 == t1:
                consensus.append(t0)
            elif t0 == "O":
                consensus.append(t1)
            elif t1 == "O":
                consensus.append(t0)
            else:
                consensus.append(t0)
        merged_row = dict(r0)
        merged_row["ner_tags"] = consensus
        merged.append(merged_row)
    return merged

@torch.no_grad()
def tag_words_batch(model, tok, batch_tokens):
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
            field_key = tag[2:]
            if field_key in FIELD_MAP:
                rec[FIELD_MAP[field_key]].append(token)
    return {k: " ".join(v) for k, v in rec.items()}

def run_inference(model, tok, records, batch_size=64):
    model.eval()
    results = []
    for i in range(0, len(records), batch_size):
        chunk = records[i: i + batch_size]
        tags_batch = tag_words_batch(model, tok, [r["tokens"] for r in chunk])
        for r, tags in zip(chunk, tags_batch):
            fields = extract_fields(r["tokens"], tags)
            results.append({"id": r["id"], **fields})
    return results

def evaluate_predictions(val_rows, predictions_by_id):
    consensus_val = build_consensus(val_rows)
    gold_by_id = {}
    for r in consensus_val:
        rec = {v: [] for v in FIELD_MAP.values()}
        for token, tag in zip(r["tokens"], r["ner_tags"]):
            if tag.startswith("I-"):
                field_key = tag[2:]
                if field_key in FIELD_MAP:
                    rec[FIELD_MAP[field_key]].append(token)
        gold_by_id[r["id"]] = {k: " ".join(v) for k, v in rec.items()}

    field_stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in SCORED_FIELDS}
    for rid, pred in predictions_by_id.items():
        if rid not in gold_by_id:
            continue
        gold = gold_by_id[rid]
        for field in ["counterparty", "transaction_method", "processor"]:
            pred_tokens = pred.get(field, "").split() if pred.get(field) else []
            gold_tokens = gold.get(field, "").split() if gold.get(field) else []
            pred_c = Counter(t.lower() for t in pred_tokens)
            gold_c = Counter(t.lower() for t in gold_tokens)
            tp = sum((pred_c & gold_c).values())
            field_stats[field]["tp"] += tp
            field_stats[field]["fp"] += sum(pred_c.values()) - tp
            field_stats[field]["fn"] += sum(gold_c.values()) - tp
        pred_pos = len(pred.get("recurring_flag", "").strip()) > 0
        gold_pos = len(gold.get("recurring_flag", "").strip()) > 0
        if pred_pos and gold_pos:
            field_stats["recurring_flag"]["tp"] += 1
        elif pred_pos and not gold_pos:
            field_stats["recurring_flag"]["fp"] += 1
        elif not pred_pos and gold_pos:
            field_stats["recurring_flag"]["fn"] += 1

    f1_scores = {}
    for field in SCORED_FIELDS:
        s = field_stats[field]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1_scores[field] = f1
        print(f"  {field:25s}: P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    macro_f1 = np.mean(list(f1_scores.values()))
    print(f"  MACRO-F1: {macro_f1:.4f}")
    return macro_f1, f1_scores


def main():
    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "ckpt_best"
    apply_recurring = "--apply-recurring" in sys.argv

    print(f"Loading model from: {ckpt_dir}")
    tok = AutoTokenizer.from_pretrained(MODEL)

    # Try to load from checkpoint directory
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            ckpt_dir, num_labels=len(LABELS), id2label=id2label, label2id=label2id
        )
    except Exception as e:
        print(f"Failed to load from {ckpt_dir}: {e}")
        print("Trying to load from pretrained MODEL...")
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True
        )
    model.to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load val data and evaluate
    print("\nLoading val data...")
    val_rows = load_jsonl(VAL_PATH)
    val_consensus = build_consensus(val_rows)
    print(f"Val: {len(val_rows)} rows, {len(val_consensus)} unique IDs")

    print("Running val inference...")
    val_results = run_inference(model, tok, val_consensus, batch_size=64)
    val_preds_by_id = {r["id"]: r for r in val_results}

    # Override recurring_flag to empty (no annotators used it)
    for rec in val_preds_by_id.values():
        rec["recurring_flag"] = ""

    print("\nLocal val metrics:")
    evaluate_predictions(val_rows, val_preds_by_id)

    # Load test and generate predictions
    print("\nLoading test data...")
    test_rows = load_jsonl(TEST_PATH)
    print(f"Test: {len(test_rows)} rows")

    print("Running test inference...")
    test_results = run_inference(model, tok, test_rows, batch_size=64)

    # Post-processing
    for rec in test_results:
        rec["recurring_flag"] = ""  # No annotator used recurring_flag
        for field in SCORED_FIELDS + ["bank_service_event"]:
            if field not in rec:
                rec[field] = ""

    # Verify
    assert len(test_results) == len(test_rows), "Missing predictions!"
    assert len({r["id"] for r in test_results}) == len(test_rows), "Duplicate IDs!"

    json.dump(test_results, open(PRED_PATH, "w"), indent=2)
    print(f"\nWrote {len(test_results)} predictions -> {PRED_PATH}")
    print("Sample:", json.dumps(test_results[0], indent=2))

    # Check recurring flag stats
    rec_count = sum(1 for r in test_results if r.get("recurring_flag", ""))
    print(f"Predictions with recurring_flag: {rec_count}/{len(test_results)}")

if __name__ == "__main__":
    main()
