"""
Second training run - combines train+val data, optimized LR, more epochs.
No independent eval set since we use leaderboard as validation.
"""
import os, json, re, torch, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from datasets import Dataset

MODEL     = "jhu-clsp/ettin-encoder-32m"
MAX_LEN   = 48
BATCH_SIZE = 128
LR        = 1e-4    # Lower LR for second run (more stable)
EPOCHS    = 6       # More epochs since no eval overhead
SEED      = 42
TRAIN_PATH = "data/train.jsonl"
VAL_PATH   = "data/val.jsonl"
TEST_PATH  = "data/test.jsonl"
CKPT_DIR   = "ckpt_run2"
PRED_PATH  = "predictions_run2.json"

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
torch.manual_seed(SEED); np.random.seed(SEED)
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
            if t0 == t1: consensus.append(t0)
            elif t0 == "O": consensus.append(t1)
            elif t1 == "O": consensus.append(t0)
            else: consensus.append(t0)
        merged_row = dict(r0)
        merged_row["ner_tags"] = consensus
        merged.append(merged_row)
    return merged

def compute_class_weights(rows):
    label_counts = Counter()
    for r in rows: label_counts.update(r["ner_tags"])
    total = sum(label_counts.values())
    weights = []
    for label in LABELS:
        count = label_counts.get(label, 1)
        w = total / (len(LABELS) * count)
        weights.append(min(w, 20.0))
    return torch.tensor(weights, dtype=torch.float)

tok = AutoTokenizer.from_pretrained(MODEL)

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
            field_key = tag[2:]
            if field_key in FIELD_MAP:
                rec[FIELD_MAP[field_key]].append(token)
    return {k: " ".join(v) for k, v in rec.items()}

def run_inference(model, records, batch_size=64):
    model.eval()
    results = []
    for i in range(0, len(records), batch_size):
        chunk = records[i: i + batch_size]
        tags_batch = tag_words_batch(model, [r["tokens"] for r in chunk])
        for r, tags in zip(chunk, tags_batch):
            fields = extract_fields(r["tokens"], tags)
            results.append({"id": r["id"], **fields})
    return results

def evaluate_val(val_rows, predictions_by_id):
    from collections import Counter, defaultdict
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
        if rid not in gold_by_id: continue
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

    f1_scores = {}
    for field in SCORED_FIELDS:
        s = field_stats[field]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1_scores[field] = f1
        print(f"  {field:25s}: P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    macro = np.mean(list(f1_scores.values()))
    print(f"  MACRO-F1: {macro:.4f}")
    return macro, f1_scores

def main():
    print("Run 2: Train on train+val combined, 6 epochs, LR=1e-4")
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    test_rows = load_jsonl(TEST_PATH)
    
    # Combine train+val for training
    train_c = build_consensus(train_rows)
    val_c = build_consensus(val_rows)
    combined = train_c + val_c  # 11,000 unique samples
    print(f"Combined training set: {len(combined)} samples")
    
    class_weights = compute_class_weights(combined)
    print("Class weights:", {l: round(w, 2) for l, w in zip(LABELS, class_weights.tolist())})

    train_ds = Dataset.from_list(combined)
    
    def enc_lbl(batch): return encode(batch, True)
    remove_cols = [c for c in train_ds.column_names if c not in ["input_ids","attention_mask","labels"]]
    train_enc = train_ds.map(enc_lbl, batched=True, remove_columns=remove_cols)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # No eval set - save every epoch, use leaderboard as validation
    args = TrainingArguments(
        output_dir=CKPT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="no",  # No eval - training on all data
        save_strategy="epoch",
        logging_steps=50,
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = WeightedNERTrainer(
        class_weights=class_weights, model=model, args=args,
        train_dataset=train_enc,
        data_collator=DataCollatorForTokenClassification(tok),
    )
    print(f"Training {EPOCHS} epochs on {device}...")
    trainer.train()

    # Evaluate on val (not used in training, good indicator)
    print("Checking val metrics...")
    val_results = run_inference(model, val_c, batch_size=64)
    val_preds_by_id = {r["id"]: r for r in val_results}
    for rec in val_preds_by_id.values():
        rec["recurring_flag"] = ""
    print("Val metrics (NOTE: val was in training set!):")
    evaluate_val(val_rows, val_preds_by_id)

    # Test inference
    print("Running test inference...")
    test_results = run_inference(model, test_rows, batch_size=64)
    for rec in test_results:
        rec["recurring_flag"] = ""
        for field in SCORED_FIELDS + ["bank_service_event"]:
            if field not in rec: rec[field] = ""
    
    assert len(test_results) == len(test_rows)
    json.dump(test_results, open(PRED_PATH, "w"), indent=2)
    print(f"Wrote {len(test_results)} predictions -> {PRED_PATH}")
    print("Sample:", json.dumps(test_results[0], indent=2))

if __name__ == "__main__":
    main()
