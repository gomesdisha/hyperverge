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
MAX_LEN   = 64
BATCH_SIZE = 64   # larger batch for CPU throughput
LR        = 3e-4
EPOCHS    = 5    # 5 epochs on CPU should be manageable
SEED      = 42
TRAIN_PATH = "data/train.jsonl"
VAL_PATH   = "data/val.jsonl"
TEST_PATH  = "data/test.jsonl"
CKPT_DIR   = "ckpt_best"
PRED_PATH  = "predictions.json"

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
torch.manual_seed(SEED); np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_consensus_dataset(rows):
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
        n = len(tags0)
        consensus = []
        for i in range(n):
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

def compute_class_weights(rows):
    label_counts = Counter()
    for r in rows:
        label_counts.update(r["ner_tags"])
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
                if w is None:
                    labels.append(-100)
                elif w != prev:
                    labels.append(label2id.get(tags[w], 0))
                else:
                    labels.append(-100)
                prev = w
            all_labels.append(labels)
        enc["labels"] = all_labels
    return enc

# Recurring flag rule patterns (token-level)
RECURRING_TOKEN_PATTERNS = [
    re.compile(r"^RECURRING$", re.I),
    re.compile(r"^RECUR$", re.I),
    re.compile(r"^AUTOPAY$", re.I),
    re.compile(r"^PREAUTH(ORIZED)?$", re.I),
    re.compile(r"^PREAUTHORIZED$", re.I),
    re.compile(r"^SUBSCRIPTION$", re.I),
    re.compile(r"^SCHEDULED$", re.I),
    re.compile(r"^PPD$"),
    re.compile(r"^CCD$"),
    re.compile(r"^ACH$"),
    re.compile(r"^WEB$"),
    re.compile(r"^DD$"),
    re.compile(r"^AUTO-?PAY$", re.I),
    re.compile(r"^AUTO-?DEBIT$", re.I),
    re.compile(r"^AUTORIZED$", re.I),
    re.compile(r"^DIRECTDEBIT$", re.I),
]

def apply_recurring_flag_rules(tokens, predicted_tags):
    tags = list(predicted_tags)
    for i, token in enumerate(tokens):
        for pat in RECURRING_TOKEN_PATTERNS:
            if pat.match(token):
                tags[i] = "I-RECURRING_FLAG"
                break
    return tags

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
def tag_words_batch(model, batch_tokens, apply_rules=True):
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
        if apply_rules:
            out = apply_recurring_flag_rules(tokens, out)
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

def run_inference(model, records, batch_size=64, apply_rules=True):
    model.eval()
    results = []
    for i in range(0, len(records), batch_size):
        chunk = records[i: i + batch_size]
        tags_batch = tag_words_batch(model, [r["tokens"] for r in chunk], apply_rules)
        for r, tags in zip(chunk, tags_batch):
            fields = extract_fields(r["tokens"], tags)
            results.append({"id": r["id"], **fields})
    return results

def evaluate_predictions(val_rows, predictions_by_id):
    consensus_val = build_consensus_dataset(val_rows)
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
    print("Loading data...")
    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    test_rows = load_jsonl(TEST_PATH)
    print(f"  Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")

    train_consensus = build_consensus_dataset(train_rows)
    val_consensus = build_consensus_dataset(val_rows)
    class_weights = compute_class_weights(train_consensus)
    print("Class weights:", {l: round(w, 2) for l, w in zip(LABELS, class_weights.tolist())})

    train_ds = Dataset.from_list(train_consensus)
    val_ds = Dataset.from_list(val_consensus)

    def enc_lbl(batch):
        return encode(batch, True)

    remove_train = [c for c in train_ds.column_names if c not in ["input_ids","attention_mask","labels"]]
    remove_val = [c for c in val_ds.column_names if c not in ["input_ids","attention_mask","labels"]]
    train_enc = train_ds.map(enc_lbl, batched=True, remove_columns=remove_train)
    val_enc = val_ds.map(enc_lbl, batched=True, remove_columns=remove_val)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    args = TrainingArguments(
        output_dir=CKPT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        warmup_steps=200,   # ~10% warmup; warmup_ratio not in transformers 5.x
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = WeightedNERTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        data_collator=DataCollatorForTokenClassification(tok),
    )
    print(f"Training {EPOCHS} epochs on {device}...")
    trainer.train()

    print("Running val inference...")
    val_results = run_inference(model, val_consensus, batch_size=64)
    val_preds_by_id = {r["id"]: r for r in val_results}
    print("Local validation metrics:")
    evaluate_predictions(val_rows, val_preds_by_id)

    print("Running test inference...")
    test_results = run_inference(model, test_rows, batch_size=64)
    assert len(test_results) == len(test_rows), f"Missing predictions"
    for rec in test_results:
        for field in SCORED_FIELDS + ["bank_service_event"]:
            if field not in rec:
                rec[field] = ""
    json.dump(test_results, open(PRED_PATH, "w"), indent=2)
    print(f"Wrote {len(test_results)} predictions -> {PRED_PATH}")
    print("Sample:", json.dumps(test_results[0], indent=2))

if __name__ == "__main__":
    main()
