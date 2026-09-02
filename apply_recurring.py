import json
import re

TEST_PATH = "data/test.jsonl"
PRED_PATH = "predictions.json"
OUT_PATH = "predictions_v2.json"

# Exact keywords from annotator instruction #4:
# "recurring / pre authorized / subscription / auto debit / auto pay / auto approved"
RECURRING_PATTERNS = [
    re.compile(r"\b(subscription|subscr)\b", re.IGNORECASE),
    re.compile(r"\b(recurring)\b", re.IGNORECASE),
    re.compile(r"\b(auto[\s-]?pay)\b", re.IGNORECASE),
    re.compile(r"\b(auto[\s-]?debit)\b", re.IGNORECASE),
    re.compile(r"\b(pre[\s-]?auth\w*)\b", re.IGNORECASE),
    re.compile(r"\b(auto[\s-]?approv\w*)\b", re.IGNORECASE),
]

def main():
    test_rows = [json.loads(line) for line in open(TEST_PATH, encoding="utf-8")]
    test_map = {r["id"]: r for r in test_rows}
    
    preds = json.load(open(PRED_PATH, encoding="utf-8"))
    
    flagged_count = 0
    cleaned_preds = []
    
    for pred in preds:
        rec = dict(pred)
        rid = rec["id"]
        raw_tokens = test_map[rid]["tokens"]
        text = " ".join(raw_tokens)
        
        # Check recurring patterns
        matched_words = []
        for token in raw_tokens:
            for pat in RECURRING_PATTERNS:
                if pat.search(token):
                    matched_words.append(token)
                    break
                    
        if matched_words:
            rec["recurring_flag"] = " ".join(matched_words)
            flagged_count += 1
        else:
            rec["recurring_flag"] = ""
            
        # Ensure only official fields are kept cleanly
        cleaned_rec = {
            "id": rec["id"],
            "counterparty": rec.get("counterparty", ""),
            "transaction_method": rec.get("transaction_method", ""),
            "processor": rec.get("processor", ""),
            "recurring_flag": rec.get("recurring_flag", ""),
            "bank_service_event": rec.get("bank_service_event", "")
        }
        cleaned_preds.append(cleaned_rec)
        
    print(f"Total test transactions: {len(cleaned_preds)}")
    print(f"Transactions flagged as recurring: {flagged_count}")
    
    json.dump(cleaned_preds, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    # Also update predictions.json
    json.dump(cleaned_preds, open(PRED_PATH, "w", encoding="utf-8"), indent=2)
    print(f"Wrote updated predictions to {OUT_PATH} and {PRED_PATH}")

if __name__ == "__main__":
    main()
