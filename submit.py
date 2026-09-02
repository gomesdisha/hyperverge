"""
Submit predictions.json to the leaderboard API.
Usage: python submit.py predictions.json
"""
import sys
import json
import requests

BASE_URL = "http://3.6.116.106:8990"

# Fill these in!
NAME = "Disha"
EMAIL = "dishagomes2005@gmail.com"
ROLL_NUMBER = "230953010"
COLLEGE_NAME = "Manipal Institute of Technology"


def _print_error(resp):
    print(f"Request failed: {resp.status_code} {resp.reason}")
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        print(resp.text)
        return
    if isinstance(detail, dict):
        header = detail.get("error", "Validation failed")
        print(header)
        problems = detail.get("problems", [])
        if isinstance(problems, list):
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"  {problems}")
        for k, v in detail.items():
            if k not in ("error", "problems"):
                print(f"  {k}: {v}")
    else:
        print(detail)


def submit_predictions(pred_path, name, email, roll_number, college):
    url = f"{BASE_URL}/submit"
    with open(pred_path, "rb") as f:
        files = {"predictions": (pred_path, f, "application/json")}
        data = {
            "name": name,
            "email": email,
            "roll_number": roll_number,
            "college": college,
        }
        resp = requests.post(url, data=data, files=files, timeout=120)
    if not resp.ok:
        _print_error(resp)
        return {}
    return resp.json()


def print_results(result):
    if not result:
        return
    per_field = result["per_field"]
    headers = ["field", "metric", "f1", "precision", "recall", "exact", "support"]
    rows = []
    for field, m in per_field.items():
        rows.append([
            field, m["metric"],
            f'{m["f1"]:.4f}', f'{m["precision"]:.4f}', f'{m["recall"]:.4f}',
            f'{m["exact"]:.4f}', str(m["support"]),
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print()
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))
    print()
    print(f'This submission macro-F1 : {result["this_submission_macro_f1"]:.4f}')
    print(f'Best macro-F1 so far     : {result["best_macro_f1"]:.4f}  (at {result["best_at"]})')
    print(f'Attempts so far          : {result["attempts"]}')
    print(f'Attempts remaining       : {result["attempts_remaining"]}')


def validate_predictions(pred_path):
    """Validate predictions.json format before submitting."""
    preds = json.load(open(pred_path))
    print(f"Loaded {len(preds)} predictions")
    required_fields = ["id", "counterparty", "transaction_method", "processor", "recurring_flag"]
    for i, rec in enumerate(preds[:5]):
        for field in required_fields:
            if field not in rec:
                print(f"WARNING: Missing field '{field}' in record {i}")
    ids = [r["id"] for r in preds]
    if len(ids) != len(set(ids)):
        print("ERROR: Duplicate IDs in predictions!")
    else:
        print(f"All {len(ids)} IDs are unique - OK")
    print("Sample:", json.dumps(preds[0], indent=2))
    return preds


if __name__ == "__main__":
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "predictions.json"
    print(f"Validating {pred_path}...")
    validate_predictions(pred_path)
    print(f"\nSubmitting to leaderboard...")
    result = submit_predictions(pred_path, NAME, EMAIL, ROLL_NUMBER, COLLEGE_NAME)
    print_results(result)
