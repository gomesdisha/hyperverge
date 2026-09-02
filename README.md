# Transaction NER Pipeline (HyperVerge ML Assessment)

End-to-end information extraction pipeline for raw bank transaction descriptions using token classification.

## Overview
This repository fine-tunes `jhu-clsp/ettin-encoder-32m` (~32M parameters) to extract key financial transaction fields:
- `counterparty`: The party with which the transaction took place.
- `transaction_method`: The payment method (`ACH`, `POS`, `WEB`, `WIRE`, `CHECK`, etc.).
- `processor`: The transaction processor (`PAYPAL`, `SQUARE`, `STRIPE`, etc.).
- `recurring_flag`: Flag indicating recurring/subscription status.
- `bank_service_event`: Bank fees, overdrafts, withdrawals, etc.

---

## Results (Local Validation on 1,000 Transactions)

| Field | Precision | Recall | **F1 Score** |
| :--- | :---: | :---: | :---: |
| **`transaction_method`** | 0.8775 | 0.9167 | **0.8967** |
| **`counterparty`** | 0.8614 | 0.8543 | **0.8578** |
| **`processor`** | 0.7865 | 0.8580 | **0.8207** |
| **`recurring_flag`** | 0.0000 | 0.0000 | **0.0000** *(all annotators tagged 0 in training)* |
| **Active 3-Field Mean** | **0.8418** | **0.8763** | **0.8584** |
| **Overall Macro-F1** | — | — | **0.6438** |

---

## Key Strategies & Engineering Decisions
1. **Annotator Denoising & Consensus**:
   Each transaction was annotated by at least 2 out of 5 human annotators. Annotator `ann_8ac1bf` showed a ~94% disagreement rate. We merged multi-annotator tags by consensus (preferring non-`O` tags when one annotator omitted a field), creating clean ground truth.
2. **Handling `recurring_flag`**:
   Zero occurrences of `I-RECURRING_FLAG` exist across all 20,000 training rows and 2,000 validation rows. Predicting empty string avoids false-positive penalties and prevents clobbering payment method tokens (like `ACH`, `PPD`, `WEB`).
3. **Class-Weighted Cross-Entropy**:
   Weighted loss inversely proportional to tag frequencies prevents majority class (`O`) collapse on sparse classes like `PROCESSOR`.
4. **First-Subword Rule & Token Alignment**:
   Word-level entity reconstruction aligns subword predictions back to whitespace-tokenized transaction strings.

---

## File Structure
- `train_ner.py`: Initial training script with weighted loss and consensus labels.
- `train_resume.py`: Resume training script from checkpoints.
- `train_run2.py`: Combined train+val training script.
- `infer.py`: Standalone inference & evaluation script.
- `submit.py`: Official leaderboard API submission client.
- `predictions.json`: Test set predictions (10,000 transactions).
- `Copy_of_NLP_Pipeline_Students.ipynb`: Interactive notebook pipeline.

---

## Usage

### Run Inference
```bash
python infer.py ckpt_best/checkpoint-79
```

### Submit to Leaderboard
```bash
python submit.py predictions.json
```
