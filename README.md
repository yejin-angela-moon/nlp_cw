# NLP Coursework — PCL Binary Classification

Binary classification of **Patronising and Condescending Language (PCL)** using a fine-tuned `roberta-base` model, submitted for the SemEval-style shared task.

## Task

Given a short news paragraph, predict:
- **0** — non-PCL
- **1** — contains PCL

## Dataset

| File | Description |
|------|-------------|
| `data/dontpatronizeme_pcl.tsv` | Full corpus with paragraph text |
| `data/train_semeval_parids-labels.csv` | Train split with 7-annotator label vectors |
| `data/dev_semeval_parids-labels.csv` | Dev split with 7-annotator label vectors |
| `data/task4_test.tsv` | Unlabelled test set |

Labels are binarised: all-zeros → 0 (non-PCL); any non-zero → 1 (PCL).
The training set is heavily imbalanced (~9.5:1, non-PCL:PCL).

## Model & Training

- **Base model:** `roberta-base` via HuggingFace Transformers
- **Sequence length:** 256 tokens
- **Batch size:** 16 with `WeightedRandomSampler` to handle class imbalance
- **Optimiser:** AdamW (`lr=1e-5`, `weight_decay=0.01`)
- **Scheduler:** Linear warmup (10%) + linear decay
- **Early stopping:** patience = 2 (monitored on dev binary F1)

## Results (Dev Set)

| Epoch | Train Loss | Accuracy | F1 (PCL) | F1 (Macro) |
|-------|------------|----------|-----------|------------|
| 6     | 0.0507     | 0.9298   | **0.6370**| 0.7991     |

Best checkpoint saved to `BestModel/best_roberta_pcl.pt`.

## Project Structure

```
.
├── BestModel/
│   ├── pcl_classification.ipynb   # Main training notebook
│   └── best_roberta_pcl.pt        # Best model checkpoint
├── data/                          # Raw data files
├── src/
│   └── exploratory_data_analysis/ # EDA scripts and outputs
├── dev.txt                        # Dev set predictions
├── test.txt                       # Test set predictions
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Then open and run `BestModel/pcl_classification.ipynb` top-to-bottom.
A CUDA-capable GPU is recommended (training takes ~8 epochs).

## Predictions

- `dev.txt` — one prediction per line for the dev split (2,093 lines)
- `test.txt` — one prediction per line for the test split (3,832 lines)
