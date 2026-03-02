"""
Dataset for PCL binary classification: load SemEval labels + Don't Patronize Me TSV,
join to get (text, binary_label), and expose a PyTorch Dataset.
"""

import ast
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def load_pcl_splits(
    data_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and dev splits with text and binary PCL labels.
    Returns (train_df, dev_df) each with columns: text, label (0=non-PCL, 1=PCL).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"

    train = pd.read_csv(data_dir / "train_semeval_parids-labels.csv")
    dev = pd.read_csv(data_dir / "dev_semeval_parids-labels.csv")
    tsv = pd.read_csv(
        data_dir / "dontpatronizeme_pcl.tsv",
        sep="\t",
        skiprows=4,
        names=["id", "art_id", "keyword", "country", "text", "label_tsv"],
    )
    tsv["id"] = tsv["id"].astype(int)

    def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
        label_col = [c for c in df.columns if "label" in c.lower()][0]
        df = df.copy()
        df["label_vec"] = df[label_col].apply(ast.literal_eval)
        df["is_pcl"] = df["label_vec"].apply(lambda v: sum(v) > 0)
        df["binary_label"] = df["is_pcl"].astype(int)
        return df

    train = add_binary_label(train)
    dev = add_binary_label(dev)

    train = train.merge(tsv[["id", "text"]], left_on="par_id", right_on="id", how="inner").drop(columns=["id"], errors="ignore")
    dev = dev.merge(tsv[["id", "text"]], left_on="par_id", right_on="id", how="inner").drop(columns=["id"], errors="ignore")

    train = train[["text", "binary_label"]].rename(columns={"binary_label": "label"})
    dev = dev[["text", "binary_label"]].rename(columns={"binary_label": "label"})
    return train, dev


class PCLBinaryDataset(Dataset):
    """
    PyTorch Dataset yielding tokenized input_ids, attention_mask, and label (0/1)
    for PCL binary classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = int(row["label"])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_dataloaders(
    data_dir: Optional[Path] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 256,
    batch_size: int = 16,
):
    """
    Build train and dev DataLoaders. Requires tokenizer to be provided
    (e.g. from teacher or student get_model_and_tokenizer).
    """
    from torch.utils.data import DataLoader

    if tokenizer is None:
        raise ValueError("tokenizer is required to build dataloaders")
    train_df, dev_df = load_pcl_splits(data_dir=data_dir)
    train_ds = PCLBinaryDataset(train_df, tokenizer, max_length=max_length)
    dev_ds = PCLBinaryDataset(dev_df, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, dev_loader
