"""
Teacher model for PCL binary classification: HuggingFace encoder with
sequence classification head. Provides get_model_and_tokenizer and optional
train_teacher (CE loss, AdamW, save best checkpoint).
"""

from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_model_and_tokenizer(
    teacher_name: str,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load teacher model and tokenizer for binary sequence classification.
    Returns (model, tokenizer). Model is on device if provided.
    """
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_name,
        num_labels=num_labels,
    )
    if device is not None:
        model = model.to(device)
    return model, tokenizer


def _accuracy_and_f1(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Compute accuracy and macro F1 (binary: (F1_0 + F1_1) / 2)."""
    preds = preds.cpu()
    labels = labels.cpu()
    acc = (preds == labels).float().mean().item()
    # Binary F1: TP, FP, FN for class 1
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    # Class 0
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn0 = ((preds == 1) & (labels == 0)).sum().item()
    fp0 = ((preds == 0) & (labels == 1)).sum().item()
    prec0 = tn / (tn + fp0) if (tn + fp0) > 0 else 0.0
    rec0 = tn / (tn + fn0) if (tn + fn0) > 0 else 0.0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
    f1 = (f1_0 + f1_1) / 2.0
    return acc, f1


def train_teacher(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 2e-5,
) -> dict[str, Any]:
    """
    Train teacher with cross-entropy on hard labels. Saves best checkpoint by dev F1.
    Returns dict with train/dev losses and metrics per epoch.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    best_f1 = -1.0
    history: dict[str, list] = {"train_loss": [], "dev_loss": [], "dev_acc": [], "dev_f1": []}

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        dev_loss_sum = 0.0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                dev_loss_sum += outputs.loss.item()
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                all_preds.append(preds)
                all_labels.append(labels)
        avg_dev_loss = dev_loss_sum / len(dev_loader)
        history["dev_loss"].append(avg_dev_loss)
        preds_cat = torch.cat(all_preds, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        dev_acc, dev_f1 = _accuracy_and_f1(preds_cat, labels_cat)
        history["dev_acc"].append(dev_acc)
        history["dev_f1"].append(dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    return history
