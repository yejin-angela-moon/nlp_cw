"""
Student model for PCL binary classification: smaller encoder (e.g. DistilBERT)
with sequence classification head. Forward returns logits of shape (batch_size, 2).
"""

from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_model_and_tokenizer(
    student_name: str,
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load student model and tokenizer for binary sequence classification.
    Returns (model, tokenizer). Model is on device if provided.
    """
    tokenizer = AutoTokenizer.from_pretrained(student_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_name,
        num_labels=num_labels,
    )
    if device is not None:
        model = model.to(device)
    return model, tokenizer
