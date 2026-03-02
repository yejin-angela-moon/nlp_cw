"""
Knowledge distillation for PCL binary classification: teacher and student models,
dataset, and training entrypoint.
"""

from .dataset import (
    PCLBinaryDataset,
    build_dataloaders,
    load_pcl_splits,
)
from .student import get_model_and_tokenizer as get_student_model_and_tokenizer
from .teacher import get_model_and_tokenizer as get_teacher_model_and_tokenizer
from .teacher import train_teacher

__all__ = [
    "PCLBinaryDataset",
    "build_dataloaders",
    "load_pcl_splits",
    "get_student_model_and_tokenizer",
    "get_teacher_model_and_tokenizer",
    "train_teacher",
]
