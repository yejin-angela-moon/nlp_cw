"""
Training entrypoint: (1) train teacher with CE and save; (2) run distillation
to train student (soft + hard loss) and save best student.
"""

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from .dataset import build_dataloaders
from .student import get_model_and_tokenizer as get_student
from .teacher import get_model_and_tokenizer as get_teacher
from .teacher import train_teacher
from .teacher import _accuracy_and_f1


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Combined loss: alpha * hard CE + (1 - alpha) * soft KL.
    Soft: KL(softmax(teacher/T), softmax(student/T)) * T^2.
    """
    T = temperature
    hard_loss = torch.nn.functional.cross_entropy(student_logits, labels)
    soft_teacher = torch.softmax(teacher_logits / T, dim=-1)
    log_soft_student = torch.log_softmax(student_logits / T, dim=-1)
    soft_loss = torch.nn.functional.kl_div(
        log_soft_student,
        soft_teacher,
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def run_distillation(
    teacher_path: Path,
    student_model_name: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    temperature: float = 2.0,
    alpha: float = 0.5,
    epochs: int = 3,
    lr: float = 2e-5,
) -> dict:
    """
    Load pretrained teacher from teacher_path, load student, train student with
    distillation loss. Save best student by dev F1. Uses teacher's tokenizer
    (loaded with teacher from teacher_path) for consistency.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path)
    teacher = teacher.to(device)
    teacher.eval()

    student, _ = get_student(student_model_name, num_labels=2, device=device)
    optimizer = AdamW(student.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    best_f1 = -1.0
    history = {"train_loss": [], "dev_loss": [], "dev_acc": [], "dev_f1": []}

    for epoch in range(epochs):
        student.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits
            student_out = student(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits
            loss = _distillation_loss(
                student_logits, teacher_logits, labels, temperature, alpha
            )
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train = train_loss_sum / len(train_loader)
        history["train_loss"].append(avg_train)

        student.eval()
        dev_loss_sum = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                student_out = student(input_ids=input_ids, attention_mask=attention_mask)
                loss = _distillation_loss(
                    student_out.logits, teacher_out.logits, labels, temperature, alpha
                )
                dev_loss_sum += loss.item()
                preds = student_out.logits.argmax(dim=-1)
                all_preds.append(preds)
                all_labels.append(labels)
        avg_dev = dev_loss_sum / len(dev_loader)
        history["dev_loss"].append(avg_dev)
        preds_cat = torch.cat(all_preds, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)
        dev_acc, dev_f1 = _accuracy_and_f1(preds_cat, labels_cat)
        history["dev_acc"].append(dev_acc)
        history["dev_f1"].append(dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            out = output_dir / "student_distilled"
            out.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(out)
            tokenizer.save_pretrained(out)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teacher then distill to student (PCL binary).")
    parser.add_argument("--data_dir", type=Path, default=None, help="Data directory (default: project data/)")
    parser.add_argument("--teacher_model", type=str, default="bert-base-uncased", help="Teacher HuggingFace model name")
    parser.add_argument("--student_model", type=str, default="distilbert-base-uncased", help="Student HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for both teacher and student")
    parser.add_argument("--T", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for hard CE (1-alpha for soft KL)")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Teacher and student checkpoints")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda or cpu")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 1) Teacher model + tokenizer, build dataloaders with teacher tokenizer
    teacher, tokenizer = get_teacher(args.teacher_model, num_labels=2, device=device)
    train_loader, dev_loader = build_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # 2) Train teacher and save
    teacher_dir = args.output_dir / "teacher"
    train_teacher(
        model=teacher,
        tokenizer=tokenizer,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        output_dir=teacher_dir,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"Teacher saved to {teacher_dir}")

    # 3) Distillation: load teacher from checkpoint, train student (same dataloaders)
    run_distillation(
        teacher_path=teacher_dir,
        student_model_name=args.student_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        output_dir=args.output_dir,
        temperature=args.T,
        alpha=args.alpha,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"Student saved to {args.output_dir / 'student_distilled'}")


if __name__ == "__main__":
    main()
