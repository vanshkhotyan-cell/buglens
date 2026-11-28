from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from buglens.modeling.fusion import (
    ALL_CLASSES,
    CLASS_TO_IDX,
    apply_temperature_scaling,
    compute_ece,
    fit_pooling_weights,
    fit_temperature,
    softmax,
    sweep_teach_threshold,
)


def format_deberta_sequence(norm_q: str, norm_corr: str, norm_stud: str) -> str:
    """Step 2: Strictly serialize student attempt without leaking symbolic replay outputs."""
    return f"[CLS] Question: {norm_q.strip()} [SEP] Correct: {norm_corr.strip()} [SEP] Student: {norm_stud.strip()} [SEP]"


def apply_bug_plus_slip_perturbation(expr: str, rng: np.random.Generator) -> str:
    """Step 2: Apply a +/-1 arithmetic slip or dropped coefficient to ~15% of synthetic items."""
    tokens = list(re.finditer(r"\d+", expr))
    if not tokens:
        return expr

    choice = int(rng.integers(0, 2))
    if choice == 0 or len(tokens) == 1:
        match = tokens[int(rng.integers(0, len(tokens)))]
        val = int(match.group(0))
        delta = 1 if (val <= 1 or rng.random() < 0.5) else -1
        new_val = str(max(1, val + delta))
        return expr[: match.start()] + new_val + expr[match.end() :]

    coeff_match = re.search(r"(?<![0-9a-zA-Z])([2-9])([a-zA-Z])", expr)
    if coeff_match:
        return expr[: coeff_match.start(1)] + expr[coeff_match.end(1) :]

    match = tokens[0]
    val = int(match.group(0))
    return expr[: match.start()] + str(val + 1) + expr[match.end() :]


def augment_with_bug_plus_slip(
    df: pd.DataFrame,
    slip_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Step 2: Create ~15% noisy bug-plus-slip training rows while preserving true bug_id."""
    rng = np.random.default_rng(seed)
    eligible = df[df["bug_id"].isin(ALL_CLASSES[:5])].copy()
    if len(eligible) == 0:
        return df.copy()

    n_aug = max(1, int(round(len(eligible) * slip_fraction)))
    sampled = eligible.sample(n=min(n_aug, len(eligible)), random_state=seed).copy()
    sampled["StudentAnswerText"] = (
        sampled["StudentAnswerText"]
        .astype(str)
        .apply(lambda s: apply_bug_plus_slip_perturbation(s, rng))
    )
    sampled["source"] = "synthetic_bug_plus_slip"
    return pd.concat([df, sampled], ignore_index=True)


def compute_class_weights(labels: np.ndarray, num_classes: int = len(ALL_CLASSES)) -> np.ndarray:
    """Inverse-frequency class weights for imbalanced 6-class Cross-Entropy."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return weights / np.mean(weights)


def prepare_training_records(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Serialize DataFrame rows into DeBERTa input strings and integer labels [0..5]."""
    texts = [
        format_deberta_sequence(
            str(r["QuestionText"]),
            str(r["CorrectAnswerText"]),
            str(r["StudentAnswerText"]),
        )
        for _, r in df.iterrows()
    ]
    labels = np.array([CLASS_TO_IDX[str(b)] for b in df["bug_id"]], dtype=int)
    return texts, labels


@dataclass
class CalibrationArtifact:
    """Frozen post-training calibration & fusion parameters saved to models/calibration.json."""

    temperature: float
    alphas: tuple[float, float, float]
    tau_teach: float
    cal_ece_before: float
    cal_ece_after: float

    def save(self, path: Path = Path("models/calibration.json")) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "temperature": round(self.temperature, 6),
            "alphas": [round(a, 6) for a in self.alphas],
            "tau_teach": round(self.tau_teach, 4),
            "cal_ece_before": round(self.cal_ece_before, 4),
            "cal_ece_after": round(self.cal_ece_after, 4),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fit_post_training_calibration(
    cal_logits: np.ndarray,
    cal_p_sym: np.ndarray,
    cal_labels: np.ndarray,
    target_precision: float = 0.90,
) -> CalibrationArtifact:
    """Fit T (ECE <= 0.08), tune (alpha_0, alpha_1, alpha_2), and set tau_teach on cal."""
    raw_probs = softmax(cal_logits)
    ece_before = compute_ece(raw_probs, cal_labels)

    t_opt = fit_temperature(cal_logits, cal_labels)
    cal_p_nn = apply_temperature_scaling(cal_logits, t_opt)
    ece_after = compute_ece(cal_p_nn, cal_labels)

    pi_0_cal = np.full_like(cal_p_nn, 1.0 / len(ALL_CLASSES))
    alphas = fit_pooling_weights(pi_0_cal, cal_p_sym, cal_p_nn, cal_labels)

    delta = 1e-6
    log_scores = (
        alphas[0] * np.log(pi_0_cal + delta)
        + alphas[1] * np.log(cal_p_sym + delta)
        + alphas[2] * np.log(cal_p_nn + delta)
    )
    fused_cal = softmax(log_scores)
    tau_teach = sweep_teach_threshold(fused_cal, cal_labels, target_precision=target_precision)

    return CalibrationArtifact(
        temperature=t_opt,
        alphas=alphas,
        tau_teach=tau_teach,
        cal_ece_before=ece_before,
        cal_ece_after=ece_after,
    )


def train_deberta_classifier(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    model_name: str = "microsoft/deberta-v3-small",
    output_dir: Path = Path("models/deberta-v3-small"),
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2.5e-5,
    max_length: int = 192,
    seed: int = 42,
) -> dict[str, float]:
    """Fine-tune microsoft/deberta-v3-small with class-weighted Cross-Entropy and export ONNX."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug_train_df = augment_with_bug_plus_slip(train_df, slip_fraction=0.15, seed=seed)
    train_texts, train_labels = prepare_training_records(aug_train_df)
    cal_texts, cal_labels = prepare_training_records(cal_df)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class BugDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray) -> None:
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_loader = DataLoader(
        BugDataset(train_texts, train_labels), batch_size=batch_size, shuffle=True
    )
    cal_loader = DataLoader(BugDataset(cal_texts, cal_labels), batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ALL_CLASSES)
    )
    model.to(device)

    weights_tensor = torch.tensor(
        compute_class_weights(train_labels), dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss={total_loss / max(1, len(train_loader)):.4f}"
        )

    model.eval()
    cal_logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in cal_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            cal_logits_list.append(logits.cpu().numpy())

    cal_logits = np.vstack(cal_logits_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    np.save(output_dir / "cal_logits.npy", cal_logits)

    cpu_model = model.to("cpu").eval()
    dummy = tokenizer(
        ["[CLS] Question: 1/2 + 1/3 [SEP] Correct: 5/6 [SEP] Student: 2/5 [SEP]"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        cpu_model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    return {"cal_samples": float(len(cal_labels))}
