#!/usr/bin/env python3
"""Small PyTorch heads used by the episode task suite neural baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeuralConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    batch_size: int
    dropout: float
    device: str
    seed: int


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --include-neural. Install requirements-omni.txt or add torch to the environment."
        ) from exc
    return torch, nn, F


def _resolve_device(torch, device_spec: str):
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_spec)


def _standardize(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[train_idx].mean(axis=0).astype(np.float32)
    std = X[train_idx].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((X - mean) / std).astype(np.float32), mean, std


def _batch_indices(torch, train_idx: np.ndarray, batch_size: int, seed: int, epoch: int):
    gen = torch.Generator()
    gen.manual_seed(seed + epoch)
    idx = torch.from_numpy(train_idx.astype(np.int64))
    return idx[torch.randperm(len(idx), generator=gen)]


def _history_epoch(epoch: int, epochs: int) -> bool:
    report_every = max(1, epochs // 5)
    return epoch == 1 or epoch == epochs or epoch % report_every == 0


def _make_mlp(nn, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_classes: int,
    config: NeuralConfig,
    use_class_weights: bool = True,
) -> dict:
    torch, nn, F = _import_torch()
    device = _resolve_device(torch, config.device)
    torch.manual_seed(config.seed)

    Xs, mean, std = _standardize(X.astype(np.float32), train_idx)
    x_tensor = torch.from_numpy(Xs)
    y_tensor = torch.from_numpy(y.astype(np.int64))
    model = _make_mlp(nn, X.shape[1], n_classes, config.hidden_dim, config.dropout).to(device)

    class_weights = None
    if use_class_weights:
        counts = np.bincount(y[train_idx], minlength=n_classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0) / max(n_classes, 1)
        weights = weights / max(float(weights.mean()), 1e-6)
        class_weights = torch.from_numpy(weights.astype(np.float32)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = _batch_indices(torch, train_idx, config.batch_size, config.seed, epoch)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, len(perm), config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
            total_seen += len(idx)
        if _history_epoch(epoch, config.epochs):
            history.append({
                "epoch": epoch,
                "loss": total_loss / max(total_seen, 1),
                "train_accuracy": total_correct / max(total_seen, 1),
            })

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor[test_idx].to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
    return {
        "pred": np.argmax(probs, axis=1).astype(np.int64),
        "prob": probs,
        "history": history,
        "mean": mean,
        "std": std,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "device": str(device),
    }


def train_multilabel(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: NeuralConfig,
) -> dict:
    torch, nn, F = _import_torch()
    device = _resolve_device(torch, config.device)
    torch.manual_seed(config.seed)

    Xs, mean, std = _standardize(X.astype(np.float32), train_idx)
    x_tensor = torch.from_numpy(Xs)
    y_tensor = torch.from_numpy(Y.astype(np.float32))
    model = _make_mlp(nn, X.shape[1], Y.shape[1], config.hidden_dim, config.dropout).to(device)

    counts = Y[train_idx].sum(axis=0).astype(np.float32)
    neg = len(train_idx) - counts
    pos_weight = np.clip(neg / np.maximum(counts, 1.0), 1.0, 20.0)
    pos_weight_tensor = torch.from_numpy(pos_weight.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = _batch_indices(torch, train_idx, config.batch_size, config.seed, epoch)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(perm), config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight_tensor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_seen += len(idx)
        if _history_epoch(epoch, config.epochs):
            history.append({"epoch": epoch, "loss": total_loss / max(total_seen, 1)})

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_tensor[test_idx].to(device))).cpu().numpy().astype(np.float32)
    return {
        "prob": prob,
        "pred": (prob >= 0.5).astype(np.float32),
        "history": history,
        "mean": mean,
        "std": std,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "device": str(device),
    }


def train_regressor(
    X: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: NeuralConfig,
) -> dict:
    torch, nn, F = _import_torch()
    device = _resolve_device(torch, config.device)
    torch.manual_seed(config.seed)

    Xs, x_mean, x_std = _standardize(X.astype(np.float32), train_idx)
    y_mean = Y[train_idx].mean(axis=0).astype(np.float32)
    y_std = Y[train_idx].std(axis=0).astype(np.float32)
    y_std = np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)
    Ys = ((Y - y_mean) / y_std).astype(np.float32)

    x_tensor = torch.from_numpy(Xs)
    y_tensor = torch.from_numpy(Ys)
    model = _make_mlp(nn, X.shape[1], Y.shape[1], config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        perm = _batch_indices(torch, train_idx, config.batch_size, config.seed, epoch)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(perm), config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = x_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_seen += len(idx)
        if _history_epoch(epoch, config.epochs):
            history.append({"epoch": epoch, "loss": total_loss / max(total_seen, 1)})

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor[test_idx].to(device)).cpu().numpy().astype(np.float32)
    pred = pred_scaled * y_std + y_mean
    return {
        "pred": pred.astype(np.float32),
        "history": history,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "device": str(device),
    }


def save_torch_model(path, payload: dict) -> None:
    torch, _nn, _F = _import_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
