"""Prediction metrics retained for external-data evaluation."""

import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def entropy(distribution: np.ndarray) -> float:
    nz = distribution[distribution > 0]
    if nz.size == 0:
        return 0.0
    return float(-(nz * np.log(nz)).sum())


def classification_metrics(
    confusion: np.ndarray,
    prob_sum: np.ndarray,
    total_loss: float,
    mean_max_prob_sum: float,
) -> dict[str, object]:
    total = int(confusion.sum())
    num_classes = int(confusion.shape[0])
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.divide(true_positive, support, out=np.zeros_like(true_positive, dtype=float), where=support > 0)
        precisions = np.divide(true_positive, predicted, out=np.zeros_like(true_positive, dtype=float), where=predicted > 0)
        f1 = np.divide(
            2.0 * precisions * recalls,
            precisions + recalls,
            out=np.zeros_like(recalls, dtype=float),
            where=(precisions + recalls) > 0,
        )
    valid = support > 0
    num_eval_classes = int(valid.sum())
    pred_total = max(int(predicted.sum()), 1)
    true_total = max(int(support.sum()), 1)
    pred_distribution = predicted.astype(float) / float(pred_total)
    true_distribution = support.astype(float) / float(true_total)
    prob_distribution = prob_sum.astype(float) / max(float(prob_sum.sum()), 1.0)
    pred_entropy = entropy(pred_distribution)
    prob_entropy = entropy(prob_distribution)
    true_majority_ratio = float(true_distribution.max()) if true_distribution.size else 0.0
    pred_majority_ratio = float(pred_distribution.max()) if pred_distribution.size else 0.0
    pred_true_tv = float(0.5 * np.abs(pred_distribution - true_distribution).sum()) if pred_distribution.size else 0.0
    prob_true_tv = float(0.5 * np.abs(prob_distribution - true_distribution).sum()) if prob_distribution.size else 0.0
    return {
        "accuracy": float(true_positive.sum() / max(total, 1)),
        "balanced_accuracy": float(recalls[valid].mean()) if valid.any() else 0.0,
        "macro_f1": float(f1[valid].mean()) if valid.any() else 0.0,
        "mean_precision": float(precisions[valid].mean()) if valid.any() else 0.0,
        "loss": float(total_loss / max(total, 1)),
        "num_samples": int(total),
        "num_classes": int(num_classes),
        "num_total_classes": int(num_classes),
        "num_eval_classes": num_eval_classes,
        "true_majority_class": int(support.argmax()) if support.size else -1,
        "true_majority_ratio": true_majority_ratio,
        "pred_majority_class": int(predicted.argmax()) if predicted.size else -1,
        "collapse_ratio": pred_majority_ratio,
        "collapse_over_true_majority": float(pred_majority_ratio / max(true_majority_ratio, 1e-12)),
        "collapse_excess": float(pred_majority_ratio - true_majority_ratio),
        "per_class_recall": " ".join(f"{x:.6f}" for x in recalls.tolist()),
        "per_class_precision": " ".join(f"{x:.6f}" for x in precisions.tolist()),
        "per_class_f1": " ".join(f"{x:.6f}" for x in f1.tolist()),
        "class_support": " ".join(str(int(x)) for x in support.tolist()),
        "true_counts": " ".join(str(int(x)) for x in support.tolist()),
        "predicted_class_counts": " ".join(str(int(x)) for x in predicted.tolist()),
        "majority_prediction_ratio": pred_majority_ratio,
        "effective_predicted_classes": float(np.exp(pred_entropy)),
        "pred_nonzero_classes": int((predicted > 0).sum()),
        "prediction_entropy": pred_entropy,
        "pred_entropy_norm": float(pred_entropy / math.log(num_classes)) if num_classes > 1 else 0.0,
        "prob_entropy_norm": float(prob_entropy / math.log(num_classes)) if num_classes > 1 else 0.0,
        "mean_max_probability": float(mean_max_prob_sum / max(total, 1)),
        "pred_true_tv": pred_true_tv,
        "prob_true_tv": prob_true_tv,
        "prob_mean": " ".join(f"{float(x):.6f}" for x in prob_distribution.tolist()),
        "confusion_matrix_json": json.dumps(confusion.tolist(), separators=(",", ":")),
    }


def evaluate_checkpoint_predictions(meta: dict[str, object], checkpoint: dict[str, object], cfg: dict[str, object]) -> dict[str, object]:
    from utils.runtime import build_runtime

    device = torch.device(str(cfg.get("device", "cpu")))
    runtime = build_runtime(
        meta=meta,
        data_root=str(cfg["data_root"]),
        split=str(cfg.get("split", "test")),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=int(cfg.get("num_workers", 4)),
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    loader = runtime["loader"]
    num_classes = int(meta.get("num_classes") or len(meta.get("class_names", [])))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    prob_sum = np.zeros(num_classes, dtype=np.float64)
    total_loss = 0.0
    mean_max_prob_sum = 0.0
    amp_enabled = bool(cfg.get("amp", False)) and device.type == "cuda"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = forward_fn(model, x)
                    loss = F.cross_entropy(logits, y, reduction="sum")
            else:
                logits = forward_fn(model, x)
                loss = F.cross_entropy(logits, y, reduction="sum")
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu())
            prob_sum += probs.detach().cpu().double().sum(dim=0).numpy()
            mean_max_prob_sum += float(probs.max(dim=1).values.detach().cpu().sum())
            y_np = y.detach().cpu().numpy().astype(int)
            p_np = preds.detach().cpu().numpy().astype(int)
            for target, pred in zip(y_np, p_np):
                if 0 <= target < num_classes and 0 <= pred < num_classes:
                    confusion[target, pred] += 1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return classification_metrics(confusion, prob_sum, total_loss, mean_max_prob_sum)
