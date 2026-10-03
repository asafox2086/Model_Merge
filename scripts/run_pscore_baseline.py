#!/usr/bin/env python3
"""Run supervised Pscore-MLP fusion on the existing shared-backbone benchmark."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dataset.medmnist_npz import NpzTensorDataset
from exp_analyze.collect_prediction_diagnostics import classification_metrics, macro_ovr_auc
from methods.head_only import classifier_param_keys
from methods.pscore import ProbabilityFusion, client_probabilities, fit_fusion
from model import build_model
from utils.reference_cache import get_reference_bundle_path
from utils.runtime import _build_small_transform
from utils.state_dict import extract_state_dict


DATASETS = ["bloodmnist_224", "dermamnist_224", "organcmnist_224", "organsmnist_224", "chaoshengmnist_224"]
MODELS = ["resnet", "convnext", "vit_t", "swin_tiny"]


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def features_for(meta, args, device):
    reference_path = get_reference_bundle_path(meta)
    if not reference_path.exists():
        raise FileNotFoundError(f"Required reference checkpoint: {reference_path}")
    dataset_path = ROOT / "Med_data" / (meta["dataset"] + ".npz")
    provenance = {"reference_path": str(reference_path.relative_to(ROOT)), "reference_sha256": sha256(reference_path),
                  "data_path": str(dataset_path.relative_to(ROOT)), "data_size": dataset_path.stat().st_size,
                  "data_mtime_ns": dataset_path.stat().st_mtime_ns, "model": meta["model"],
                  "image_size": meta["image_size"], "extraction_batch_size": args.extract_batch_size,
                  "precision": "cuda_autocast_fp16" if device.type == "cuda" else "float32", "version": 1}
    directory = args.output_root / "features" / meta["dataset"] / meta["model"]
    paths = {split: directory / (split + ".pt") for split in ("train", "val", "test")}
    metadata = directory / "provenance.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != provenance:
            raise ValueError(f"Feature provenance changed: {directory}; use a new output root")
        if all(path.exists() for path in paths.values()):
            return {split: load(path) for split, path in paths.items()}, provenance
    directory.mkdir(parents=True, exist_ok=True)
    reference = load(reference_path)["state_dict"]
    model, _, _, _ = build_model(meta["model"], int(meta["num_classes"]), int(meta["in_channels"]), pretrained=False)
    model.load_state_dict(reference, strict=True)
    model.to(device).eval().requires_grad_(False)
    head_keys = classifier_param_keys(reference, int(meta["num_classes"]))
    weight_key = next(key for key in head_keys if reference[key].ndim == 2)
    module_name = weight_key.rsplit(".", 1)[0]
    module = model.get_submodule(module_name)
    captured = []

    def capture_input(layer, inputs):
        captured.append(inputs[0].detach())

    hook = module.register_forward_pre_hook(capture_input)
    with np.load(dataset_path, allow_pickle=False) as archive:
        for split, path in paths.items():
            print(f"FEATURES {meta['dataset']} {meta['model']} {split}", flush=True)
            images = archive[split + "_images"]
            labels = archive[split + "_labels"].reshape(-1).astype(np.int64)
            dataset = NpzTensorDataset(images, labels, transform=_build_small_transform(meta, images))
            loader = DataLoader(dataset, batch_size=args.extract_batch_size, shuffle=False,
                                num_workers=0, pin_memory=device.type == "cuda")
            chunks = []
            with torch.inference_mode():
                for batch_index, (inputs, _) in enumerate(loader):
                    captured.clear()
                    inputs = inputs.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                        logits = model(inputs)
                        if len(captured) != 1 or captured[0].ndim != 2:
                            raise ValueError("Classifier input is not a single feature vector per image")
                        reconstructed = functional.linear(captured[0], module.weight, module.bias)
                    torch.testing.assert_close(logits, reconstructed, rtol=0, atol=0)
                    chunks.append(captured[0].cpu())
            features = torch.cat(chunks)
            if not torch.isfinite(features).all():
                raise ValueError("Non-finite reference features")
            torch.save({"features": features, "labels": torch.from_numpy(labels)}, path)
            del chunks, features, dataset, loader, images
    hook.remove()
    del model, reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dump(metadata, provenance)
    return {split: load(path) for split, path in paths.items()}, provenance


def head_bank(meta, directory):
    weights, biases, sources = [], [], []
    for client in sorted(meta["clients"], key=lambda item: int(item["client_id"])):
        path = directory / client["checkpoint"]
        state = extract_state_dict(load(path))
        keys = classifier_param_keys(state, int(meta["num_classes"]))
        weight_key = next(key for key in keys if state[key].ndim == 2)
        bias_key = weight_key.rsplit(".", 1)[0] + ".bias"
        weight = state[weight_key].float().clone()
        bias = state[bias_key].float().clone() if bias_key in state else torch.zeros(len(weight))
        weights.append(weight)
        biases.append(bias)
        sources.append({"client_id": int(client["client_id"]), "path": str(path.relative_to(ROOT)),
                        "weight_key": weight_key,
                        "head_sha256": hashlib.sha256(weight.numpy().tobytes() + bias.numpy().tobytes()).hexdigest()})
    return torch.stack(weights), torch.stack(biases), sources


def evaluate(model, scores, labels, num_classes):
    with torch.inference_mode():
        logits = model(scores)
        probabilities = logits.softmax(1)
        predicted = logits.argmax(1)
        confusion = torch.bincount(labels * num_classes + predicted, minlength=num_classes ** 2).reshape(num_classes, num_classes)
        loss = float(functional.cross_entropy(logits, labels, reduction="sum"))
    metrics = classification_metrics(confusion.numpy(), probabilities.double().sum(0).numpy(), loss,
                                     float(probabilities.max(1).values.sum()),
                                     macro_ovr_auc(labels.numpy(), probabilities.numpy(), num_classes))
    return metrics, probabilities


def run_case(meta_path, features, provenance, args, device, code_hash):
    started = time.monotonic()
    meta = json.loads(meta_path.read_text())
    relative = meta_path.parent.relative_to(ROOT / "model_hub")
    output = args.output_root / "cases" / relative
    result_path = output / "result.json"
    settings = {"seed": int(meta["seed"]), "epochs": int(meta["epochs"]), "batch_size": int(meta["batch_size"]),
                "lr": float(meta["lr"]), "weight_decay": float(meta["weight_decay"])}
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    weights, biases, sources = head_bank(meta, meta_path.parent)
    identity = {"code_sha256": code_hash, "settings": settings, "heads": sources, "features": provenance,
                "fit_split": "train", "selection_split": "val", "evaluation_split": "test",
                "selection_rule": "maximum validation accuracy, then minimum validation cross entropy",
                "method": "pscore_mlp", "reference_doi": "10.1109/TMI.2021.3129068"}
    if result_path.exists():
        if json.loads((output / "protocol.json").read_text()) != identity:
            raise ValueError(f"Run protocol changed: {output}; use a new output root")
        print(f"RESUME {relative}", flush=True)
        return json.loads(result_path.read_text())
    train_scores = client_probabilities(features["train"]["features"], weights, biases, device)
    val_scores = client_probabilities(features["val"]["features"], weights, biases, device)
    classes = int(meta["num_classes"])
    model, selection, trials = fit_fusion(train_scores, features["train"]["labels"], val_scores,
                                          features["val"]["labels"], classes, settings, torch.device("cpu"))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "input_size": train_scores.shape[1], "num_classes": classes,
                  "hidden_sizes": selection["hidden_sizes"], "head_weights": weights, "head_biases": biases,
                  "reference_path": provenance["reference_path"], "selection": selection}
    torch.save(checkpoint, output / "fusion.pt")
    restored = load(output / "fusion.pt")
    restored_model = ProbabilityFusion(restored["input_size"], classes, restored["hidden_sizes"])
    restored_model.load_state_dict(restored["state_dict"])
    restored_model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(val_scores), restored_model(val_scores), rtol=0, atol=0)
    test_scores = client_probabilities(features["test"]["features"], weights, biases, device)
    metrics, probabilities = evaluate(restored_model, test_scores, features["test"]["labels"], classes)
    np.savez_compressed(output / "test_predictions.npz", probabilities=probabilities.numpy(),
                        labels=features["test"]["labels"].numpy())
    row = {"status": "OK", "source": "supervised_prediction_fusion", "task_type": "small",
           **{key: meta[key] for key in ("dataset", "model", "num_clients", "beta", "seed")},
           "method": "pscore_mlp", "split": "test", **metrics, "seconds": round(time.monotonic() - started, 3),
           "selected_hidden_sizes": str(selection["hidden_sizes"]), "selected_epoch": selection["epoch"],
           "selection_val_accuracy": selection["val_accuracy"], "checkpoint_path": str((output / "fusion.pt").relative_to(ROOT))}
    dump(output / "protocol.json", identity)
    dump(output / "training_history.json", trials)
    dump(result_path, row)
    print(f"OK {relative} ACC={100 * metrics['accuracy']:.2f} F1={100 * metrics['macro_f1']:.2f}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--clients", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--betas", nargs="+", type=float, default=[0, 0.01, 0.1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, help="Override only for smoke tests; formal runs use upstream epochs")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "methods/pscore.py").read_bytes()).hexdigest()
    failures = []
    for model in args.models:
        records = []
        for dataset in args.datasets:
            root = ROOT / "model_hub/small" / dataset / model
            meta = json.loads((root / "clients_3/beta_0/seed_42/meta.json").read_text())
            features, provenance = features_for(meta, args, device)
            for clients in args.clients:
                for beta in args.betas:
                    meta_path = root / f"clients_{clients}" / ("beta_" + f"{beta:g}".replace(".", "p")) / "seed_42/meta.json"
                    try:
                        records.append(run_case(meta_path, features, provenance, args, device, code_hash))
                    except Exception:
                        failure = {"path": str(meta_path), "error": traceback.format_exc()}
                        failures.append(failure)
                        print(failure["error"], flush=True)
                        dump(args.output_root / (model + "_failures.json"), failures)
            del features
            if records:
                path = args.output_root / (model + "_metrics.csv")
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
    if failures:
        raise SystemExit(f"{len(failures)} cases failed")


if __name__ == "__main__":
    main()
