#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import NpzTensorDataset
from export_lamp_merge_prototypes import (
    build_transform,
    iter_meta_paths,
    load_split_arrays,
    load_client_indices,
    output_path,
    select_feature_indices,
)
from model import build_model
from utils import extract_state_dict, load_checkpoint, load_json


def parse_args():
    p = argparse.ArgumentParser("Export class-balanced client parameter sensitivity for LAMP-Merge")
    p.add_argument("--model-hub-root", type=str, default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument("--output-root", type=str, default=str(ROOT / "ClientSensitivityStats"))
    p.add_argument(
        "--partition-root",
        type=str,
        default="",
        help="Root containing true client partition artifacts in the same layout as model_hub.",
    )
    p.add_argument("--task-type", choices=["small"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train", "val", "trainval"], default="train")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples-per-client", type=int, default=1024)
    p.add_argument("--max-samples-per-class", type=int, default=512)
    p.add_argument("--class-balance-power", type=float, default=0.5)
    p.add_argument("--sensitivity-mode", choices=["expected", "ce"], default="expected")
    p.add_argument("--store-fp16", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def class_weights_from_counts(counts, power):
    counts = counts.float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    if present.any():
        mean_count = counts[present].mean().clamp_min(1.0)
        weights[present] = (mean_count / counts[present].clamp_min(1.0)).pow(float(power))
    return weights.clamp(0.05, 20.0)


def compute_client_stats(meta, model, loader, selected_labels, device, args):
    num_classes = int(meta["num_classes"])
    counts = torch.bincount(torch.as_tensor(selected_labels, dtype=torch.long), minlength=num_classes).float()
    class_weights = class_weights_from_counts(counts, args.class_balance_power).to(device)
    correct = torch.zeros(num_classes, dtype=torch.float32)
    sensitivity = {}
    processed = 0

    model.train()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        pred = logits.detach().argmax(dim=1).cpu()
        y_cpu = y.detach().cpu()
        for cls in y_cpu.unique().tolist():
            cls = int(cls)
            mask = y_cpu == cls
            correct[cls] += float((pred[mask] == cls).sum().item())

        sample_weights = class_weights[y].detach()
        if args.sensitivity_mode == "ce":
            losses = F.cross_entropy(logits, y, reduction="none")
            loss = (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        else:
            probs = torch.softmax(logits, dim=-1).detach()
            log_probs = torch.log_softmax(logits, dim=-1)
            expectations = (torch.sqrt(probs.clamp_min(1e-8)) * log_probs).sum(dim=1)
            loss = (expectations * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        loss.backward()
        batch_weight = float(sample_weights.sum().detach().cpu().item())
        processed += int(y.numel())
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None or not torch.is_floating_point(param):
                continue
            value = param.grad.detach().float().cpu().square() * batch_weight
            if name not in sensitivity:
                sensitivity[name] = value
            else:
                sensitivity[name].add_(value)

    if processed <= 0:
        raise RuntimeError("No samples were processed for sensitivity export.")
    for key in list(sensitivity.keys()):
        value = sensitivity[key] / float(processed)
        sensitivity[key] = value.to(torch.float16) if args.store_fp16 else value
    recall = correct / counts.clamp_min(1.0)
    return counts, recall, sensitivity


def export_one(meta_path, args):
    meta = load_json(meta_path)
    if meta.get("task_type") != "small":
        return None
    out_path = output_path(args.output_root, meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = load_split_arrays(args.data_root, meta["dataset"], args.split)
    transform = build_transform(meta, images)
    dataset = NpzTensorDataset(images, labels, transform=transform)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    clients_payload = []

    for client_idx, client in enumerate(meta.get("clients", [])):
        classes = [int(c) for c in client.get("classes", [])]
        local_indices, local_index_source = load_client_indices(meta_path, args, client, client_idx, len(labels))
        if local_indices is None:
            raise ValueError(
                "Sensitivity export requires true local D_i indices; class-filter proxy sampling "
                f"is not valid. Missing client {client_idx}: {meta_path}"
            )
        indices = select_feature_indices(
            labels,
            local_indices,
            client_idx,
            max_samples=int(args.max_samples_per_client),
            max_per_class=int(args.max_samples_per_class),
        )
        selected_labels = labels[indices]
        loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=str(device).startswith("cuda"),
        )
        model, _, _, _ = build_model(
            name=meta["model"],
            num_classes=int(meta["num_classes"]),
            in_channels=int(meta.get("in_channels", 3)),
            pretrained=False,
        )
        ckpt_path = meta_path.parent / client.get("checkpoint", f"client_{client_idx}.pt")
        checkpoint = load_checkpoint(ckpt_path, device="cpu")
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        model.to(device)
        counts, recall, sensitivity = compute_client_stats(meta, model, loader, selected_labels, device, args)
        clients_payload.append(
            {
                "client_id": int(client.get("client_id", client_idx)),
                "classes": classes,
                "local_index_source": local_index_source,
                "num_selected_samples": int(counts.sum().item()),
                "class_counts": counts.tolist(),
                "class_recall": recall.tolist(),
                "parameter_sensitivity": sensitivity,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "format": "lamp_merge_class_balanced_sensitivity_v1",
        "privacy": "client-side aggregate gradients only; no raw image and no per-sample feature is stored",
        "source_split": args.split,
        "class_balance_power": float(args.class_balance_power),
        "sensitivity_mode": args.sensitivity_mode,
        "meta_path": str(meta_path),
        "task_type": meta["task_type"],
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "seed": int(meta["seed"]),
        "num_classes": int(meta["num_classes"]),
        "clients": clients_payload,
    }
    torch.save(payload, out_path)
    return out_path


def main():
    args = parse_args()
    paths = list(iter_meta_paths(args))
    print(f"sensitivity export start | tasks={len(paths)} | output_root={args.output_root}")
    done = 0
    for idx, meta_path in enumerate(paths, start=1):
        out_path = export_one(meta_path, args)
        if out_path is not None:
            done += 1
            print(f"done ({idx}/{len(paths)}) {meta_path} -> {out_path}")
    print(f"sensitivity export done | written={done}")


if __name__ == "__main__":
    main()
