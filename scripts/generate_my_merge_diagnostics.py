#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.clip_model import encode_image_features, get_clip_base
from utils import load_checkpoint, load_json, save_csv
from utils.runtime import build_runtime


EPS = 1e-8

WEIGHT_FIELDS = [
    "task_type",
    "dataset",
    "model",
    "clip_model",
    "num_clients",
    "beta",
    "seed",
    "selected_candidate",
    "client_index",
    "client_name",
    "seen_classes",
    "prior_weight_pi",
    "diagnostic_weight_alpha_all",
    "medical_weight_alpha_morph",
    "acc_A",
    "medical_acc_M",
    "hard_acc_H",
    "margin_Q",
    "focal_acc_F",
]

VIZ_FIELDS = [
    "task_type",
    "dataset",
    "model",
    "clip_model",
    "num_clients",
    "beta",
    "seed",
    "selected_candidate",
    "split",
    "embedding_source",
    "num_samples",
    "plot_path",
    "status",
    "note",
]

COMBINED_FIELDS = [
    "task_type",
    "dataset",
    "model",
    "clip_model",
    "num_clients",
    "beta",
    "seed",
    "selected_candidate",
    "client_weights",
    "plot_path",
    "status",
    "image",
]


def parse_args():
    p = argparse.ArgumentParser("Export my_merge client weights and classification PCA plots")
    p.add_argument("--output-root", required=True, help="Batch output root containing merged/**/merge_result.json")
    p.add_argument("--data-root", default=str(ROOT / "Med_data"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=2, help="0 means all batches")
    p.add_argument("--max-plots", type=int, default=0, help="0 means no plot limit")
    p.add_argument("--plot-subdir", default="diagnostic_figures", help="Subdirectory under reports/ for PCA images")
    p.add_argument("--no-plots", action="store_true", help="Only export client diagnostic weights")
    return p.parse_args()


def case_fields(payload):
    return {
        "task_type": payload.get("task_type", ""),
        "dataset": payload.get("dataset", ""),
        "model": payload.get("model", ""),
        "clip_model": payload.get("clip_model", ""),
        "num_clients": payload.get("num_clients", ""),
        "beta": payload.get("beta", ""),
        "seed": payload.get("seed", ""),
        "selected_candidate": payload.get("method_info", {}).get("selected_candidate", ""),
    }


def discover_merge_results(output_root):
    output_root = Path(output_root)
    return sorted(output_root.glob("merged/**/merge_result.json"))


def safe_float(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def load_csv_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_weight_rows(payload):
    base = case_fields(payload)
    method_info = payload.get("method_info", {})
    client_info = method_info.get("client_diagnostic_information", [])
    rows = []
    if client_info:
        for item in client_info:
            row = dict(base)
            row.update(
                {
                    "client_index": item.get("client_index", ""),
                    "client_name": item.get("client_name", ""),
                    "seen_classes": " ".join(str(cls) for cls in item.get("seen_classes", [])),
                    "prior_weight_pi": safe_float(item.get("base_weight", "")),
                    "diagnostic_weight_alpha_all": safe_float(item.get("overall_weight", "")),
                    "medical_weight_alpha_morph": safe_float(item.get("morphology_weight", "")),
                    "acc_A": safe_float(item.get("ordinary_accuracy", "")),
                    "medical_acc_M": safe_float(item.get("medical_weighted_accuracy", "")),
                    "hard_acc_H": safe_float(item.get("hard_case_accuracy", "")),
                    "margin_Q": safe_float(item.get("margin_confidence", "")),
                    "focal_acc_F": safe_float(item.get("focal_hard_case_accuracy", "")),
                }
            )
            rows.append(row)
        return rows

    base_weights = method_info.get("base_weights", method_info.get("normalized_weights", []))
    source_clients = payload.get("source_clients", [])
    for client_idx, weight in enumerate(base_weights):
        row = dict(base)
        row.update(
            {
                "client_index": client_idx,
                "client_name": source_clients[client_idx] if client_idx < len(source_clients) else f"client_{client_idx}.pt",
                "seen_classes": "",
                "prior_weight_pi": safe_float(weight),
                "diagnostic_weight_alpha_all": "",
                "medical_weight_alpha_morph": "",
                "acc_A": "",
                "medical_acc_M": "",
                "hard_acc_H": "",
                "margin_Q": "",
                "focal_acc_F": "",
            }
        )
        rows.append(row)
    return rows


def markdown_table(rows, fields, *, image_field=None, max_rows=None):
    shown_rows = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in shown_rows:
        values = []
        for field in fields:
            value = str(row.get(field, ""))
            if image_field and field == image_field and value:
                value = f"![]({value})"
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"| ... | {len(rows) - max_rows} more rows omitted in markdown; see CSV. |" + " |" * (len(fields) - 2))
    return "\n".join(lines) + "\n"


def write_markdown(path, title, description, rows, fields, *, image_field=None, max_rows=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [f"# {title}", "", description, ""]
    content.append(markdown_table(rows, fields, image_field=image_field, max_rows=max_rows))
    path.write_text("\n".join(content), encoding="utf-8")


def sanitize_name(text):
    text = str(text or "none")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "none"


def small_embedding(model, x):
    if hasattr(model, "forward_features"):
        features = model.forward_features(x)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if hasattr(model, "forward_head"):
            try:
                return model.forward_head(features, pre_logits=True).detach().float()
            except TypeError:
                pass
        if torch.is_tensor(features) and features.ndim == 4:
            return features.mean(dim=(2, 3)).detach().float()
        if torch.is_tensor(features) and features.ndim == 3:
            return features[:, 0].detach().float()
        if torch.is_tensor(features):
            return features.detach().float().flatten(1)
    return model(x).detach().float()


def pca_2d(embeddings):
    embeddings = torch.nan_to_num(embeddings.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if embeddings.ndim != 2:
        embeddings = embeddings.flatten(1)
    if embeddings.shape[0] == 0:
        return torch.zeros(0, 2)
    if embeddings.shape[0] == 1:
        return torch.zeros(1, 2)
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    if embeddings.shape[1] == 1:
        return torch.cat([embeddings, torch.zeros_like(embeddings)], dim=1)
    try:
        _u, _s, vh = torch.linalg.svd(embeddings, full_matrices=False)
        coords = embeddings @ vh[:2].t()
    except RuntimeError:
        coords = embeddings[:, :2]
    if coords.shape[1] == 1:
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    coords = torch.nan_to_num(coords[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    return coords.cpu()


def collect_embeddings(meta, checkpoint, data_root, split, device, batch_size, num_workers, max_batches):
    runtime = build_runtime(
        meta=meta,
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    model = runtime["model"]
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    forward_fn = runtime["forward_fn"]
    embeddings = []
    labels = []
    preds = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(runtime["loader"]):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            if meta["task_type"] == "vlm":
                emb = encode_image_features(get_clip_base(model), x).detach().float()
            else:
                emb = small_embedding(model, x)
            logits = forward_fn(model, x).detach().float()
            embeddings.append(emb.cpu())
            labels.append(y.cpu())
            preds.append(logits.argmax(dim=1).cpu())
    if not embeddings:
        return torch.zeros(0, 2), torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), "empty"
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0), torch.cat(preds, dim=0), "ok"


def plot_embeddings(coords, labels, preds, meta, title, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
        labels_np = labels.numpy()
        coords_np = coords.numpy()
        correct = (preds == labels).numpy()
        num_classes = int(meta.get("num_classes", int(labels.max().item()) + 1 if labels.numel() else 1))
        cmap = plt.get_cmap("tab20", max(num_classes, 1))

        ax.scatter(
            coords_np[correct, 0],
            coords_np[correct, 1],
            c=labels_np[correct],
            cmap=cmap,
            s=22,
            alpha=0.82,
            marker="o",
            edgecolors="white",
            linewidths=0.35,
            vmin=0,
            vmax=max(num_classes - 1, 1),
        )
        if (~correct).any():
            ax.scatter(
                coords_np[~correct, 0],
                coords_np[~correct, 1],
                c=labels_np[~correct],
                cmap=cmap,
                s=44,
                alpha=0.95,
                marker="o",
                edgecolors="red",
                linewidths=1.15,
                vmin=0,
                vmax=max(num_classes - 1, 1),
            )
        class_names = meta.get("class_names", [])
        handles = []
        max_legend = min(num_classes, 12)
        for cls_idx in range(max_legend):
            label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"{cls_idx}: {label}",
                    markerfacecolor=cmap(cls_idx),
                    markersize=6,
                )
            )
        ax.legend(handles=handles, loc="best", fontsize=6, frameon=True)
        acc = float(correct.mean()) if correct.size else math.nan
        ax.set_title(f"{title}\nPCA of classifier features, acc={acc:.3f}; red outline = misclassified")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
    except ImportError:
        plot_embeddings_pillow(coords, labels, preds, meta, title, output_path)


def class_color(cls_idx):
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (0, 172, 193),
        (255, 112, 67),
        (124, 179, 66),
        (92, 107, 192),
        (120, 144, 156),
    ]
    return palette[int(cls_idx) % len(palette)]


def plot_embeddings_pillow(coords, labels, preds, meta, title, output_path):
    from PIL import Image, ImageDraw, ImageFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1024, 832
    margin_left, margin_top, margin_right, margin_bottom = 86, 92, 260, 76
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    coords = coords.float()
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min().item()), float(x.max().item())
    y_min, y_max = float(y.min().item()), float(y.max().item())
    if abs(x_max - x_min) < EPS:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < EPS:
        y_max = y_min + 1.0

    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(200, 200, 200),
        width=1,
    )
    for idx in range(1, 5):
        gx = margin_left + int(plot_w * idx / 5)
        gy = margin_top + int(plot_h * idx / 5)
        draw.line([gx, margin_top, gx, margin_top + plot_h], fill=(235, 235, 235), width=1)
        draw.line([margin_left, gy, margin_left + plot_w, gy], fill=(235, 235, 235), width=1)

    correct = preds == labels
    for idx in range(coords.shape[0]):
        px = margin_left + int((float(x[idx].item()) - x_min) / (x_max - x_min) * plot_w)
        py = margin_top + plot_h - int((float(y[idx].item()) - y_min) / (y_max - y_min) * plot_h)
        color = class_color(int(labels[idx].item()))
        if bool(correct[idx].item()):
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color, outline=(255, 255, 255))
        else:
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color, outline=(220, 0, 0), width=2)

    acc = float(correct.float().mean().item()) if labels.numel() else math.nan
    draw.text((24, 20), title, fill=(20, 20, 20), font=font)
    draw.text((24, 42), f"PCA of classifier features, acc={acc:.3f}", fill=(20, 20, 20), font=font)
    draw.text((margin_left + plot_w // 2 - 10, height - 34), "PC1", fill=(40, 40, 40), font=font)
    draw.text((18, margin_top + plot_h // 2), "PC2", fill=(40, 40, 40), font=font)

    class_names = meta.get("class_names", [])
    num_classes = int(meta.get("num_classes", int(labels.max().item()) + 1 if labels.numel() else 1))
    legend_x = margin_left + plot_w + 24
    legend_y = margin_top
    draw.text((legend_x, legend_y - 22), "Classes", fill=(20, 20, 20), font=font)
    for cls_idx in range(min(num_classes, 18)):
        y0 = legend_y + cls_idx * 28
        color = class_color(cls_idx)
        label = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        draw.ellipse([legend_x, y0, legend_x + 12, y0 + 12], fill=color, outline=color)
        draw.text((legend_x + 20, y0 - 2), f"{cls_idx}: {label}"[:32], fill=(30, 30, 30), font=font)
    draw.text((legend_x, height - 64), "red outline = misclassified", fill=(80, 80, 80), font=font)
    image.save(output_path)


def visualization_row(payload, split, embedding_source, plot_path, status, num_samples=0, note=""):
    row = case_fields(payload)
    row.update(
        {
            "split": split,
            "embedding_source": embedding_source,
            "num_samples": num_samples,
            "plot_path": plot_path,
            "status": status,
            "note": note,
        }
    )
    return row


def compact_case_key(row):
    return (
        row.get("task_type", ""),
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("clip_model", ""),
        str(row.get("num_clients", "")),
        str(row.get("beta", "")),
        str(row.get("seed", "")),
        row.get("selected_candidate", ""),
    )


def payload_case_key(payload):
    base = case_fields(payload)
    return compact_case_key(base)


def format_client_weight_summary(rows):
    chunks = []
    for row in sorted(rows, key=lambda item: int(item.get("client_index", 0) or 0)):
        client = row.get("client_name") or f"client_{row.get('client_index', '')}"
        chunks.append(
            (
                f"{client}: "
                f"pi={row.get('prior_weight_pi', '')}, "
                f"alpha_all={row.get('diagnostic_weight_alpha_all', '')}, "
                f"alpha_morph={row.get('medical_weight_alpha_morph', '')}, "
                f"A={row.get('acc_A', '')}, "
                f"M={row.get('medical_acc_M', '')}, "
                f"H={row.get('hard_acc_H', '')}, "
                f"Q={row.get('margin_Q', '')}, "
                f"F={row.get('focal_acc_F', '')}"
            )
        )
    return "<br>".join(chunks)


def build_combined_rows(weight_rows, viz_rows):
    weight_groups = {}
    for row in weight_rows:
        weight_groups.setdefault(compact_case_key(row), []).append(row)
    viz_lookup = {compact_case_key(row): row for row in viz_rows}

    combined = []
    for key in sorted(weight_groups.keys()):
        first = weight_groups[key][0]
        viz = viz_lookup.get(key, {})
        image_path = viz.get("plot_path", "") if viz.get("status") == "ok" else ""
        combined.append(
            {
                "task_type": first.get("task_type", ""),
                "dataset": first.get("dataset", ""),
                "model": first.get("model", ""),
                "clip_model": first.get("clip_model", ""),
                "num_clients": first.get("num_clients", ""),
                "beta": first.get("beta", ""),
                "seed": first.get("seed", ""),
                "selected_candidate": first.get("selected_candidate", ""),
                "client_weights": format_client_weight_summary(weight_groups[key]),
                "plot_path": viz.get("plot_path", ""),
                "status": viz.get("status", "not_plotted"),
                "image": image_path,
            }
        )
    return combined


def build_visualization(
    payload,
    result_path,
    output_root,
    data_root,
    split,
    device,
    batch_size,
    num_workers,
    max_batches,
    plot_subdir,
):
    checkpoint_path = Path(payload.get("merged_checkpoint", ""))
    if not checkpoint_path.exists():
        return visualization_row(
            payload,
            split,
            "",
            "",
            "checkpoint_missing",
            note=f"merged checkpoint not found: {checkpoint_path}",
        )

    meta_path = Path(payload.get("meta_path", result_path.parent / "meta.json"))
    if not meta_path.exists():
        return visualization_row(payload, split, "", "", "meta_missing", note=f"meta not found: {meta_path}")

    try:
        meta = load_json(meta_path)
        checkpoint = load_checkpoint(checkpoint_path, device="cpu")
        embeddings, labels, preds, status = collect_embeddings(
            meta=meta,
            checkpoint=checkpoint,
            data_root=data_root,
            split=split,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_batches=max_batches,
        )
        if status != "ok" or embeddings.shape[0] == 0:
            return visualization_row(payload, split, "", "", "empty", note="no embeddings collected")
        coords = pca_2d(embeddings)
        selected = payload.get("method_info", {}).get("selected_candidate", "unknown")
        name_parts = [
            payload.get("task_type", ""),
            payload.get("dataset", ""),
            payload.get("model") or payload.get("clip_model", ""),
            f"c{payload.get('num_clients')}",
            f"b{payload.get('beta')}",
            f"s{payload.get('seed')}",
            selected,
        ]
        image_name = "__".join(sanitize_name(part) for part in name_parts) + ".png"
        rel_plot_path = Path(plot_subdir) / image_name
        abs_plot_path = Path(output_root) / "reports" / rel_plot_path
        title = " / ".join(str(part) for part in name_parts[:3] if part)
        plot_embeddings(coords, labels, preds, meta, title, abs_plot_path)
        embedding_source = "clip_image_features" if meta["task_type"] == "vlm" else "model_pre_logits"
        return visualization_row(
            payload,
            split,
            embedding_source,
            str(rel_plot_path),
            "ok",
            num_samples=int(labels.numel()),
        )
    except Exception as exc:
        return visualization_row(payload, split, "", "", "failed", note=str(exc))


def build_diagnostics(
    output_root,
    data_root,
    device="cuda:0",
    split="test",
    batch_size=64,
    num_workers=0,
    max_batches=2,
    make_plots=True,
    max_plots=0,
    plot_subdir="diagnostic_figures",
):
    output_root = Path(output_root)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    result_paths = discover_merge_results(output_root)

    weight_rows = []
    viz_rows = []
    viz_csv = report_dir / "my_merge_visualization_index.csv"
    existing_viz_rows = load_csv_rows(viz_csv) if make_plots else []
    existing_viz_lookup = {
        compact_case_key(row): row
        for row in existing_viz_rows
        if row.get("status") == "ok"
    }
    plot_count = 0
    for result_path in result_paths:
        payload = load_json(result_path)
        if payload.get("method") != "my_merge":
            continue
        weight_rows.extend(build_weight_rows(payload))
        if make_plots and (max_plots <= 0 or plot_count < max_plots):
            key = payload_case_key(payload)
            if key in existing_viz_lookup:
                row = existing_viz_lookup[key]
            else:
                row = build_visualization(
                    payload=payload,
                    result_path=result_path,
                    output_root=output_root,
                    data_root=data_root,
                    split=split,
                    device=torch.device(device),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    max_batches=max_batches,
                    plot_subdir=plot_subdir,
                )
            viz_rows.append(row)
            if row["status"] == "ok":
                plot_count += 1
        elif make_plots:
            viz_rows.append(visualization_row(payload, split, "", "", "plot_limit_skipped"))

    weights_csv = report_dir / "my_merge_client_diagnostic_weights.csv"
    weights_md = report_dir / "my_merge_client_diagnostic_weights.md"
    viz_md = report_dir / "my_merge_visualization_index.md"
    combined_csv = report_dir / "my_merge_weight_visualization_table.csv"
    combined_md = report_dir / "my_merge_weight_visualization_table.md"

    save_csv(weights_csv, weight_rows)
    write_markdown(
        weights_md,
        "my_merge Client Diagnostic Weights",
        "Each row is one source client model. Columns are limited to quantities that enter the diagnostic client-information formulas: prior weight pi, final diagnostic weights, and the information vector (A, M, H, Q, F).",
        weight_rows,
        WEIGHT_FIELDS,
        max_rows=300,
    )
    if make_plots:
        save_csv(viz_csv, viz_rows)
        write_markdown(
            viz_md,
            "my_merge Classification PCA Visualizations",
            "Each row identifies the dataset/model/case and shows the corresponding PCA visualization. Points are circles colored by true class; red outlines mark misclassified samples.",
            viz_rows,
            VIZ_FIELDS + ["image"],
            image_field="image",
        )
        # Rewrite image column after table generation needs row augmentation.
        viz_rows_with_images = []
        for row in viz_rows:
            item = dict(row)
            item["image"] = row["plot_path"] if row.get("status") == "ok" else ""
            viz_rows_with_images.append(item)
        write_markdown(
            viz_md,
            "my_merge Classification PCA Visualizations",
            "Each row identifies the dataset/model/case and shows the corresponding PCA visualization. Points are circles colored by true class; red outlines mark misclassified samples.",
            viz_rows_with_images,
            VIZ_FIELDS + ["image"],
            image_field="image",
        )

    combined_rows = build_combined_rows(weight_rows, viz_rows)
    save_csv(combined_csv, combined_rows)
    write_markdown(
        combined_md,
        "my_merge Weight and PCA Visualization Table",
        "Each row is one dataset/model/case. `client_weights` summarizes the diagnostic weights used by the fusion formula; the image column shows the corresponding PCA classification visualization when a checkpoint was available.",
        combined_rows,
        COMBINED_FIELDS,
        image_field="image",
        max_rows=300,
    )

    return {
        "weights_csv": str(weights_csv),
        "weights_md": str(weights_md),
        "visualization_csv": str(viz_csv) if make_plots else "",
        "visualization_md": str(viz_md) if make_plots else "",
        "combined_csv": str(combined_csv),
        "combined_md": str(combined_md),
        "num_weight_rows": len(weight_rows),
        "num_visualization_rows": len(viz_rows),
    }


def main():
    args = parse_args()
    result = build_diagnostics(
        output_root=args.output_root,
        data_root=args.data_root,
        device=args.device,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        make_plots=not args.no_plots,
        max_plots=args.max_plots,
        plot_subdir=args.plot_subdir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
