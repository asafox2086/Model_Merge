from pathlib import Path
import re

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_CACHE_ROOT = ROOT / "reference_cache"


def _slugify(value):
    text = str(value).strip().replace("/", "__")
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("._-") or "unknown"


def get_reference_cache_root():
    return DEFAULT_REFERENCE_CACHE_ROOT


def get_reference_bundle_path(meta):
    root = get_reference_cache_root()
    task_type = meta["task_type"]
    if task_type == "small":
        model = _slugify(meta["model"])
        num_classes = int(meta["num_classes"])
        in_channels = int(meta.get("in_channels", 3))
        pretrained = int(bool(meta.get("pretrained", False)))
        seed = int(meta.get("seed", 0))
        filename = f"{model}__cls{num_classes}__in{in_channels}__pretrained{pretrained}__seed{seed}.pt"
        return root / "small" / filename
    if task_type == "vlm":
        clip_model = _slugify(meta["clip_model"])
        random_init = int(bool(meta.get("clip_random_init", False)))
        seed = int(meta.get("seed", 0))
        filename = f"{clip_model}__random{random_init}__seed{seed}.pt"
        return root / "vlm" / filename
    raise ValueError(f"Unsupported task_type for reference cache: {task_type}")


def load_reference_bundle(meta):
    path = get_reference_bundle_path(meta)
    if not path.exists():
        return None, path
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state_dict = payload["state_dict"]
    param_names = list(payload["param_names"])
    cloned_state = {key: value.detach().cpu().clone() for key, value in state_dict.items()}
    return {
        "state_dict": cloned_state,
        "param_names": param_names,
        "path": path,
        "meta": payload.get("meta", {}),
    }, path


def save_reference_bundle(meta, state_dict, param_names):
    path = get_reference_bundle_path(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "task_type": meta["task_type"],
            "dataset": meta.get("dataset", ""),
            "model": meta.get("model", ""),
            "clip_model": meta.get("clip_model", ""),
            "num_classes": int(meta.get("num_classes", 0)),
            "in_channels": int(meta.get("in_channels", 3)),
            "pretrained": bool(meta.get("pretrained", False)),
            "clip_random_init": bool(meta.get("clip_random_init", False)),
            "seed": int(meta.get("seed", 0)),
        },
        "state_dict": {key: value.detach().cpu().clone() for key, value in state_dict.items()},
        "param_names": list(param_names),
    }
    torch.save(payload, path)
    return path
