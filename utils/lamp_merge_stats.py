from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from utils.hub import beta_to_dirname


DEFAULT_PROTOTYPE_ROOT_NAME = "lamp_merge_client_local_proto_stats"
VALID_STATS_FORMAT = "lamp_merge_client_prototype_stats_v2"


def default_prototype_root(root: Path) -> Path:
    return Path(root) / "outputs" / DEFAULT_PROTOTYPE_ROOT_NAME


def is_valid_prevalence_source(source: object) -> bool:
    source = str(source or "")
    return source.startswith("client_local_dataset") or source.startswith("client_uploaded_label_counts")


def prototype_stats_path(root: Path, meta_or_row: Mapping[str, object]) -> Path:
    model_name = str(meta_or_row.get("model") or str(meta_or_row.get("clip_model", "")).replace("/", "__"))
    return (
        Path(root)
        / str(meta_or_row.get("task_type"))
        / str(meta_or_row.get("dataset"))
        / model_name
        / f"clients_{int(meta_or_row.get('num_clients'))}"
        / beta_to_dirname(meta_or_row.get("beta"))
        / f"seed_{int(meta_or_row.get('seed'))}"
        / "prototype_stats.pt"
    )


def validate_lamp_merge_stats_payload(payload: object, path: Path | None = None) -> None:
    location = f": {path}" if path is not None else ""
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid LAMP-Merge prototype statistics payload{location}: expected dict.")
    if payload.get("format") != VALID_STATS_FORMAT:
        raise ValueError(
            f"Invalid LAMP-Merge prototype statistics payload{location}: "
            f"format must be {VALID_STATS_FORMAT!r}."
        )
    if payload.get("feature_space") != "reference_model":
        raise ValueError(
            f"Invalid LAMP-Merge prototype statistics payload{location}: "
            "feature_space must be 'reference_model'."
        )
    clients = payload.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError(f"Invalid LAMP-Merge prototype statistics payload{location}: missing clients.")

    client_sources = []
    for idx, item in enumerate(clients):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid client statistics{location}: client {idx} is not a dict.")
        if item.get("class_feature_mean") is None:
            raise ValueError(f"Invalid client statistics{location}: client {idx} lacks class_feature_mean.")
        if item.get("class_feature_counts") is None and item.get("class_counts") is None:
            raise ValueError(f"Invalid client statistics{location}: client {idx} lacks class_feature_counts.")
        prevalence = item.get("class_prevalence_counts")
        if prevalence is None:
            raise ValueError(f"Invalid client statistics{location}: client {idx} lacks class_prevalence_counts.")
        prevalence_tensor = torch.as_tensor(prevalence, dtype=torch.float32).view(-1)
        if float(prevalence_tensor.sum().item()) <= 0.0:
            raise ValueError(f"Invalid client statistics{location}: client {idx} has empty class_prevalence_counts.")
        source = item.get("prevalence_source")
        if source is not None:
            client_sources.append(source)

    payload_source = payload.get("prevalence_source")
    if is_valid_prevalence_source(payload_source):
        return
    if client_sources and len(client_sources) == len(clients) and all(is_valid_prevalence_source(source) for source in client_sources):
        return
    raise ValueError(
        f"Invalid LAMP-Merge prototype statistics payload{location}: class_prevalence_counts must be "
        "computed by each client from its local D_i or explicitly uploaded as client label counts."
    )


def load_lamp_merge_stats(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    validate_lamp_merge_stats_payload(payload, Path(path))
    return payload


def validate_lamp_merge_stats_root(root: Path, *, min_files: int = 1) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Missing LAMP-Merge client statistics root: {root}. "
            "Export client-local aggregate prototype statistics before running LAMP-Merge analyses."
        )
    paths = sorted(root.glob("small/*/*/clients_*/*/seed_*/prototype_stats.pt"))
    if len(paths) < int(min_files):
        raise FileNotFoundError(f"No LAMP-Merge prototype_stats.pt files found under {root}.")
    for path in paths:
        load_lamp_merge_stats(path)
    return paths
