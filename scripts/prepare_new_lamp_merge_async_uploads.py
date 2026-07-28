#!/usr/bin/env python3
"""Create one replayable client-upload artifact per exported LAMP statistic."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser("Split aggregate prototype statistics into client upload artifacts.")
    parser.add_argument("--prototype-stats-path", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    payload = torch.load(args.prototype_stats_path, map_location="cpu", weights_only=False)
    if payload.get("feature_space") != "reference_model":
        raise ValueError("prototype statistics must use feature_space=reference_model.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for client_id, client in enumerate(payload.get("clients", [])):
        torch.save(
            {
                "format": "new_lamp_merge_client_upload_v1",
                "client_id": client_id,
                "feature_space": payload["feature_space"],
                "prevalence_source": payload.get("prevalence_source", ""),
                "client": client,
            },
            output_root / f"client_{client_id}.pt",
        )
    print(f"wrote {len(payload.get('clients', []))} client upload artifacts to: {output_root}")


if __name__ == "__main__":
    main()
