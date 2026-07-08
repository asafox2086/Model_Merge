#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_npz_splits
from utils import load_json
from utils.hub import beta_to_dirname


INF = 10**18


def parse_args():
    p = argparse.ArgumentParser(
        "Prepare client-local partition artifacts for LAMP-Merge prototype export."
    )
    p.add_argument("--model-hub-root", type=str, default=str(ROOT / "model_hub"))
    p.add_argument("--data-root", type=str, default=str(ROOT / "Med_data"))
    p.add_argument(
        "--output-root",
        type=str,
        default=str(ROOT / "outputs" / "lamp_merge_client_partitions"),
    )
    p.add_argument("--task-type", choices=["small"], default="small")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--small-models", nargs="*", default=None)
    p.add_argument("--num-clients", nargs="*", type=int, default=None)
    p.add_argument("--betas", nargs="*", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["train"], default="train")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def iter_meta_paths(args):
    root = Path(args.model_hub_root) / "small"
    datasets = args.datasets or sorted(path.name for path in root.iterdir() if path.is_dir())
    for dataset in datasets:
        ds_dir = root / dataset
        if not ds_dir.exists():
            continue
        models = args.small_models or sorted(path.name for path in ds_dir.iterdir() if path.is_dir())
        for model in models:
            model_dir = ds_dir / model
            if not model_dir.exists():
                continue
            client_dirs = (
                [f"clients_{n}" for n in args.num_clients]
                if args.num_clients
                else sorted(path.name for path in model_dir.iterdir() if path.is_dir())
            )
            for client_dir in client_dirs:
                cdir = model_dir / client_dir
                if not cdir.exists():
                    continue
                beta_dirs = (
                    [beta_to_dirname(beta) for beta in args.betas]
                    if args.betas
                    else sorted(path.name for path in cdir.iterdir() if path.is_dir())
                )
                for beta_dir in beta_dirs:
                    meta_path = cdir / beta_dir / f"seed_{int(args.seed)}" / "meta.json"
                    if meta_path.exists():
                        yield meta_path


def load_labels(data_root, dataset, split):
    splits = load_npz_splits(str(Path(data_root) / f"{dataset}.npz"))
    return np.asarray(splits[split].labels).reshape(-1).astype(np.int64)


def stable_seed(*items):
    text = "::".join(str(item) for item in items)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


class Dinic:
    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, v, to, cap):
        fwd = [to, cap, None]
        rev = [v, 0, fwd]
        fwd[2] = rev
        self.g[v].append(fwd)
        self.g[to].append(rev)
        return fwd

    def max_flow(self, s, t):
        flow = 0
        n = self.n
        while True:
            level = [-1] * n
            level[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                for e in self.g[v]:
                    if e[1] > 0 and level[e[0]] < 0:
                        level[e[0]] = level[v] + 1
                        q.append(e[0])
            if level[t] < 0:
                return flow
            it = [0] * n

            def dfs(v, f):
                if v == t:
                    return f
                while it[v] < len(self.g[v]):
                    e = self.g[v][it[v]]
                    if e[1] > 0 and level[v] + 1 == level[e[0]]:
                        ret = dfs(e[0], min(f, e[1]))
                        if ret:
                            e[1] -= ret
                            e[2][1] += ret
                            return ret
                    it[v] += 1
                return 0

            while True:
                pushed = dfs(s, INF)
                if not pushed:
                    break
                flow += pushed


def ipf_expected(support, row_targets, col_targets, rounds=400):
    x = support.astype(np.float64)
    x[x > 0] = 1.0
    row_targets = row_targets.astype(np.float64)
    col_targets = col_targets.astype(np.float64)
    for _ in range(rounds):
        row_sum = x.sum(axis=1)
        for i, total in enumerate(row_targets):
            if total > 0 and row_sum[i] > 0:
                x[i, :] *= total / row_sum[i]
        col_sum = x.sum(axis=0)
        for c, total in enumerate(col_targets):
            if total > 0 and col_sum[c] > 0:
                x[:, c] *= total / col_sum[c]
    x *= support
    return x


def complete_integer_table(base, support, row_targets, col_targets, expected):
    counts = base.copy()
    rem_rows = row_targets - counts.sum(axis=1)
    rem_cols = col_targets - counts.sum(axis=0)
    if np.any(rem_rows < 0) or np.any(rem_cols < 0):
        raise ValueError("Lower-bound table exceeds row or column targets.")
    if int(rem_rows.sum()) != int(rem_cols.sum()):
        raise ValueError("Row and column remainders do not match.")
    if int(rem_rows.sum()) == 0:
        return counts

    residual = np.maximum(expected, 0.0) * support
    rem_rows = row_targets - counts.sum(axis=1)
    rem_cols = col_targets - counts.sum(axis=0)
    total = int(rem_rows.sum())
    if total == 0:
        return counts

    k, c_num = support.shape
    source = 0
    client_offset = 1
    class_offset = client_offset + k
    sink = class_offset + c_num
    dinic = Dinic(sink + 1)
    for i, need in enumerate(rem_rows.tolist()):
        dinic.add_edge(source, client_offset + i, int(need))
    edge_records = []
    frac = residual - np.floor(residual)
    for i in range(k):
        classes = np.flatnonzero(support[i] > 0).tolist()
        classes.sort(key=lambda cls: (-float(frac[i, cls]), cls))
        for cls in classes:
            edge = dinic.add_edge(client_offset + i, class_offset + cls, INF)
            edge_records.append((i, cls, edge))
    for cls, need in enumerate(rem_cols.tolist()):
        dinic.add_edge(class_offset + cls, sink, int(need))
    flow = dinic.max_flow(source, sink)
    if int(flow) != total:
        raise ValueError(f"Unable to complete feasible client/class table: flow={flow}, need={total}.")
    for i, cls, edge in edge_records:
        used = edge[2][1]
        if used:
            counts[i, cls] += int(used)
    return counts


def solve_class_count_table(meta, labels):
    num_classes = int(meta["num_classes"])
    clients = meta.get("clients", [])
    k = len(clients)
    support = np.zeros((k, num_classes), dtype=np.int64)
    row_targets = np.zeros(k, dtype=np.int64)
    for i, client in enumerate(clients):
        row_targets[i] = int(client["num_samples"])
        for cls in client.get("classes", []):
            support[i, int(cls)] = 1
    col_targets = np.bincount(labels, minlength=num_classes).astype(np.int64)

    base = support.copy()
    unsupported = np.flatnonzero((support.sum(axis=0) == 0) & (col_targets > 0))
    if unsupported.size:
        raise ValueError(f"Classes without client support: {unsupported.tolist()}")
    if np.any(base.sum(axis=1) > row_targets):
        raise ValueError("A client has fewer samples than its supported-class lower bound.")
    if np.any(base.sum(axis=0) > col_targets):
        raise ValueError("A class has fewer samples than its positive-client lower bound.")

    expected = ipf_expected(support, row_targets - base.sum(axis=1), col_targets - base.sum(axis=0))
    counts = complete_integer_table(base, support, row_targets, col_targets, expected)
    if not np.array_equal(counts.sum(axis=1), row_targets):
        raise ValueError("Solved table row sums do not match client sample counts.")
    if not np.array_equal(counts.sum(axis=0), col_targets):
        raise ValueError("Solved table column sums do not match training class counts.")
    if not np.array_equal((counts > 0).astype(np.int64), support):
        raise ValueError("Solved table support does not match client classes.")
    return counts


def output_dir(output_root, model_hub_root, meta_path):
    rel_dir = meta_path.parent.relative_to(Path(model_hub_root))
    return Path(output_root) / rel_dir


def build_indices(meta, labels, counts):
    k, num_classes = counts.shape
    out = [[] for _ in range(k)]
    rng = np.random.default_rng(
        stable_seed(meta["dataset"], meta["model"], meta["num_clients"], meta["beta"], meta["seed"])
    )
    for cls in range(num_classes):
        cls_indices = np.flatnonzero(labels == cls).astype(np.int64)
        rng.shuffle(cls_indices)
        cursor = 0
        for client_idx in range(k):
            take = int(counts[client_idx, cls])
            if take <= 0:
                continue
            out[client_idx].extend(cls_indices[cursor : cursor + take].tolist())
            cursor += take
        if cursor != cls_indices.size:
            raise ValueError(f"Did not assign all samples for class {cls}: {cursor}/{cls_indices.size}.")
    for client_idx in range(k):
        rng.shuffle(out[client_idx])
    return {f"client_{idx}": np.asarray(values, dtype=np.int64) for idx, values in enumerate(out)}


def write_partition(meta_path, args, labels):
    meta = load_json(meta_path)
    out_dir = output_dir(args.output_root, args.model_hub_root, meta_path)
    out_npz = out_dir / "client_indices.npz"
    out_json = out_dir / "partition_meta.json"
    if out_npz.exists() and out_json.exists() and not args.overwrite:
        return out_npz, "skip"

    counts = solve_class_count_table(meta, labels)
    indices = build_indices(meta, labels, counts)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **indices)
    summary = {
        "format": "lamp_merge_client_partition_v1",
        "privacy_note": (
            "This artifact is an experiment-side carrier for client-local export only. "
            "It is not consumed by LAMP-Merge at server merge time; the server consumes "
            "only aggregate class prototypes and class prevalence counts."
        ),
        "source_meta": str(meta_path),
        "dataset": meta["dataset"],
        "model": meta["model"],
        "num_clients": int(meta["num_clients"]),
        "beta": float(meta["beta"]),
        "seed": int(meta["seed"]),
        "split": args.split,
        "class_count_table": counts.astype(int).tolist(),
        "client_sample_counts": [int(len(indices[f"client_{i}"])) for i in range(len(indices))],
        "global_class_counts": counts.sum(axis=0).astype(int).tolist(),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_npz, "write"


def main():
    args = parse_args()
    meta_paths = list(iter_meta_paths(args))
    print(f"partition preparation start | tasks={len(meta_paths)} | output_root={args.output_root}")
    label_cache = {}
    wrote = 0
    skipped = 0
    for idx, meta_path in enumerate(meta_paths, start=1):
        meta = load_json(meta_path)
        dataset = meta["dataset"]
        if dataset not in label_cache:
            label_cache[dataset] = load_labels(args.data_root, dataset, args.split)
        out_path, status = write_partition(meta_path, args, label_cache[dataset])
        if status == "write":
            wrote += 1
        else:
            skipped += 1
        print(f"{status} ({idx}/{len(meta_paths)}) {meta_path} -> {out_path}")
    print(f"partition preparation done | written={wrote} | skipped={skipped}")


if __name__ == "__main__":
    main()
