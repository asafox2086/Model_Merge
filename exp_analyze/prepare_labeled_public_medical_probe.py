#!/usr/bin/env python3
"""Prepare independent labeled public medical probe data for LAMP-Merge.

The output follows the repository's Med_data npz schema. Every supported
builder must provide labels that can be mapped to the target medical class
taxonomy. Samples that exactly match the benchmark/client data after resizing
are removed.
"""

import argparse
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


IMAGE_SIZE = 224
DEFAULT_TOTAL = 1024


BLOOD_REPO = "Falah/Blood_8_classes_Dataset"
BLOOD_PARQUET = "data/train-00000-of-00001-46b5d4047c7ad5e0.parquet"
BLOOD_LABEL_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "ig",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]

# BloodMNIST label order:
# 0 basophil, 1 eosinophil, 2 erythroblast, 3 immature granulocytes,
# 4 lymphocyte, 5 monocyte, 6 neutrophil, 7 platelet.
BLOOD_TO_MEDMNIST = {idx: idx for idx in range(8)}


DERMA_REPO = "flwrlabs/fed-isic2019"
DERMA_PARQUETS = [
    "data/train-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]
DERMA_SOURCE_LABEL_NAMES = [
    "melanoma",
    "melanocytic nevus",
    "basal cell carcinoma",
    "actinic keratoses and intraepithelial carcinoma",
    "benign keratosis-like lesions",
    "dermatofibroma",
    "vascular lesions",
    "squamous cell carcinoma",
]

# DermaMNIST label order:
# 0 actinic keratoses and intraepithelial carcinoma, 1 basal cell carcinoma,
# 2 benign keratosis-like lesions, 3 dermatofibroma, 4 melanoma,
# 5 melanocytic nevi, 6 vascular lesions.
# Fed-ISIC-2019 follows the ISIC 2019 challenge order:
# 0 MEL, 1 NV, 2 BCC, 3 AK, 4 BKL, 5 DF, 6 VASC, 7 SCC.
DERMA_TO_MEDMNIST = {
    0: 4,
    1: 5,
    2: 1,
    3: 0,
    4: 2,
    5: 3,
    6: 6,
}


TOTALSEG_LITE_REPO = "YongchengYAO/TotalSegmentator-CT-Lite"
TOTALSEG_CT_REPO = "Angelou0516/totalsegmentator-organs"
TOTALSEG_META = "meta.csv"
TOTALSEG_MASKS_ZIP = "Masks.zip"

ORGAN_LABEL_NAMES = [
    "bladder",
    "femur-left",
    "femur-right",
    "heart",
    "kidney-left",
    "kidney-right",
    "liver",
    "lung-left",
    "lung-right",
    "pancreas",
    "spleen",
]

# OrganMNIST label order: 0 bladder, 1 femur-left, 2 femur-right, 3 heart,
# 4 kidney-left, 5 kidney-right, 6 liver, 7 lung-left, 8 lung-right,
# 9 pancreas, 10 spleen. TotalSegmentator-CT-Lite uses a 117-class label map.
ORGAN_TO_TOTALSEG = {
    0: [21],
    1: [75],
    2: [76],
    3: [51],
    4: [3],
    5: [2],
    6: [5],
    7: [10, 11],
    8: [12, 13, 14],
    9: [7],
    10: [1],
}

ORGAN_DATASET_ORIENTATION = {
    "organamnist_224": "axial",
    "organcmnist_224": "coronal",
    "organsmnist_224": "sagittal",
}


def direct_hf_download(repo_id, filename, cache_root):
    cache_path = cache_root / repo_id.replace("/", "__") / filename
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    quoted = urllib.parse.quote(filename)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quoted}"
    with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(cache_path)
    return cache_path


def hub_download(repo_id, filename, cache_root):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to prepare this public probe dataset.") from exc
    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, cache_dir=cache_root))


def image_to_array(image):
    if isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, dict) and image.get("bytes") is not None:
        pil = Image.open(BytesIO(image["bytes"]))
    else:
        raise TypeError(f"Unsupported image payload: {type(image)!r}")
    pil = pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def sample_hash(array):
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def perceptual_hashes(array, size=16):
    image = Image.fromarray(array.astype(np.uint8)).convert("L")
    small = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32)
    ahash = (small > small.mean()).reshape(-1)
    diff = np.asarray(image.resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    dhash = (diff[:, 1:] > diff[:, :-1]).reshape(-1)
    return ahash, dhash


def benchmark_hashes(data_root, dataset_name):
    path = Path(data_root) / f"{dataset_name}.npz"
    if not path.exists():
        return set(), np.empty((0, 256), dtype=bool), np.empty((0, 256), dtype=bool)
    data = np.load(path)
    hashes, ahashes, dhashes = set(), [], []
    for split in ("train", "val", "test"):
        key = f"{split}_images"
        if key not in data:
            continue
        for image in data[key]:
            hashes.add(sample_hash(image))
            ahash, dhash = perceptual_hashes(image)
            ahashes.append(ahash)
            dhashes.append(dhash)
    return (
        hashes,
        np.stack(ahashes) if ahashes else np.empty((0, 256), dtype=bool),
        np.stack(dhashes) if dhashes else np.empty((0, 256), dtype=bool),
    )


def is_near_duplicate(array, benchmark_ahashes, benchmark_dhashes, ahash_threshold=16, dhash_threshold=32):
    if benchmark_ahashes.size == 0 or benchmark_dhashes.size == 0:
        return False
    ahash, dhash = perceptual_hashes(array)
    ahash_min = np.count_nonzero(benchmark_ahashes != ahash[None, :], axis=1).min()
    if ahash_min <= ahash_threshold:
        return True
    dhash_min = np.count_nonzero(benchmark_dhashes != dhash[None, :], axis=1).min()
    return bool(dhash_min <= dhash_threshold)


def build_bloodmnist(data_root, cache_root, total):
    import pyarrow.parquet as pq

    blocked, blocked_ahashes, blocked_dhashes = benchmark_hashes(data_root, "bloodmnist_224")
    per_class_limit = max(1, int(np.ceil(total / len(set(BLOOD_TO_MEDMNIST.values())))))
    buckets = defaultdict(list)
    skipped = Counter()
    seen_hashes = set()

    path = direct_hf_download(BLOOD_REPO, BLOOD_PARQUET, cache_root)
    table = pq.read_table(path, columns=["image", "label"])
    for image_payload, source_label in zip(table["image"].to_pylist(), table["label"].to_pylist()):
        source_label = int(source_label)
        if source_label not in BLOOD_TO_MEDMNIST:
            skipped["unmapped_label"] += 1
            continue
        target_label = BLOOD_TO_MEDMNIST[source_label]
        if len(buckets[target_label]) >= per_class_limit:
            skipped["class_full"] += 1
            continue
        image = image_to_array(image_payload)
        digest = sample_hash(image)
        if digest in blocked:
            skipped["benchmark_exact_duplicate"] += 1
            continue
        if digest in seen_hashes:
            skipped["source_duplicate"] += 1
            continue
        if is_near_duplicate(image, blocked_ahashes, blocked_dhashes):
            skipped["benchmark_near_duplicate"] += 1
            continue
        seen_hashes.add(digest)
        buckets[target_label].append(image)
        if sum(len(items) for items in buckets.values()) >= total:
            break

    labels_present = sorted(buckets)
    if labels_present != list(range(8)):
        raise RuntimeError(
            "bloodmnist_224 public probe must cover the full BloodMNIST class taxonomy "
            f"after duplicate removal; observed classes={labels_present}."
        )
    images, labels = [], []
    for label in labels_present:
        for image in buckets[label]:
            images.append(image)
            labels.append(label)
    if not images:
        raise RuntimeError("bloodmnist_224 public probe is empty after duplicate removal.")
    for key in ("benchmark_exact_duplicate", "benchmark_near_duplicate", "source_duplicate", "unmapped_label", "class_full"):
        skipped.setdefault(key, 0)
    return np.stack(images), np.asarray(labels, dtype=np.int64), {
        "source": BLOOD_REPO,
        "source_label_names": BLOOD_LABEL_NAMES,
        "label_mapping": {str(k): int(v) for k, v in BLOOD_TO_MEDMNIST.items()},
        "duplicate_filter": {
            "exact": "sha256_after_resize",
            "near": "ahash16_hamming<=16_or_dhash16_hamming<=32_after_resize",
        },
        "skipped": dict(skipped),
        "class_counts": dict(Counter(labels)),
    }


def build_dermamnist(data_root, cache_root, total):
    import pyarrow.parquet as pq

    blocked, blocked_ahashes, blocked_dhashes = benchmark_hashes(data_root, "dermamnist_224")
    per_class_limit = max(1, int(np.ceil(total / len(set(DERMA_TO_MEDMNIST.values())))))
    buckets = defaultdict(list)
    skipped = Counter()
    seen_hashes = set()

    for filename in DERMA_PARQUETS:
        path = direct_hf_download(DERMA_REPO, filename, cache_root)
        table = pq.read_table(path, columns=["image", "label"])
        for image_payload, source_label in zip(table["image"].to_pylist(), table["label"].to_pylist()):
            source_label = int(source_label)
            if source_label not in DERMA_TO_MEDMNIST:
                skipped["unmapped_label"] += 1
                continue
            target_label = DERMA_TO_MEDMNIST[source_label]
            if len(buckets[target_label]) >= per_class_limit:
                skipped["class_full"] += 1
                continue
            image = image_to_array(image_payload)
            digest = sample_hash(image)
            if digest in blocked:
                skipped["benchmark_exact_duplicate"] += 1
                continue
            if digest in seen_hashes:
                skipped["source_duplicate"] += 1
                continue
            if is_near_duplicate(image, blocked_ahashes, blocked_dhashes):
                skipped["benchmark_near_duplicate"] += 1
                continue
            seen_hashes.add(digest)
            buckets[target_label].append(image)
            if sum(len(items) for items in buckets.values()) >= total:
                break
        if sum(len(items) for items in buckets.values()) >= total:
            break

    labels_present = sorted(buckets)
    if labels_present != list(range(7)):
        raise RuntimeError(
            "dermamnist_224 public probe must cover the full DermaMNIST class taxonomy "
            f"after duplicate removal; observed classes={labels_present}."
        )
    images, labels = [], []
    for label in labels_present:
        for image in buckets[label]:
            images.append(image)
            labels.append(label)
    if not images:
        raise RuntimeError("dermamnist_224 public probe is empty after duplicate removal.")
    for key in ("benchmark_exact_duplicate", "benchmark_near_duplicate", "source_duplicate", "unmapped_label", "class_full"):
        skipped.setdefault(key, 0)
    return np.stack(images), np.asarray(labels, dtype=np.int64), {
        "source": DERMA_REPO,
        "source_label_names": DERMA_SOURCE_LABEL_NAMES,
        "label_mapping": {str(k): int(v) for k, v in DERMA_TO_MEDMNIST.items()},
        "discarded_source_labels": {"7": "squamous cell carcinoma"},
        "duplicate_filter": {
            "exact": "sha256_after_resize",
            "near": "ahash16_hamming<=16_or_dhash16_hamming<=32_after_resize",
        },
        "skipped": dict(skipped),
        "class_counts": dict(Counter(labels)),
    }


def split_and_save(dest_root, dataset_name, images, labels, train_fraction):
    order = np.arange(len(labels))
    rng = np.random.default_rng(42)
    rng.shuffle(order)
    images = images[order]
    labels = labels[order]
    split = max(1, min(len(labels) - 1, int(round(len(labels) * train_fraction))))
    empty_shape = (0, IMAGE_SIZE, IMAGE_SIZE, images.shape[-1]) if images.ndim == 4 else (0, IMAGE_SIZE, IMAGE_SIZE)
    empty_images = np.empty(empty_shape, dtype=np.uint8)
    empty_labels = np.empty((0,), dtype=np.int64)
    dest = dest_root / f"{dataset_name}.npz"
    np.savez_compressed(
        dest,
        train_images=images[:split],
        train_labels=labels[:split],
        val_images=images[split:],
        val_labels=labels[split:],
        test_images=empty_images,
        test_labels=empty_labels,
    )
    return dest, split


def _read_totalseg_meta(cache_root):
    path = hub_download(TOTALSEG_LITE_REPO, TOTALSEG_META, cache_root)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    preferred = [row for row in rows if row.get("split") == "train"]
    return preferred or rows


def _extract_mask_from_zip(zip_path, case_id, cache_root):
    extract_root = cache_root / "totalseg_lite_masks_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    dest = extract_root / f"{case_id}.nii.gz"
    if dest.exists():
        return dest
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        suffixes = [
            f"Masks/{case_id}.nii.gz",
            f"{case_id}.nii.gz",
            f"{case_id}/mask.nii.gz",
            f"{case_id}/seg.nii.gz",
        ]
        name = next((item for item in names if any(item.endswith(suffix) for suffix in suffixes)), None)
        if name is None:
            name = next((item for item in names if case_id in Path(item).name and item.endswith(".nii.gz")), None)
        if name is None:
            raise FileNotFoundError(f"Could not find TotalSegmentator mask for case {case_id} inside {zip_path}.")
        with archive.open(name) as src, dest.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return dest


def _load_nifti(path, dtype=np.float32):
    import nibabel as nib

    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(dtype, copy=False)


def _window_ct(volume):
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.clip(volume, -1000.0, 1000.0)
    return ((volume + 1000.0) * (255.0 / 2000.0)).astype(np.uint8)


def _slice_for_orientation(volume, mask, orientation, index):
    if orientation == "axial":
        return volume[:, :, index], mask[:, :, index]
    if orientation == "coronal":
        return volume[:, index, :], mask[:, index, :]
    if orientation == "sagittal":
        return volume[index, :, :], mask[index, :, :]
    raise ValueError(f"Unsupported orientation: {orientation}")


def _axis_for_orientation(orientation):
    return {"sagittal": 0, "coronal": 1, "axial": 2}[orientation]


def _top_mask_slices(mask, orientation, max_slices):
    axis = _axis_for_orientation(orientation)
    reduce_axes = tuple(dim for dim in range(3) if dim != axis)
    areas = mask.sum(axis=reduce_axes)
    candidates = np.flatnonzero(areas > 0)
    if candidates.size == 0:
        return []
    order = candidates[np.argsort(areas[candidates])[::-1]]
    return [int(idx) for idx in order[:max_slices]]


def _crop_resize_slice(image2d, mask2d):
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    pad = max(8, int(round(0.35 * side)))
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    side = side + 2 * pad
    yy0 = max(0, cy - side // 2)
    xx0 = max(0, cx - side // 2)
    yy1 = min(image2d.shape[0], yy0 + side)
    xx1 = min(image2d.shape[1], xx0 + side)
    yy0 = max(0, yy1 - side)
    xx0 = max(0, xx1 - side)
    crop = image2d[yy0:yy1, xx0:xx1]
    if crop.size == 0:
        return None
    return np.asarray(Image.fromarray(crop).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR), dtype=np.uint8)


def build_organmnist(dataset_name, data_root, cache_root, total):
    if dataset_name not in ORGAN_DATASET_ORIENTATION:
        raise ValueError(f"Unsupported organ dataset: {dataset_name}")
    orientation = ORGAN_DATASET_ORIENTATION[dataset_name]
    blocked, blocked_ahashes, blocked_dhashes = benchmark_hashes(data_root, dataset_name)
    per_class_limit = max(1, int(np.ceil(total / len(ORGAN_TO_TOTALSEG))))
    buckets = defaultdict(list)
    skipped = Counter()
    seen_hashes = set()

    meta_rows = _read_totalseg_meta(cache_root)
    masks_zip = hub_download(TOTALSEG_LITE_REPO, TOTALSEG_MASKS_ZIP, cache_root)
    max_slices_per_case = 6

    for row in meta_rows:
        case_id = row.get("image_id", "").strip()
        if not case_id:
            continue
        if all(len(buckets[label]) >= per_class_limit for label in ORGAN_TO_TOTALSEG):
            break
        try:
            ct_path = hub_download(TOTALSEG_CT_REPO, f"{case_id}/ct.nii.gz", cache_root)
            mask_path = _extract_mask_from_zip(masks_zip, case_id, cache_root)
            ct = _window_ct(_load_nifti(ct_path, dtype=np.float32))
            label_map = _load_nifti(mask_path, dtype=np.int16)
        except Exception:
            skipped["case_load_error"] += 1
            continue
        if ct.shape != label_map.shape:
            skipped["shape_mismatch"] += 1
            continue
        for target_label, source_labels in ORGAN_TO_TOTALSEG.items():
            if len(buckets[target_label]) >= per_class_limit:
                continue
            mask = np.isin(label_map, np.asarray(source_labels, dtype=np.int16))
            slice_indices = _top_mask_slices(mask, orientation, max_slices_per_case)
            if not slice_indices:
                skipped[f"missing_label_{target_label}"] += 1
                continue
            for slice_idx in slice_indices:
                if len(buckets[target_label]) >= per_class_limit:
                    break
                image2d, mask2d = _slice_for_orientation(ct, mask, orientation, slice_idx)
                image = _crop_resize_slice(image2d, mask2d)
                if image is None:
                    skipped["empty_crop"] += 1
                    continue
                digest = sample_hash(image)
                if digest in blocked:
                    skipped["benchmark_exact_duplicate"] += 1
                    continue
                if digest in seen_hashes:
                    skipped["source_duplicate"] += 1
                    continue
                if is_near_duplicate(image, blocked_ahashes, blocked_dhashes):
                    skipped["benchmark_near_duplicate"] += 1
                    continue
                seen_hashes.add(digest)
                buckets[target_label].append(image)

    labels_present = sorted(buckets)
    if labels_present != list(range(11)):
        raise RuntimeError(
            f"{dataset_name} public probe must cover the full OrganMNIST class taxonomy "
            f"after duplicate removal; observed classes={labels_present}."
        )
    images, labels = [], []
    for label in labels_present:
        for image in buckets[label]:
            images.append(image)
            labels.append(label)
    if not images:
        raise RuntimeError(f"{dataset_name} public probe is empty after duplicate removal.")
    for key in ("benchmark_exact_duplicate", "benchmark_near_duplicate", "source_duplicate", "case_load_error", "shape_mismatch", "empty_crop"):
        skipped.setdefault(key, 0)
    return np.stack(images), np.asarray(labels, dtype=np.int64), {
        "source": f"{TOTALSEG_LITE_REPO} masks + {TOTALSEG_CT_REPO} CT images",
        "source_label_names": ORGAN_LABEL_NAMES,
        "label_mapping": {str(k): [int(v) for v in values] for k, values in ORGAN_TO_TOTALSEG.items()},
        "orientation": orientation,
        "duplicate_filter": {
            "exact": "sha256_after_resize",
            "near": "ahash16_hamming<=16_or_dhash16_hamming<=32_after_resize",
        },
        "skipped": dict(skipped),
        "class_counts": dict(Counter(labels)),
    }


def main():
    parser = argparse.ArgumentParser("Prepare type-matched labeled public probe npz files.")
    parser.add_argument("--dest-root", default="PublicMedLabeled_data")
    parser.add_argument("--data-root", default="Med_data")
    parser.add_argument("--cache-root", default=".cache/public_labeled_probe")
    parser.add_argument("--datasets", nargs="+", default=["bloodmnist_224"])
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    args = parser.parse_args()

    dest_root = Path(args.dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    manifest_path = dest_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = [item for item in manifest if item.get("dataset") not in set(args.datasets)]
    else:
        manifest = []

    for dataset_name in args.datasets:
        if dataset_name == "bloodmnist_224":
            images, labels, meta = build_bloodmnist(args.data_root, cache_root, int(args.total))
        elif dataset_name == "dermamnist_224":
            images, labels, meta = build_dermamnist(args.data_root, cache_root, int(args.total))
        elif dataset_name in ORGAN_DATASET_ORIENTATION:
            images, labels, meta = build_organmnist(dataset_name, args.data_root, cache_root, int(args.total))
        else:
            raise ValueError(f"No strict labeled public builder is implemented for {dataset_name}.")
        dest, split = split_and_save(dest_root, dataset_name, images, labels, float(args.train_fraction))
        meta.update({
            "dataset": dataset_name,
            "path": str(dest),
            "num_samples": int(labels.size),
            "train_samples": int(split),
            "val_samples": int(labels.size - split),
        })
        manifest.append(meta)
        print(f"[public-labeled] wrote {dest} samples={labels.size} classes={sorted(set(labels.tolist()))}")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[public-labeled] wrote {manifest_path}")


if __name__ == "__main__":
    main()
