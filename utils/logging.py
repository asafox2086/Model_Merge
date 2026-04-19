import csv
from pathlib import Path


DEFAULT_SUMMARY_KEY_FIELDS = (
    'task_type',
    'dataset',
    'model',
    'clip_model',
    'num_clients',
    'beta',
    'seed',
    'method',
    'merge_weight_mode',
    'split',
)


def _load_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _build_key(row, key_fields):
    normalized = []
    for field in key_fields:
        value = row.get(field, '')
        if field == 'beta' and value != '':
            normalized.append(format(float(value), 'g'))
        elif field in {'num_clients', 'seed'} and value != '':
            normalized.append(str(int(value)))
        else:
            normalized.append(str(value))
    return tuple(normalized)


def upsert_csv_row(csv_path, row, key_fields):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_csv_rows(csv_path)
    row = {key: row.get(key, '') for key in row.keys()}
    target_key = _build_key(row, key_fields)
    replaced = False
    for idx, existing in enumerate(rows):
        if _build_key(existing, key_fields) == target_key:
            rows[idx] = {key: row.get(key, '') for key in row.keys()}
            replaced = True
            break
    if not replaced:
        rows.append(row)

    fieldnames = list(row.keys())
    normalized_rows = [{key: item.get(key, '') for key in fieldnames} for item in rows]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def append_summary_row(csv_path, row):
    key_fields = [field for field in DEFAULT_SUMMARY_KEY_FIELDS if field in row]
    upsert_csv_row(csv_path, row, key_fields=key_fields)
