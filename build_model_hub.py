#!/usr/bin/env python3
import csv
import json
import shutil
from pathlib import Path

SOURCE_ROOT = Path('/data1/users/weiyipan/ML/MedModelMerging')
TARGET_ROOT = Path('/data1/users/weiyipan/ML/MedMNSITMerge')
MODEL_HUB = TARGET_ROOT / 'model_hub'

VLM_BATCHES = [
    '20260309_101100_vlm_main',
    '20260309_195218_vlm_main',
    '20260310_185907_vlm_main',
]


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_beta(value):
    return format(float(value), 'g').replace('.', 'p')


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def is_complete_vlm_run(results):
    if not results:
        return False
    for item in results:
        ckpt = Path(item['checkpoint'])
        if not ckpt.exists():
            return False
    return True


def collect_existing_small_manifest_rows():
    rows = []
    for meta_path in sorted((MODEL_HUB / 'small').glob('*/*/clients_*/beta_*/seed_*/meta.json')):
        meta = load_json(meta_path)
        hub_dir = meta_path.parent
        rows.append({
            'task_type': 'small',
            'dataset': meta['dataset'],
            'model': meta['model'],
            'clip_model': '',
            'num_clients': int(meta['num_clients']),
            'beta': meta['beta'],
            'seed': int(meta['seed']),
            'hub_dir': str(hub_dir.relative_to(MODEL_HUB)),
            'meta_path': str(meta_path.relative_to(MODEL_HUB)),
            'source_batch_dir': meta.get('source_batch_dir', ''),
            'source_run_name': meta.get('source_run_name', ''),
        })
    return rows


def collect_vlm_runs():
    runs = []
    for batch in VLM_BATCHES:
        batch_dir = SOURCE_ROOT / 'logs' / batch
        for cfg_path in sorted(batch_dir.glob('*/config.json')):
            run_dir = cfg_path.parent
            results_path = run_dir / 'client_results.json'
            if not results_path.exists():
                continue
            cfg = load_json(cfg_path)
            results = load_json(results_path)
            if not is_complete_vlm_run(results):
                continue
            runs.append((batch, run_dir.name, cfg, results))
    return runs


def build_vlm_hub(manifest_rows):
    vlm_root = MODEL_HUB / 'vlm'
    if vlm_root.exists():
        shutil.rmtree(vlm_root)
    vlm_root.mkdir(parents=True, exist_ok=True)

    for batch, run_name, cfg, results in collect_vlm_runs():
        dataset = cfg['dataset']
        clip_model = cfg['clip_model']
        clip_dir = clip_model.split('/')[-1]
        clients = int(cfg['num_clients'])
        beta_str = format_beta(cfg['beta'])
        seed = int(cfg['seed'])
        hub_dir = vlm_root / dataset / clip_dir / f'clients_{clients}' / f'beta_{beta_str}' / f'seed_{seed}'
        hub_dir.mkdir(parents=True, exist_ok=True)

        clients_meta = []
        for item in results:
            src_ckpt = Path(item['checkpoint'])
            dst_name = f"client_{item['unit_id']}.pt"
            dst_ckpt = hub_dir / dst_name
            copy_file(src_ckpt, dst_ckpt)
            clients_meta.append({
                'client_id': item['unit_id'],
                'unit_name': item.get('unit_name', f"client_{item['unit_id']}"),
                'num_samples': item['num_samples'],
                'classes': item['classes'],
                'best_val_acc': item['best_val_acc'],
                'test_acc': item['test_acc'],
                'test_loss': item['test_loss'],
                'checkpoint': dst_name,
                'source_checkpoint': str(src_ckpt),
            })

        meta = {
            'task_type': 'vlm',
            'dataset': dataset,
            'model': clip_dir,
            'clip_model': clip_model,
            'train_mode': cfg.get('train_mode', cfg.get('run_mode', 'fl')).lower(),
            'num_clients': clients,
            'beta': cfg['beta'],
            'seed': seed,
            'num_classes': cfg['num_classes'],
            'class_names': cfg.get('class_names', []),
            'epochs': cfg['epochs'],
            'batch_size': cfg['batch_size'],
            'lr': cfg['lr'],
            'weight_decay': cfg['weight_decay'],
            'image_size': cfg['image_size'],
            'clip_random_init': cfg.get('clip_random_init', False),
            'text_template': cfg.get('text_template'),
            'source_batch_dir': batch,
            'source_run_name': run_name,
            'source_log_dir': str(SOURCE_ROOT / 'logs' / batch / run_name),
            'source_ckpt_dir': str(SOURCE_ROOT / 'checkpoints' / batch / run_name),
            'client_classes': cfg.get('client_classes', {}),
            'clients': clients_meta,
        }
        save_json(hub_dir / 'meta.json', meta)
        manifest_rows.append({
            'task_type': 'vlm',
            'dataset': dataset,
            'model': clip_dir,
            'clip_model': clip_model,
            'num_clients': clients,
            'beta': cfg['beta'],
            'seed': seed,
            'hub_dir': str(hub_dir.relative_to(MODEL_HUB)),
            'meta_path': str((hub_dir / 'meta.json').relative_to(MODEL_HUB)),
            'source_batch_dir': batch,
            'source_run_name': run_name,
        })


def save_manifest(rows):
    rows = sorted(
        rows,
        key=lambda x: (
            x['task_type'], x['dataset'], x['model'], x['clip_model'],
            x['num_clients'], float(x['beta']), x['seed']
        ),
    )
    fieldnames = [
        'task_type', 'dataset', 'model', 'clip_model',
        'num_clients', 'beta', 'seed', 'hub_dir', 'meta_path',
        'source_batch_dir', 'source_run_name'
    ]
    with (MODEL_HUB / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (MODEL_HUB / 'manifest.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    manifest_rows = collect_existing_small_manifest_rows()
    build_vlm_hub(manifest_rows)
    save_manifest(manifest_rows)
    print(f'model_hub updated at: {MODEL_HUB}')
    print(f'total entries: {len(manifest_rows)}')


if __name__ == '__main__':
    main()
