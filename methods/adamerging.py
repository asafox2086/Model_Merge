from collections import OrderedDict

import torch

from utils.runtime import build_runtime
from utils.state_dict import average_state_dicts


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()


def merge_adamerging(meta, state_dicts, base_state, weights, cfg):
    if meta['task_type'] != 'small':
        raise NotImplementedError('AdaMerging v1 currently supports only task_type=small')

    device = torch.device(cfg.get('device', 'cpu'))
    stats_batch_size = int(cfg.get('stats_batch_size', 0) or 0)
    if stats_batch_size <= 0:
        stats_batch_size = int(cfg.get('batch_size', 64))
    stats_num_workers = int(cfg.get('stats_num_workers', cfg.get('num_workers', 0)))
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=stats_batch_size,
        num_workers=stats_num_workers,
        device=device,
    )
    model = runtime['model']
    loader = runtime['loader']

    param_names = [name for name, _ in model.named_parameters()]
    buffer_dict = OrderedDict((name, buf.detach()) for name, buf in model.named_buffers())
    base_params = OrderedDict((name, base_state[name].detach().to(device).float()) for name in param_names)
    deltas = []
    for state_dict in state_dicts:
        delta_dict = OrderedDict()
        for name in param_names:
            delta_dict[name] = (state_dict[name].detach().to(device).float() - base_state[name].detach().to(device).float())
        deltas.append(delta_dict)

    raw = torch.nn.Parameter(torch.full((len(state_dicts),), float(cfg.get('adamerging_prior', 0.3)), device=device))
    optimizer = torch.optim.Adam([raw], lr=float(cfg.get('adamerging_lr', 1e-2)), betas=(0.9, 0.999), weight_decay=0.0)
    epochs = int(cfg.get('adamerging_epochs', 50))
    max_batches = int(cfg.get('adamerging_max_batches', 1))

    try:
        from torch.func import functional_call
    except Exception:
        from torch.nn.utils.stateless import functional_call

    best = None
    best_loss = None
    for _ in range(epochs):
        merged_params = OrderedDict()
        coeffs = raw.clamp(min=0.0, max=1.0)
        for name in param_names:
            merged = base_params[name]
            for idx in range(len(deltas)):
                merged = merged + coeffs[idx] * deltas[idx][name]
            merged_params[name] = merged
        total_loss = 0.0
        steps = 0
        for batch_idx, (x, _) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            logits = functional_call(model, {**merged_params, **buffer_dict}, (x,))
            loss = _softmax_entropy(logits)
            total_loss = total_loss + loss
            steps += 1
        if steps == 0:
            raise RuntimeError('No validation batches available for AdaMerging')
        total_loss = total_loss / steps
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        cur_loss = float(total_loss.detach().cpu())
        if best_loss is None or cur_loss < best_loss:
            best_loss = cur_loss
            best = raw.detach().clamp(min=0.0, max=1.0).cpu()

    if best is None:
        best = raw.detach().clamp(min=0.0, max=1.0).cpu()
    avg_state, _ = average_state_dicts(state_dicts, weights)
    merged = OrderedDict((key, value.detach().cpu().clone()) for key, value in avg_state.items())
    for name in param_names:
        out = base_state[name].detach().cpu().float().clone()
        for idx, state_dict in enumerate(state_dicts):
            delta = state_dict[name].detach().cpu().float() - base_state[name].detach().cpu().float()
            out.add_(delta, alpha=float(best[idx]))
        merged[name] = out.to(dtype=avg_state[name].dtype)
    return merged, {
        'implementation': 'official_adamerging_taskwise_adapted_for_single_dataset',
        'epochs': epochs,
        'lr': float(cfg.get('adamerging_lr', 1e-2)),
        'prior': float(cfg.get('adamerging_prior', 0.3)),
        'max_batches': max_batches,
        'learned_lambdas': [float(x) for x in best.tolist()],
        'stats_split': cfg.get('stats_split', 'val'),
    }
