from collections import defaultdict

import torch
import torch.nn.functional as F

from .runtime import build_runtime, validate_loader_settings


def _iter_batches(loader, max_batches=0):
    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break
        yield batch


def resolve_stats_loader_settings(cfg):
    batch_size = int(cfg.get('stats_batch_size', 0) or 0)
    if batch_size <= 0:
        batch_size = int(cfg.get('batch_size', 64))
    num_workers = int(cfg.get('stats_num_workers', cfg.get('num_workers', 4)))
    return validate_loader_settings(batch_size, num_workers, context='statistics runtime')


def compute_fisher_diagonal(meta, checkpoint, cfg):
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    batch_size, num_workers = resolve_stats_loader_settings(cfg)
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    model = runtime['model']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.train()
    forward_fn = runtime['forward_fn']
    fisher = {}
    counts = 0
    max_batches = int(cfg.get('fisher_max_batches', 0))
    for x, y in _iter_batches(runtime['loader'], max_batches=max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        logits = forward_fn(model, x)
        if logits.ndim == 1 or logits.shape[-1] == 1:
            loss = F.mse_loss(logits.reshape(-1), y.float().reshape(-1))
            loss.backward()
        else:
            probs = torch.softmax(logits, dim=-1).detach()
            log_probs = torch.log_softmax(logits, dim=-1)
            expectations = torch.sqrt(probs) * log_probs
            expectations.sum().backward()
        bs = int(y.size(0))
        counts += bs
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None or not torch.is_floating_point(param):
                continue
            value = (param.grad.detach().float().cpu() ** 2) * bs
            if name not in fisher:
                fisher[name] = value
            else:
                fisher[name].add_(value)
    if counts == 0:
        raise RuntimeError('No batches were processed for fisher statistics')
    for key in fisher:
        fisher[key].div_(float(counts))
    return fisher


def collect_linear_covariances(meta, checkpoint, cfg):
    device = torch.device(cfg.get('stats_device', cfg.get('device', 'cpu')))
    batch_size, num_workers = resolve_stats_loader_settings(cfg)
    runtime = build_runtime(
        meta=meta,
        data_root=cfg['data_root'],
        split=cfg.get('stats_split', 'val'),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    model = runtime['model']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    forward_fn = runtime['forward_fn']
    max_batches = int(cfg.get('regmean_max_batches', 0))
    max_dim = int(cfg.get('regmean_max_dim', 1024))

    cov_sums = {}
    counts = defaultdict(int)
    handles = []

    def make_hook(module_name, in_features):
        def _hook(module, inputs, output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().cpu()
            if x.numel() == 0 or in_features > max_dim:
                return
            xtx = x.transpose(0, 1).mm(x)
            if module_name not in cov_sums:
                cov_sums[module_name] = xtx / float(x.shape[0])
                counts[module_name] = int(x.shape[0])
            else:
                prev = counts[module_name]
                cov_sums[module_name] = (cov_sums[module_name] * prev + xtx) / float(prev + x.shape[0])
                counts[module_name] += int(x.shape[0])
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.in_features <= max_dim:
            handles.append(module.register_forward_hook(make_hook(name, module.in_features)))

    with torch.no_grad():
        for x, _ in _iter_batches(runtime['loader'], max_batches=max_batches):
            x = x.to(device, non_blocking=True)
            _ = forward_fn(model, x)

    for handle in handles:
        handle.remove()

    return {key: value for key, value in cov_sums.items() if counts[key] > 0}
