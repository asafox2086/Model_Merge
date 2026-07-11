import torch


def evaluate_classification(model, loader, device, forward_fn, amp_enabled=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if amp_enabled and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    logits = forward_fn(model, x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
            else:
                logits = forward_fn(model, x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return {
        'loss': total_loss / max(total, 1),
        'acc': correct / max(total, 1),
        'num_samples': total,
    }
