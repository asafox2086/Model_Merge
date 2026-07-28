import torch


def evaluate_classification(model, loader, device, forward_fn, amp_enabled=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    targets = []
    predictions = []
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
            predicted = logits.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            targets.append(y.detach().cpu())
            predictions.append(predicted.detach().cpu())
    target = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    prediction = torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)
    num_classes = int(max(target.max().item(), prediction.max().item()) + 1) if total else 0
    if num_classes:
        confusion = torch.bincount(
            target * num_classes + prediction,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes).float()
        true_positive = confusion.diag()
        f1 = (2.0 * true_positive / (2.0 * true_positive + (confusion.sum(0) - true_positive) + (confusion.sum(1) - true_positive)).clamp_min(1.0)).mean().item()
    else:
        f1 = 0.0
    return {
        'loss': total_loss / max(total, 1),
        'acc': correct / max(total, 1),
        'num_samples': total,
        'macro_f1': f1,
    }
