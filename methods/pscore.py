"""Classification adaptation of Wimmer et al.'s TMI 2022 Pscore-MLP fusion."""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


class ProbabilityFusion(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes):
        super().__init__()
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, scores):
        return self.network(scores)


def client_probabilities(features, weights, biases, device, batch_size=1024):
    outputs = []
    weights = weights.to(device)
    biases = biases.to(device)
    with torch.inference_mode():
        for batch in features.split(batch_size):
            batch = batch.to(device).float()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = torch.stack([functional.linear(batch, weight, bias) for weight, bias in zip(weights, biases)], dim=1)
            outputs.append(logits.float().softmax(dim=-1).flatten(1).cpu())
    return torch.cat(outputs)


def fit_fusion(train_scores, train_labels, val_scores, val_labels, num_classes, settings, device):
    input_size = train_scores.shape[1]
    architectures = [(input_size,), (input_size, input_size), (input_size, max(1, input_size // 2))]
    train_scores = train_scores.to(device)
    train_labels = train_labels.to(device)
    val_scores = val_scores.to(device)
    val_labels = val_labels.to(device)
    best = None
    trials = []
    for trial_index, hidden_sizes in enumerate(architectures):
        torch.manual_seed(settings["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(settings["seed"])
        generator = torch.Generator().manual_seed(settings["seed"])
        model = ProbabilityFusion(input_size, num_classes, hidden_sizes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        history = []
        for epoch in range(1, settings["epochs"] + 1):
            model.train()
            permutation = torch.randperm(len(train_labels), generator=generator)
            total_loss = 0.0
            for indices in permutation.split(settings["batch_size"]):
                indices = indices.to(device)
                logits = model(train_scores[indices])
                loss = functional.cross_entropy(logits, train_labels[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            model.eval()
            with torch.inference_mode():
                val_logits = model(val_scores)
                val_correct = int((val_logits.argmax(1) == val_labels).sum())
                val_loss = float(functional.cross_entropy(val_logits, val_labels))
            if not np.isfinite(val_loss):
                raise ValueError("Non-finite fusion validation loss")
            record = {"epoch": epoch, "train_loss": total_loss / len(train_labels),
                      "val_accuracy": val_correct / len(val_labels), "val_loss": val_loss}
            history.append(record)
            rank = (val_correct, -val_loss)
            if best is None or rank > best["rank"]:
                best = {"rank": rank, "trial": trial_index, "hidden_sizes": hidden_sizes,
                        "epoch": epoch, "state_dict": copy.deepcopy(model.cpu().state_dict()),
                        "val_accuracy": record["val_accuracy"], "val_loss": val_loss}
                model.to(device)
        trials.append({"hidden_sizes": hidden_sizes, "history": history})
    model = ProbabilityFusion(input_size, num_classes, best["hidden_sizes"])
    model.load_state_dict(best["state_dict"])
    return model.eval(), {key: value for key, value in best.items() if key not in ("state_dict", "rank")}, trials
