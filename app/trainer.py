import torch
import torch.nn as nn
import torch.optim as optim


def get_loss_function(name: str) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")


def get_optimizer(name: str, model: nn.Module, lr: float) -> optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)