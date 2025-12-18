from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

def get_dataloaders(
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
):
    # Full training dataset
    full_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=TRANSFORM,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    # Split train → train + val
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
