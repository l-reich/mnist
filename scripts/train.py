import torch
from app.mnist_cnn import CNN
from app.data import get_dataloaders
from app.trainer import (
    get_loss_function,
    get_optimizer,
    train_one_epoch,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    train_loader, val_loader, _ = get_dataloaders()

    loss_fn = get_loss_function("cross_entropy")
    optimizer = get_optimizer("adam", model, lr=1e-3)

    training_loss_per_epoch = []
    val_loss_per_epoch = []

    num_epochs = 20

    for epoch in range(num_epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        print(f"Epoch {epoch}: loss={loss:.4f}")

        training_loss_per_epoch.append(loss)

        # Put model in evaluation mode for validation
        # This is important for some methods, eg. dropout
        model.eval()

        # Loop over validation dataset
        running_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # on validation dataset, we only do forward, without computing gradients
            with torch.no_grad():
                val_outputs = model(inputs)  # forward pass to obtain network outputs
                loss = loss_fn(val_outputs, labels)  # compute loss with respect to labels

            # print statistics
            running_loss += loss.item()

        mean_loss = running_loss / len(val_loader)
        val_loss_per_epoch.append(mean_loss)

        print(
            f'[Epoch: {epoch + 1} / {num_epochs}]'
            f' Validation loss: {mean_loss:.3f}'
        )

    print('Finished Training')

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
