

import os
import time
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=25
):
    """
    Train the model and return the weights with the best validation accuracy.
    """
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(
            tempdir,
            "best_model_params.pt"
        )

        # Save initial parameters
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            for phase in ["train", "val"]:

                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(
                        predictions == labels
                    ).item()

                if phase == "train":
                    scheduler.step()

                epoch_loss = (
                    running_loss / dataset_sizes[phase]
                )

                epoch_accuracy = (
                    running_corrects / dataset_sizes[phase]
                )

                print(
                    f"{phase} Loss: {epoch_loss:.4f} "
                    f"Acc: {epoch_accuracy:.4f}"
                )

                if (
                    phase == "val"
                    and epoch_accuracy > best_acc
                ):
                    best_acc = epoch_accuracy

                    torch.save(
                        model.state_dict(),
                        best_model_params_path
                    )

            print()

        time_elapsed = time.time() - since

        print(
            f"Training complete in "
            f"{time_elapsed // 60:.0f}m "
            f"{time_elapsed % 60:.0f}s"
        )

        print(f"Best validation accuracy: {best_acc:.4f}")

        model.load_state_dict(
            torch.load(
                best_model_params_path,
                weights_only=True,
                map_location=device
            )
        )

    return model


def main():
    # -------------------------
    # 1. Device
    # -------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using {device} device")

    # -------------------------
    # 2. Image transformations
    # -------------------------
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),

        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
    }

    # -------------------------
    # 3. Load dataset
    # -------------------------
    data_dir = "data/hymenoptera_data"

    image_datasets = {
        phase: datasets.ImageFolder(
            os.path.join(data_dir, phase),
            data_transforms[phase]
        )
        for phase in ["train", "val"]
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        for phase in ["train", "val"]
    }

    dataset_sizes = {
        phase: len(image_datasets[phase])
        for phase in ["train", "val"]
    }

    class_names = image_datasets["train"].classes

    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # -------------------------
    # 4. Create model
    # -------------------------
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze all pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace final classification layer
    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        len(class_names)
    )

    model = model.to(device)

    # -------------------------
    # 5. Loss and optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.fc.parameters(),
        lr=0.001,
        momentum=0.9
    )

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    # -------------------------
    # 6. Train model
    # -------------------------
    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        num_epochs=25
    )

    # -------------------------
    # 7. Save trained model
    # -------------------------
    os.makedirs("models", exist_ok=True)

    model_path = "models/bees_vs_ants_resnet18.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        },
        model_path
    )

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()