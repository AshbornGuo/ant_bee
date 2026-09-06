import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "models/bees_vs_ants_resnet18.pth"


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)

    number_of_features = model.fc.in_features
    model.fc = nn.Linear(
        number_of_features,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path: str):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image)

    # Change shape from [3, 224, 224]
    # to [1, 3, 224, 224]
    return image_tensor.unsqueeze(0)


def predict_image(
    model,
    image_tensor,
    class_names,
    device
):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

    predicted_class = class_names[predicted_index]
    confidence = probabilities[0][predicted_index].item()

    return predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an image contains an ant or a bee."
    )

    parser.add_argument(
        "image_path",
        help="Path to the image file"
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using {device} device")

    model, class_names = load_model(
        MODEL_PATH,
        device
    )

    image_tensor = preprocess_image(
        str(image_path)
    )

    predicted_class, confidence = predict_image(
        model,
        image_tensor,
        class_names,
        device
    )

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()