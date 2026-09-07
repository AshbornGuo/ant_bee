import io

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


MODEL_PATH = "models/bees_vs_ants_resnet18.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(
    title="Bees vs Ants Classifier",
    description="Upload an image and classify it as a bee or an ant.",
    version="1.0.0"
)


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
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


model, class_names = load_model()


image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)
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


@app.get("/")
def root():
    return {
        "message": "Bees vs Ants API - CI/CD Test Successful!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device)
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WebP images are accepted."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        ) from error

    predicted_class, confidence = predict(image)

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    }