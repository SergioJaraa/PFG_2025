# evaluate.py
import os
import urllib.request
import torch
from torchvision import transforms
from PIL import Image
from model import Net

# Define labels
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)

# Load model
@torch.no_grad()
def load_model():
    model_path = "federated_model3.pth"
    hf_url = "https://huggingface.co/SergioJaraa/federated_model3/resolve/main/federated_model3.pth"

    # Descargar solo si no existe localmente
    if not os.path.exists(model_path):
        try:
            print("Downloading model weights from Hugging Face...")
            urllib.request.urlretrieve(hf_url, model_path)
        except Exception as e:
            raise RuntimeError(f"❌ Error downloading model: {e}")

    model = Net(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model

# Classify image
@torch.no_grad()
def classify_image(image: Image.Image, model):
    tensor = transform(image).unsqueeze(0)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    class_id = predicted.item()
    label = class_labels[class_id] if class_id < len(class_labels) else f"Class {class_id}"
    return label
