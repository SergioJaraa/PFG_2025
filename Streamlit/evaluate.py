# evaluate.py
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
    model = Net(num_classes=4)
    model.load_state_dict(torch.load(
        "./federated_model3.pth", 
        map_location=torch.device("cpu")
    ))
    model.eval()
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
