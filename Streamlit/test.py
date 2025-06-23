# test_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net  # Asegúrate de tener model.py en el mismo directorio
from collections import defaultdict

# ---- CONFIGURACIÓN ----
TEST_DATA_DIR = "../Data/testing"  # Cambia esto según la ubicación de tu dataset
MODEL_PATH = "../FederatedLearning/FL_Models/federated_model2.pth"
BATCH_SIZE = 32
CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No_Tumor"]

# ---- TRANSFORMACIÓN ----
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---- DATASET Y DATALOADER ----
test_dataset = datasets.ImageFolder(TEST_DATA_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- CARGA DEL MODELO ----
device = torch.device("cpu")  # O 'cuda' si tienes GPU
model = Net(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ---- EVALUACIÓN ----
correct = 0
total = 0
per_class_correct = defaultdict(int)
per_class_total = defaultdict(int)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        for label, prediction in zip(labels, predicted):
            per_class_total[label.item()] += 1
            if label == prediction:
                per_class_correct[label.item()] += 1

# ---- RESULTADOS ----
print(f"\n✅ Overall Accuracy: {100 * correct / total:.2f}%\n")

for i, class_name in enumerate(CLASS_NAMES):
    total_cls = per_class_total[i]
    correct_cls = per_class_correct[i]
    acc = 100 * correct_cls / total_cls if total_cls > 0 else 0
    print(f"🔹 {class_name}: {acc:.2f}% ({correct_cls}/{total_cls})")
