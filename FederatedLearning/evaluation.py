import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Net  

# Config
MODEL_PATH = "federated_m4.pth"       
TEST_DATA_PATH = "../Data/Testing"           
BATCH_SIZE = 32 
IMAGE_SIZE = (256, 256)                  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Net()                            
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation
correct, total = 0, 0
criterion = nn.CrossEntropyLoss()
total_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
avg_loss = total_loss / len(test_loader)

print(f"\nResults:")
print(f"   - Accuracy: {accuracy:.2f}%")
print(f"   - AVG Loss: {avg_loss:.4f}")
