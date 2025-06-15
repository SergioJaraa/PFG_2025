import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import Net  
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# CLIENTS ROUTE
# Get client ID from environment variable
client_id = os.environ.get("CLIENT_ID", "client1")
DATA_PATH = f"./data/{client_id}"

# Remove hidden files like .DS_Store from classes list
classes = [cls for cls in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, cls)) and not cls.startswith('.')]



transform = transforms.Compose([
    transforms.Grayscale(),                  # grayscale
    transforms.Resize((64, 64)),             # resizing the data
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                   # converting to a tensor 
    transforms.Normalize((0.5,), (0.5,)),    # normalizing 
])

# Datasets
train_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
model = Net(num_classes=4).to(DEVICE)

# Loss and optimizer
weights = torch.tensor([1.5, 1.5, 1.0, 1.0], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Flower client definition
class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # Return model parameters as a list of NumPy arrays
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load parameters back into the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Train model using client-local data
        self.set_parameters(parameters)
        model.train()
        for epoch in range(1):  # One epoch per round
            for batch in train_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Evaluate model on client-local data
        self.set_parameters(parameters)
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss += criterion(y_pred, y).item()
                correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
        loss /= len(train_loader.dataset)
        accuracy = correct / len(train_loader.dataset)
        return float(loss), len(train_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())