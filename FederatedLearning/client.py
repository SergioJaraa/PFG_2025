import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

transform = transforms.Compose([
    transforms.Grayscale(),                  # grayscale
    transforms.Resize((64, 64)),             # resizing the data
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                   # converting to a tensor 
    transforms.Normalize((0.5,), (0.5,)),    # normalizing 
])

# Load local dataset (change to client2, client3... as needed)
dataset = datasets.ImageFolder("./data/client1", transform=transform)
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, and optimizer
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Flower client class
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # Send model parameters to server
        return [val.cpu().numpy() for val in model.state_dict().values()]
    
    def set_parameters(self, parameters):
        # Load model parameters from server
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for epoch in range(1):  # Local training: 1 epoch per round
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Optional: implement local evaluation
        self.set_parameters(parameters)
        return 0.0, len(trainloader.dataset), {"accuracy": 0.0}

# Start the client and connect to the server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())