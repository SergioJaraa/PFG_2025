import flwr as fl
import torch
from model import Net  
from flwr.common import parameters_to_ndarrays

# Strategy to save the model
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.final_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters  # weights saved 
        return aggregated_parameters, {}

    def aggregate_evaluate(self, rnd, results, failures):
        print(f"[SUMMARY] Round {rnd} finished.")
        return super().aggregate_evaluate(rnd, results, failures)

# Strat
strategy = SaveModelStrategy()

# Server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy
)

model = Net()  
weights = parameters_to_ndarrays(strategy.final_parameters)

# Weights
params_dict = zip(model.state_dict().keys(), weights)
state_dict = {k: torch.tensor(v) for k, v in params_dict}
model.load_state_dict(state_dict, strict=True)

# Saved to disk
torch.save(model.state_dict(), "federated_m4.pth")
print("Saved as 'federated_m4.pth'")
