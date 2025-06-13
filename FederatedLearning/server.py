import flwr as fl

# Start the federated learning server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=3)  # Number of communication rounds
)
