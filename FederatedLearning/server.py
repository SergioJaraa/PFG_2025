import flwr as fl

# Start the Flower federated learning server
fl.server.start_server(
    server_address="127.0.0.1:8080",  # Local server address
    config=fl.server.ServerConfig(num_rounds=5)  # Total communication rounds
)

