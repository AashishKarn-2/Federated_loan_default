import flwr as fl


# Start Flower server
fl.server.start_server(
    server_address="127.168.1.69:8080",
    config=fl.server.ServerConfig(num_rounds=2),
)

