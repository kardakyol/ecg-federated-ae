import flwr as fl
from fl.flower_client import ECGClient
from flwr.common import Context

def main():
    # Sprint 1: Standard FedAvg
    # Sprint 2: TODO: Custom Strategy in fl/strategies.py to handle dynamic client selection (eg. 5 out of 10 clients)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    def client_fn(context: Context):
        # Sprint 2: partition_id from Ghouse
        # partition_id = context.node_config["partition-id"]
        return ECGClient(client_id=context.node_config["partition-id"]).to_client()
    
    # Sprint 1: Simulation for 3 rounds
    # Sprint 2: TODO: Increase the num_rounds to 50 
    fl.simulation.run_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()