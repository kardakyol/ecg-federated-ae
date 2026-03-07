import flwr as fl
import argparse
from fl.flower_client import ECGClient
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.client import ClientApp

def main():
    # Sprint 1: Standard FedAvg
    # Sprint 2: TODO: Custom Strategy in fl/strategies.py to handle dynamic client selection (eg. 5 out of 10 clients)

    parser = argparse.ArgumentParser(description="Flower Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=50, help="Number of FL rounds (default:50)")
    parser.add_argument("--epochs", type=int, default=5, help="Local epochs")
    parser.add_argument("--model", type=str, default="vanilla", help="Model: vanillaAE, convAE or VAE")
    parser.add_argument("--clients", type=int, default=2, help="Number of virtual clients")
    args = parser.parse_args()

    def fit_config(server_round: int):
        return {
            "local_epochs": args.epochs,
            "model_type": args.model,
        }

    # Standard FedAVG Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=args.clients,
        min_available_clients=args.clients,
        on_fit_config_fn=fit_config,
    )

    def client_fn(context: Context):
        # Sprint 2: partition_id from Ghouse
        # partition_id = context.node_config["partition-id"]
        return ECGClient(
            client_id=context.node_config["partition-id"],
            model_type=args.model    
        ).to_client()
    
    client_app = ClientApp(client_fn = client_fn)

    def server_fn(context: Context):
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
            on_fit_config_fn=lambda _: {
                "local_epochs": args.epochs,
                "model_type": args.model
            },
        )
        config = ServerConfig(num_rounds=args.rounds)
        return ServerAppComponents(strategy=strategy, config=config)
    
    server_app = ServerApp(server_fn=server_fn)
    
    # Sprint 1: Simulation for 3 rounds
    # Sprint 2: TODO: Increase the num_rounds to 50 
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.clients,
        #client_fn=client_fn,
        #num_clients=args.clients,
        #config=fl.server.ServerConfig(num_rounds=args.rounds),
        #strategy=strategy,
    )

if __name__ == "__main__":
    main()