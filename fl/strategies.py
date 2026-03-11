import flwr as fl
from flwr.common import Metrics
from typing import List, Dict

# ── Metric Aggregation Logic ───────────────────────────────────────────────────────
def weighted_average(metrics: List[tuple[int, Metrics]]) -> Metrics:
    """
    Dynamically aggregates all metrics (AUROC, AUPRC, etc.) from clients.
    Weights each metric by the number of samples (num_examples).
    """
    if not metrics:
        return {}
    
    # Get a list of all unique metric keys present in the first client's results
    # (e.g., ['auroc', 'auprc', 'sensitivity'])
    all_keys=metrics[0][1].keys()
    aggregated_metrics = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    for key in all_keys:
        # Check if the value is numeric before trying to average it
        val = metrics[0][1][key]
        if isinstance(val, (int, float)):
            weighted_sum = sum(num_examples * m[key] for num_examples, m in metrics if key in m)
            aggregated_metrics[key] = weighted_sum / total_examples
        else:
            aggregated_metrics[key] = val

    return aggregated_metrics

class Strategy(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy
    Integrates dynamic metric aggregation for the PTB-XL results.
    """
    def __init__(self, fraction_fit, fraction_evaluate, min_fit_clients, min_available_clients, on_fit_config_fn):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=on_fit_config_fn,
        )