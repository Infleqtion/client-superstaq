from typing import Any, Counter, Dict, List, Optional, Sized, TypedDict


MaxSharpe = TypedDict(
    "MaxSharpe",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "best_sharpe_ratio": float,
        "qubo": Optional[List[float]],
    },
)

MinVol = TypedDict(
    "MinVol",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "qubo": Optional[List[float]],
    },
)

Warehouse = TypedDict(
    "Warehouse",
    {
        "warehouse_to_destination": List[str],
        "total_distance": float,
        "map_link": str,
        "open_warehouses": Sized,
        "qubo": List[Dict[str, Any]],
    },
)

TSP = TypedDict(
    "TSP",
    {
        "route_list_numbers": List[str],
        "total_distance": List[float],
        "map_link": List[str],
        "route": List,
        "qubo": List[Dict[str, Any]],
    },
)

Job = TypedDict(
    "Job",
    {
        "job_id": str,
        "num_qubits": int,
        "status": str,
        "target": str,
        "data": Optional[Dict[str, Any]],
        "samples": Optional[Counter],
        "shots": Optional[int],
    },
)
