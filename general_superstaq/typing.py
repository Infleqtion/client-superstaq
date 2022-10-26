from typing import Any, Dict, List, Optional, Sized, Tuple

from typing_extensions import TypedDict  # compatible with both python 3.7 and 3.8

QuboDict = TypedDict(
    "QuboDict",
    {
        "keys": List[str],
        "value": float,
    },
)

QuboModel = List[QuboDict]


MaxSharpe = TypedDict(
    "MaxSharpe",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "best_sharpe_ratio": float,
        "qubo": Optional[QuboModel],
    },
)

MaxSharpeJson = TypedDict(
    "MaxSharpeJson",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "best_sharpe_ratio": float,
        "qubo": QuboModel,
    },
)

MinVol = TypedDict(
    "MinVol",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "qubo": QuboModel,
    },
)

MinVolJson = TypedDict(
    "MinVolJson",
    {
        "best_portfolio": List[str],
        "best_ret": float,
        "best_std_dev": float,
        "qubo": QuboModel,
    },
)

Warehouse = TypedDict(
    "Warehouse",
    {
        "warehouse_to_destination": List[str],
        "total_distance": float,
        "map_link": str,
        "open_warehouses": Sized,
        "qubo": QuboModel,
    },
)

WareHouseJson = TypedDict(
    "WareHouseJson",
    {
        "warehouse_to_destination": List[Tuple[str, str]],
        "total_distance": float,
        "map_link": str,
        "open_warehouses": List[str],
        "qubo": QuboModel,
    },
)

TSP = TypedDict(
    "TSP",
    {
        "route_list_numbers": List[str],
        "total_distance": List[float],
        "map_link": List[str],
        "route": List[str],
        "qubo": QuboModel,
    },
)
TSPJson = TypedDict(
    "TSPJson",
    {
        "route": List[str],
        "route_list_numbers": List[int],
        "total_distance": float,
        "map_link": List[str],
        "qubo": QuboModel,
    },
)


Job = TypedDict(
    "Job",
    {
        "job_id": str,
        "num_qubits": int,
        "status": str,
        "target": str,
        "compiled_circuit": str,
        "data": Optional[Dict[str, Any]],
        "samples": Optional[Dict[str, int]],
        "shots": Optional[int],
    },
)
