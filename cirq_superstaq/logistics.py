from dataclasses import dataclass
from typing import List


@dataclass
class TSPOutput:
    route: List[str]
    route_list_numbers: List
    total_distance: float
    map_link: List[str]


def read_json_tsp(json_dict: dict) -> TSPOutput:
    """Reads out returned JSON from SuperstaQ API's tsp endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /tsp endpoint
    Returns:
        a TSPOutput object with the optimal route.
    """

    route = json_dict["route"]
    route_list_numbers = json_dict["route_list_numbers"]
    total_distance = json_dict["total_distance"]
    map_links = json_dict["map_link"]
    return TSPOutput(route, route_list_numbers, total_distance, map_links)


@dataclass
class WarehouseOutput:
    warehouse_to_destination: List
    total_distance: float
    map_link: str
    open_warehouses: List


def read_json_warehouse(json_dict: dict) -> WarehouseOutput:
    """Reads out returned JSON from SuperstaQ API's warehouse endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /warehouse endpoint
    Returns:
        a WarehouseOutput object with the optimal assignment.
    """

    warehouse_to_destination = json_dict["warehouse_to_destination"]
    total_distance = json_dict["total_distance"]
    map_link = json_dict["map_link"]
    open_warehouses = json_dict["open_warehouses"]
    return WarehouseOutput(warehouse_to_destination, total_distance, map_link, open_warehouses)
