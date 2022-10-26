from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import qubovert as qv

import general_superstaq as gss
from general_superstaq.typing import TSPJson, WareHouseJson


@dataclass
class TSPOutput:
    route: List[str]
    route_list_numbers: List[int]
    total_distance: float
    map_link: List[str]
    qubo: qv.QUBO


def read_json_tsp(json_dict: TSPJson) -> TSPOutput:
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
    qubo = gss.qubo.convert_model_to_qubo(json_dict["qubo"])
    return TSPOutput(route, route_list_numbers, total_distance, map_links, qubo)


@dataclass
class WarehouseOutput:
    warehouse_to_destination: List[Tuple[str, str]]
    total_distance: float
    map_link: str
    open_warehouses: List[str]
    qubo: qv.QUBO


def read_json_warehouse(json_dict: WareHouseJson) -> WarehouseOutput:
    """
    Reads out returned JSON from SuperstaQ API's warehouse endpoint.
    Args:
        json_dict: a JSON dictionary matching the format returned by /warehouse endpoint
    Returns:
        a WarehouseOutput object with the optimal assignment.
    """
    warehouse_to_destination = json_dict["warehouse_to_destination"]
    total_distance = json_dict["total_distance"]
    map_link = json_dict["map_link"]
    open_warehouses = json_dict["open_warehouses"]
    qubo = gss.qubo.convert_model_to_qubo(json_dict["qubo"])
    return WarehouseOutput(
        warehouse_to_destination, total_distance, map_link, open_warehouses, qubo
    )


class Logistics:
    def __init__(self, client: gss.superstaq_client._SuperstaQClient):
        self._client = client

    def tsp(self, locs: List[str], solver: str = "anneal") -> TSPOutput:
        """
        This function solves the traveling salesperson problem (TSP) and
        takes a list of strings as input. TSP finds the shortest tour that
        traverses all locations in a list.
        Each string should be an addresss or name of a landmark
        that can pinpoint a location as a Google Maps search.
        It is assumed that the first string in the list is
        the starting and ending point for the TSP tour.
        The function returns a TSPOutput object.
        Args:
            locs: List of strings where each string represents
            a location needed to be visited on tour.
            solver: A string indicating which solver to use ("rqaoa" or "anneal").
        Returns:
            A TSPOutput object with the following attributes:
            .route: The optimal TSP tour as a list of strings in order.
            .route_list_numbers: The indicies in locs of the optimal tour.
            .total_distance: The tour's total distance.
            .map_link: A link to google maps that shows the tour.
            .qubo: The qubo representation of the TSP problem
        """
        input_dict = {"locs": locs}
        json_dict = self._client.tsp(input_dict)

        return read_json_tsp(json_dict)

    def warehouse(
        self, k: int, possible_warehouses: List[str], customers: List[str], solver: str = "anneal"
    ) -> WarehouseOutput:
        """
        This function solves the warehouse location problem, which is:
        given a list of customers to be served and a list of possible warehouse
        locations, find the optimal k warehouse locations such that the sum of
        the distances to each customer from the nearest facility is minimized.
        Args:
            k: An integer representing the number of warehouses in the solution.
            possible_warehouses: A list of possible warehouse locations.
            customers: A list of customer locations.
            solver: A string indicating which solver to use ("rqaoa" or "anneal").
        Returns:
            A WarehouseOutput object with the following attributes:
            .warehouse_to_destination: The optimal warehouse-customer pairings in List(Tuple) form.
            .total_distance: The total distance among all warehouse-customer pairings.
            .map_link: A link to google maps that shows the warehouse-customer pairings.
            .open_warehouses: A list of all warehouses that are open.
            .qubo: The qubo representation of the warehouse problem
        """
        input_dict: Dict[str, Union[int, List[str], str]] = {
            "k": k,
            "possible_warehouses": possible_warehouses,
            "customers": customers,
            "solver": solver,
        }
        json_dict = self._client.warehouse(input_dict)
        return read_json_warehouse(json_dict)
