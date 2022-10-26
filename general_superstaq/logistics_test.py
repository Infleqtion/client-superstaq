from unittest import mock

import qubovert as qv

import general_superstaq as gss
from general_superstaq.typing import TSPJson, WareHouseJson


def test_read_json_tsp() -> None:
    route = ["Chicago", "St Louis", "St Paul", "Chicago"]
    route_list_numbers = [0, 1, 2, 0]
    total_distance = 100.0
    map_link = ["maps.google.com"]
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict: TSPJson = {
        "route": route,
        "route_list_numbers": route_list_numbers,
        "total_distance": total_distance,
        "map_link": map_link,
        "qubo": gss.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert gss.logistics.read_json_tsp(json_dict) == gss.logistics.TSPOutput(
        route, route_list_numbers, total_distance, map_link, qubo_obj
    )


def test_read_json_warehouse() -> None:
    warehouse_to_destination = [("Chicago", "Rockford"), ("Chicago", "Aurora")]
    total_distance = 100.0
    map_link = "map.html"
    open_warehouses = ["Chicago"]
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict: WareHouseJson = {
        "warehouse_to_destination": warehouse_to_destination,
        "total_distance": total_distance,
        "map_link": map_link,
        "open_warehouses": open_warehouses,
        "qubo": gss.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert gss.logistics.read_json_warehouse(json_dict) == gss.logistics.WarehouseOutput(
        warehouse_to_destination, total_distance, map_link, open_warehouses, qubo_obj
    )


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.tsp",
    return_value={
        "route": ["Chicago", "St Louis", "St Paul", "Chicago"],
        "route_list_numbers": [0, 1, 2, 0],
        "total_distance": 100.0,
        "map_link": ["maps.google.com"],
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_tsp(mock_tsp: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.logistics.Logistics(client)
    qubo = {("0",): 123}
    expected = gss.logistics.TSPOutput(
        ["Chicago", "St Louis", "St Paul", "Chicago"],
        [0, 1, 2, 0],
        100.0,
        ["maps.google.com"],
        qubo,
    )
    assert service.tsp(["Chicago", "St Louis", "St Paul"]) == expected


@mock.patch(
    "general_superstaq.superstaq_client._SuperstaQClient.warehouse",
    return_value={
        "warehouse_to_destination": [("Chicago", "Rockford"), ("Chicago", "Aurora")],
        "total_distance": 100.0,
        "map_link": "map.html",
        "open_warehouses": ["Chicago"],
        "qubo": [{"keys": ["0"], "value": 123}],
    },
)
def test_service_warehouse(mock_warehouse: mock.MagicMock) -> None:
    client = gss.superstaq_client._SuperstaQClient(
        remote_host="http://example.com", api_key="key", client_name="general_superstaq"
    )
    service = gss.logistics.Logistics(client)
    qubo = {("0",): 123}
    expected = gss.logistics.WarehouseOutput(
        [("Chicago", "Rockford"), ("Chicago", "Aurora")],
        100.0,
        "map.html",
        ["Chicago"],
        qubo,
    )
    assert service.warehouse(1, ["Chicago", "San Francisco"], ["Rockford", "Aurora"]) == expected
