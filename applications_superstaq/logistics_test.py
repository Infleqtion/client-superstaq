import qubovert as qv

import applications_superstaq


def test_read_json_tsp() -> None:
    route = ["Chicago", "St Louis", "St Paul", "Chicago"]
    route_list_numbers = [0, 1, 2, 0]
    total_distance = 100.0
    map_link = ["maps.google.com"]
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict = {
        "route": route,
        "route_list_numbers": route_list_numbers,
        "total_distance": total_distance,
        "map_link": map_link,
        "qubo": applications_superstaq.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert applications_superstaq.logistics.read_json_tsp(
        json_dict
    ) == applications_superstaq.logistics.TSPOutput(
        route, route_list_numbers, total_distance, map_link, qubo_obj
    )


def test_read_json_warehouse() -> None:
    warehouse_to_destination = [("Chicago", "Rockford"), ("Chicago", "Aurora")]
    total_distance = 100.0
    map_link = "map.html"
    open_warehouses = ["Chicago"]
    qubo_obj = qv.QUBO({("0", "1"): -1.0})
    json_dict = {
        "warehouse_to_destination": warehouse_to_destination,
        "total_distance": total_distance,
        "map_link": map_link,
        "open_warehouses": open_warehouses,
        "qubo": applications_superstaq.qubo.convert_qubo_to_model(qubo_obj),
    }
    assert applications_superstaq.logistics.read_json_warehouse(
        json_dict
    ) == applications_superstaq.logistics.WarehouseOutput(
        warehouse_to_destination, total_distance, map_link, open_warehouses, qubo_obj
    )
