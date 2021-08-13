import logistics


def test_read_json_tsp() -> None:
    route = ["Chicago", "St Louis", "St Paul", "Chicago"]
    route_list_numbers = [0, 1, 2, 0]
    total_distance = 100.0
    map_link = ["maps.google.com"]
    json_dict = {
        "route": route,
        "route_list_numbers": route_list_numbers,
        "total_distance": total_distance,
        "map_link": map_link,
    }
    assert logistics.read_json_tsp(json_dict) == logistics.TSPOutput(
        route, route_list_numbers, total_distance, map_link
    )


def test_read_json_warehouse() -> None:
    warehouse_to_destination = [("Chicago", "Rockford"), ("Chicago", "Aurora")]
    total_distance = 100.0
    map_link = "map.html"
    open_warehouses = ["Chicago"]
    json_dict = {
        "warehouse_to_destination": warehouse_to_destination,
        "total_distance": total_distance,
        "map_link": map_link,
        "open_warehouses": open_warehouses,
    }
    assert logistics.read_json_warehouse(json_dict) == logistics.WarehouseOutput(
        warehouse_to_destination, total_distance, map_link, open_warehouses
    )
