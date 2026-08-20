from dvrpsim import Location
from dvrpsim.utils.distances import euclidean_distance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from typing import Any
from pprint import pprint
import time

solve_times = []
reassigning_rates = []       # gesammelte Reassigning-Rate-Werte
objective_values = []        # Zielfunktionswert je Entscheidungspunkt

def travel_time_calc(origin: Location, destination: Location) -> float:
    return euclidean_distance(origin.x, origin.y, destination.x, destination.y)


def create_distance_matrix(locations: dict) -> list[list[float]]:
    """Erstellt Distanzmatrix aus Locations"""
    keys = list(locations.keys())
    matrix = []
    for i in keys:
        row = []
        for j in keys:
            row.append(travel_time_calc(locations[i], locations[j]))
        matrix.append(row)
    return matrix


def create_data_model(locations: dict, state: dict, num_vehicles: int) -> dict[str, Any]:
    """Erstellt Datenmodell für OR-Tools"""
    data = {}
    data["distance_matrix"] = create_distance_matrix(locations)
    data["num_vehicles"] = num_vehicles
    
    data["starts"] = []
    for vehicle in range(num_vehicles):
        if state['vehicles'][f'TRUCK-{vehicle+1}']['status'] == 'EN_ROUTE':
            start_location = state['vehicles'][f'TRUCK-{vehicle+1}']['next_visits'][0]['location']
        else:
            start_location = state['vehicles'][f'TRUCK-{vehicle+1}']['current_visit']['location']
        start_location_index = list(locations.keys()).index(start_location)
        data["starts"].append(start_location_index)
        
    data["ends"] = [0 for _ in range(num_vehicles)]
    data["location_ids"] = list(locations.keys())
    return data


def extract_solution(data: dict, manager, routing, solution) -> list[list[int]]:
    """Extrahiert Routen aus OR-Tools Lösung"""
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            index = solution.Value(routing.NextVar(index))
        # Füge Endpunkt hinzu (zurück zum Depot)
        route.append(manager.IndexToNode(index))
        routes.append(route)
    #print(routes)
    return routes

def print_distance(data, manager, routing, solution):
    """Print ditance for every vehicle on console."""
    print(f"Objective: {solution.ObjectiveValue()}\n")


    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            print(f"Distance for TRUCK-{vehicle_id + 1}: unused\n")
            continue
        
        index = routing.Start(vehicle_id)
        plan_output = f"Distance for TRUCK-{vehicle_id + 1}:\n"
        route_distance = 0

        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)


def solve_vrp_with_ortools(locations: dict, distance_location: dict, state: dict, num_vehicles: int) -> list[list[str]]:
    """
    Löst VRP mit OR-Tools und gibt Routen als Location-IDs zurück
    
    Returns:
        Liste von Routen, z.B. [['DEPOT', 'CUSTOMER 1', 'DEPOT'], ['DEPOT', 'CUSTOMER 2', 'DEPOT']]
    """
    #if state['time'] == 0:
    #    pprint(state)

    # Erstelle Datenmodell
    data = create_data_model(locations, state, num_vehicles)
    
    # Erstelle Routing Manager
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), 
        data["num_vehicles"], 
        data["starts"],
        data["ends"],
    )
    
    # Erstelle Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    extra_start_distance = dict() # for example {0 : 250, 2 : 439, 5 : 234}

    for vehicle_idx in range(num_vehicles):
        vehicle_id = f'TRUCK-{vehicle_idx+1}'
        if state['vehicles'][vehicle_id]['status'] == 'EN_ROUTE':
            distance_traveled = state['time'] - state['vehicles'][vehicle_id]['previous_visit']['departure_time']
            origin = state['vehicles'][vehicle_id]['previous_visit']['location']
            destination = state['vehicles'][vehicle_id]['next_visits'][0]['location']
            travel_time = travel_time_calc(distance_location[origin], distance_location[destination])
            remaining_distance = travel_time - distance_traveled
            extra_start_distance[vehicle_idx] = remaining_distance
    #print(extra_start_distance)

    transit_indices = []

    def make_callback(vehicle_id, extra_cost):
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            cost = data["distance_matrix"][from_node][to_node]

            # Zuschlag auf die erste Kante des Fahrzeugs
            if extra_cost > 0 and from_index == routing.Start(vehicle_id):
                cost += extra_cost

            return int(round(cost))

        return distance_callback

    for vehicle_id in range(data['num_vehicles']):
        extra_cost = extra_start_distance.get(vehicle_id, 0)

        transit_callback_index = routing.RegisterTransitCallback(
            make_callback(vehicle_id, extra_cost)
        )

        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle_id)
        transit_indices.append(transit_callback_index)

    # Füge Distanz-Dimension hinzu
    dimension_name = "Distance"
    routing.AddDimensionWithVehicleTransits(
        transit_indices,
        0,  # kein Slack
        300000000000,  # maximale Fahrzeugdistanz
        False,  # Do not force start cumul to zero marker
        dimension_name,
    )

    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(10)
    
    # Suchparameter
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = 60
    
    # Löse Problem
    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    solve_duration = time.perf_counter() - start_time
    solve_times.append(solve_duration)
    #print(f"OR-Tools Lösung in {solve_duration:.4f} Sekunden gefunden")
    
    if solution:
        objective_values.append(solution.ObjectiveValue())

        # Extrahiere Routen als Indizes
        routes_indices = extract_solution(data, manager, routing, solution)
        
        # Konvertiere Indizes zu Location-IDs
        routes_ids = []
        for route in routes_indices:
            route_ids = [data["location_ids"][idx] for idx in route]
            routes_ids.append(route_ids)
        return routes_ids
    else:
        print("Keine Lösung gefunden!")
        return []


def convert_ortools_solution_to_dvrp(routes: list[list[str]], state: dict, 
                                     locations: dict) -> dict[str, Any]:
    """
    Konvertiert OR-Tools Routen ins DVRP-Format
    
    Args:
        routes: Liste von Routen mit Location-IDs
        state: Aktueller Zustand des DVRP
        locations: Dictionary aller Locations
    
    Returns:
        Dictionary im DVRP-Format für vehicles und orders
    """
    #pprint(routes)
    # Sammle undelivered_orders Aufträge
    undelivered_orders = [
        order_id for order_id in state['open_orders'].keys()
    ]
    
    # Sammle alle Fahrzeuge
    all_vehicles = [
        vehicle_id for vehicle_id in state['vehicles'].keys()
    ]

    # Mappe undelivered_orders zu ihren Pickup-Locations
    order_location_map = {}
    for order_id in undelivered_orders:
        pickup_loc = state['open_orders'][order_id]['pickup_location']
        order_location_map[pickup_loc] = order_id # For example: {'Customer 1' : '0-1'}
    
    # Erstelle Fahrzeugrouten aus OR-Tools Lösung
    vehicles_dict = {}
    assigned_orders = set()
    
    for vehicle_idx, route in enumerate(routes):
        if vehicle_idx >= len(all_vehicles):
            break

        vehicle_id = all_vehicles[vehicle_idx]

        # Überspringe wenn Route nur Depot enthält
        if not route or len(route) <= 2 and all(loc == 'DEPOT' for loc in route):
            continue

        vehicle_route = []
        orders_in_route = []

        # En-route zum Depot mit bereits geladenen Aufträgen: der erste Halt
        # (next_visits[0]) ist fest zugesagt und darf nicht verändert werden,
        # daher wird er 1:1 als eigener Delivery-Stop übernommen; die
        # geladenen Aufträge werden NICHT nochmal am Routenende ausgeliefert.
        en_route_to_depot_with_load = bool(
            state['vehicles'][vehicle_id]['next_visits']
            and state['vehicles'][vehicle_id]['loaded_orders']
            and state['vehicles'][vehicle_id]['next_visits'][0]['location'] == 'DEPOT'
            and route[0] == 'DEPOT'
        )

        if en_route_to_depot_with_load:
            vehicle_route.append({
                'location': 'DEPOT',
                'delivery_list': list(state['vehicles'][vehicle_id]['loaded_orders'])
            })
            route_to_process = route[1:]
        else:
            if state['vehicles'][vehicle_id]['loaded_orders']:
                orders_in_route.extend(state['vehicles'][vehicle_id]['loaded_orders'])
            route_to_process = route

        # Verarbeite Route
        for loc_id in route_to_process:

            if state['vehicles'][vehicle_id]['current_visit'] is not None:
                if loc_id == state['vehicles'][vehicle_id]['current_visit']['location']:
                    if loc_id in order_location_map:
                        order_id = order_location_map[loc_id]
                        orders_in_route.append(order_id)
                        assigned_orders.add(order_id)
                    continue

            if loc_id in order_location_map:
                order_id = order_location_map[loc_id]
                orders_in_route.append(order_id)
                assigned_orders.add(order_id)

                # Pickup
                vehicle_route.append({
                    'location': loc_id,
                    'pickup_list': [order_id]
                })

        # Delivery am Depot dabei aber auf schon vorhandene Lieferungen prüfen
        if orders_in_route:
            vehicle_route.append({
                'location': 'DEPOT',
                'delivery_list': orders_in_route
            })

        if vehicle_route:
            vehicles_dict[vehicle_id] = {
                'next_visits': vehicle_route
            }

    # Erstelle Orders Dictionary
    orders_dict = {}

    unpicked_orders = [
        order_id for order_id in state['open_orders'].keys()
        if state['open_orders'][order_id]['pickup_vehicle'] is None
    ]

    for order_id in unpicked_orders:
        orders_dict[order_id] = {'status': 'accepted'}
    
    return {
        'vehicles': vehicles_dict,
        'orders': orders_dict
    }


def routing_algorithm(state: dict[str, Any], locations: dict) -> dict[str, Any]:
    """
    Hauptroutingalgorithmus mit OR-Tools
    """
    # Reassigning Rate: Truck-Zuordnung VOR dieser Neuplanung aus dem State bauen
    before_assignment = {}
    for order_id, order_info in state['open_orders'].items():
        if order_info['assigned_vehicle'] is not None and order_info['pickup_vehicle'] is None:
            before_assignment[order_id] = order_info['assigned_vehicle']
    # Sammle nicht delivered Aufträge
    undelivered_orders = [
        order_id for order_id in state['open_orders'].keys()
    ]
    #print(undelivered_orders)
    
    if len(undelivered_orders) == 0:
        return {'vehicles': {}, 'orders': {}}
    
    # Sammle alle Fahrzeuge
    all_vehicles = [
        vehicle_id for vehicle_id in state['vehicles'].keys()
    ]
    
    if len(all_vehicles) == 0:
        return {
            'vehicles': {},
            'orders': {
                order_id: {'status': 'rejected'} 
                for order_id in undelivered_orders
            }
        }
    
    # routing_locations enthalten nur locations welche noch nicht picked up sind
    unpicked_orders = [
        order_id for order_id in state['open_orders'].keys()
        if state['open_orders'][order_id]['pickup_vehicle'] is None
    ]

    # Erstelle Locations für Routing (nur relevante)
    routing_locations = {'DEPOT': locations['DEPOT']}
    for order_id in unpicked_orders:
        pickup_loc = state['open_orders'][order_id]['pickup_location']
        if pickup_loc in locations:
            routing_locations[pickup_loc] = locations[pickup_loc]
    
    distance_location = {'DEPOT': locations['DEPOT']}
    for order_id in undelivered_orders:
        pickup_loc = state['open_orders'][order_id]['pickup_location']
        if pickup_loc in locations:
            distance_location[pickup_loc] = locations[pickup_loc]

    # Löse mit OR-Tools
    num_vehicles = len(all_vehicles)
    routes = solve_vrp_with_ortools(routing_locations, distance_location, state, num_vehicles)
    
    # Konvertiere zu DVRP Format
    result = convert_ortools_solution_to_dvrp(routes, state, locations)

    # Reassigning Rate: Truck-Zuordnung NACH dieser Neuplanung aus dem Ergebnis bauen
    after_assignment = {}
    for vehicle_id, vehicle_data in result['vehicles'].items():
        for visit in vehicle_data['next_visits']:
            if 'delivery_list' in visit:
                for order_id in visit['delivery_list']:
                    after_assignment[order_id] = vehicle_id
    common_orders = []
    for order_id in after_assignment:
        if order_id in before_assignment:
            common_orders.append(order_id)

    if len(common_orders) > 0:
        changed_count = 0
        for order_id in common_orders:
            if after_assignment[order_id] != before_assignment[order_id]:
                changed_count += 1

        reassigning_rate = changed_count / len(common_orders)
        reassigning_rates.append(reassigning_rate)

    return result