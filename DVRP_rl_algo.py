"""Deployment der trainierten RL-Policy als Routing-Backend fuer dvrpsim.

Spiegelt die oeffentliche Form von DVRP_algo.py, damit DVRP_environment.py mit
einer Zeile zwischen OR-Tools und RL umschalten kann:

    routing_algorithm(state, locations)          <- Einstiegspunkt fuer dvrpsim
    solve_vrp_with_rl(...)                       <- statt solve_vrp_with_ortools
    convert_rl_solution_to_DVRP(...)             <- statt convert_ortools_solution_to_dvrp

Die Policy wird nicht hier trainiert, sondern von RL_dynamic_solution.py als
.pt-Datei exportiert. Welche dieser Dateien geladen wird, steht unten in
MODEL_FILE -- Training und Deployment sind damit getrennte Laeufe, und mehrere
trainierte Modelle koennen nebeneinander liegen bleiben.

Der Zustandsuebergang dvrpsim -> RL benutzt exakt dieselbe Logik wie OR-Tools:
Startknoten ist der fest zugesagte naechste Halt (bei EN_ROUTE) bzw. der aktuelle
Standort, und die Restdistanz dorthin entspricht dem Zuschlag, den OR-Tools auf
die erste Kante des Fahrzeugs rechnet (extra_start_distance in DVRP_algo.py).
"""

import glob
import os
import time
from typing import Any

import numpy as np

import DVRP_algo
from DVRP_algo import travel_time_calc
import RL_dynamic_solution as RL

solve_times = []
reassigning_rates = []
objective_values = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Welche exportierte Policy geladen wird. Bewusst hier statt aus
# RL_dynamic_solution: Training und Auswertung sind getrennte Laeufe, und ein
# neuer Trainingslauf soll nicht mitbestimmen, welches Modell ausgewertet wird.
MODEL_FILE = "rl_policy_250_seed1.pt"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

_policy = None      # lazy geladen, damit der Import ohne Modelldatei nicht bricht
_policy_config = None


def set_model_path(path: str) -> str:
    """Waehlt die zu ladende Policy-Datei (Dateiname oder vollstaendiger Pfad).

    Verwirft die gecachte Policy, sonst bliebe im selben Prozess das zuerst
    geladene Modell aktiv -- relevant, wenn mehrere Modelle nacheinander
    ausgewertet werden.
    """
    global MODEL_FILE, MODEL_PATH, _policy, _policy_config, _size_checked
    MODEL_PATH = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    MODEL_FILE = os.path.basename(MODEL_PATH)
    _policy = None
    _policy_config = None
    _size_checked = False   # neues Modell -> Groessenpruefung wieder scharf
    return MODEL_PATH


def get_policy():
    """Laedt die unter MODEL_PATH liegende Policy beim ersten Aufruf und cached sie."""
    global _policy, _policy_config
    if _policy is None:
        try:
            _policy, _policy_config = RL.load_policy(MODEL_PATH)
        except FileNotFoundError:
            vorhanden = sorted(os.path.basename(p)
                               for p in glob.glob(os.path.join(BASE_DIR, "*.pt")))
            raise FileNotFoundError(
                f"Keine trainierte Policy unter {MODEL_PATH}. "
                f"Vorhandene Modelldateien: {vorhanden if vorhanden else 'keine'}. "
                f"MODEL_FILE in DVRP_rl_algo.py anpassen oder erst "
                f"'python RL_dynamic_solution.py' laufen lassen."
            )
        print(f"RL-Policy geladen: {MODEL_PATH} ({_policy_config})")
    return _policy


_size_checked = False


def _check_instance_size(num_customers: int) -> None:
    """Warnt, wenn das geladene Modell fuer eine ganz andere Groesse trainiert wurde.

    load_policy prueft nur die Normierungskonstanten -- die haengen nicht von der
    Kundenzahl ab, ein 20er-Modell wuerde auf einer 250er-Instanz also still
    durchlaufen.

    Geprueft wird ausschliesslich der erste Entscheidungspunkt: dort ist die
    Instanz vollstaendig. Spaeter schrumpft sie mit jeder Auslieferung, eine
    Abweichung waere dann normal und keine Warnung wert.
    """
    global _size_checked
    if _size_checked:
        return
    _size_checked = True

    trained = _policy_config.get('num_customers') if _policy_config else None
    if not trained or not num_customers:
        return
    if not (1 / 1.5 <= num_customers / trained <= 1.5):
        print(f"Warnung: Policy wurde auf {trained} Kunden trainiert, die Instanz hat "
              f"hier {num_customers}. Passt MODEL_FILE zur Simulationsgroesse?")


def build_rl_start_state(locations: dict, distance_location: dict, state: dict,
                         vehicle_ids: list[str]) -> tuple[list[str], np.ndarray, list[int], list[float]]:
    """Uebersetzt den dvrpsim-Zustand in Startknoten + Restdistanzen fuer das RL.

    Gleiche Logik wie create_data_model/extra_start_distance in DVRP_algo.py:
      EN_ROUTE -> Startknoten ist next_visits[0], Restdistanz = Gesamtkante minus
                  bereits gefahrener Anteil (travel_time == euklidische Distanz,
                  da der Stochastik-Faktor in DVRP_vehicle.py auskommentiert ist)
      sonst    -> Startknoten ist current_visit, Restdistanz 0

    Returns: (location_ids, coords, start_nodes, remaining_dists)
    """
    location_ids = list(locations.keys())        # DEPOT steht an Index 0
    coord_source = dict(locations)

    start_locs, remaining_dists = [], []
    for vehicle_id in vehicle_ids:
        vehicle = state['vehicles'][vehicle_id]

        if vehicle['status'] == 'EN_ROUTE':
            start_loc = vehicle['next_visits'][0]['location']
            origin = vehicle['previous_visit']['location']
            traveled = state['time'] - vehicle['previous_visit']['departure_time']
            full_edge = travel_time_calc(distance_location[origin],
                                         distance_location[start_loc])
            remaining = max(0.0, full_edge - traveled)
        else:
            start_loc = vehicle['current_visit']['location']
            remaining = 0.0

        # Defensiv: ein Startort, der keinen offenen Auftrag mehr traegt, steht
        # nicht in routing_locations. OR-Tools liefe hier in einen ValueError von
        # .index() -- hier wird der Knoten stattdessen ergaenzt.
        if start_loc not in coord_source:
            coord_source[start_loc] = distance_location[start_loc]
            location_ids.append(start_loc)

        start_locs.append(start_loc)
        remaining_dists.append(remaining)

    coords = np.array([[coord_source[loc_id].x, coord_source[loc_id].y]
                       for loc_id in location_ids], dtype=np.float32)
    start_nodes = [location_ids.index(loc) for loc in start_locs]

    return location_ids, coords, start_nodes, remaining_dists


def solve_vrp_with_rl(locations: dict, distance_location: dict, state: dict,
                      num_vehicles: int) -> list[list[str]]:
    """Loest das VRP mit der trainierten Policy und gibt Routen als Location-IDs zurueck.

    Gleiches Rueckgabeformat wie solve_vrp_with_ortools, z.B.
        [['DEPOT', 'CUSTOMER 1', 'DEPOT'], ['CUSTOMER 4', 'DEPOT']]
    Die erste Position ist der fest zugesagte Startknoten des Fahrzeugs.
    """
    policy = get_policy()
    vehicle_ids = list(state['vehicles'].keys())[:num_vehicles]

    location_ids, coords, start_nodes, remaining_dists = build_rl_start_state(
        locations, distance_location, state, vehicle_ids
    )
    _check_instance_size(len(location_ids) - 1)

    env = RL.MultiVehicleVRPEnvironment(
        num_customers=len(location_ids) - 1,
        num_vehicles=num_vehicles,
        # Beeinflusst den greedy Rollout nicht (nur total_cost), wird aber aus der
        # Modell-Config genommen, damit Deployment und Training konsistent bleiben.
        span_coefficient=_policy_config.get('span_coefficient', 10.0),
        fixed_coords=coords,
    )

    start_time = time.perf_counter()
    rl_state = env.reset_from(start_nodes, remaining_dists)
    routes_indices, _, _ = RL.greedy_rollout(policy, env, initial_state=rl_state)
    solve_times.append(time.perf_counter() - start_time)
    #print(routes_indices)
    # Gleiche Formel wie OR-Tools' ObjectiveValue: Distanz + span_coefficient * laengste Route
    objective_values.append(env.total_cost())

    routes_ids = [[location_ids[idx] for idx in route] for route in routes_indices]
    return routes_ids


def convert_rl_solution_to_DVRP(routes: list[list[str]], state: dict,
                                locations: dict) -> dict[str, Any]:
    """Konvertiert die RL-Routen ins DVRP-Format.

    Da solve_vrp_with_rl dasselbe Format liefert wie solve_vrp_with_ortools, ist
    die Konvertierung solverunabhaengig und wird bewusst geteilt: das garantiert
    identische Semantik (insbesondere den Sonderfall 'en route zum Depot mit
    geladenen Auftraegen') und isoliert den Unterschied zwischen den beiden
    Pipelines sauber auf den Solver.
    """
    return DVRP_algo.convert_ortools_solution_to_dvrp(routes, state, locations)


def routing_algorithm(state: dict[str, Any], locations: dict) -> dict[str, Any]:
    """Hauptroutingalgorithmus mit der trainierten RL-Policy.

    Aufbau identisch zu DVRP_algo.routing_algorithm -- nur der Solver ist getauscht.
    """
    # Reassigning Rate: Truck-Zuordnung VOR dieser Neuplanung aus dem State bauen
    before_assignment = {}
    for order_id, order_info in state['open_orders'].items():
        if order_info['assigned_vehicle'] is not None and order_info['pickup_vehicle'] is None:
            before_assignment[order_id] = order_info['assigned_vehicle']

    undelivered_orders = list(state['open_orders'].keys())
    if len(undelivered_orders) == 0:
        return {'vehicles': {}, 'orders': {}}

    all_vehicles = list(state['vehicles'].keys())
    if len(all_vehicles) == 0:
        return {
            'vehicles': {},
            'orders': {order_id: {'status': 'rejected'} for order_id in undelivered_orders}
        }

    # routing_locations enthalten nur locations welche noch nicht picked up sind
    unpicked_orders = [
        order_id for order_id in state['open_orders'].keys()
        if state['open_orders'][order_id]['pickup_vehicle'] is None
    ]

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

    num_vehicles = len(all_vehicles)
    routes = solve_vrp_with_rl(routing_locations, distance_location, state, num_vehicles)

    result = convert_rl_solution_to_DVRP(routes, state, locations)

    # Reassigning Rate: Truck-Zuordnung NACH dieser Neuplanung aus dem Ergebnis bauen
    after_assignment = {}
    for vehicle_id, vehicle_data in result['vehicles'].items():
        for visit in vehicle_data['next_visits']:
            if 'delivery_list' in visit:
                for order_id in visit['delivery_list']:
                    after_assignment[order_id] = vehicle_id

    common_orders = [o for o in after_assignment if o in before_assignment]
    if len(common_orders) > 0:
        changed_count = sum(
            1 for o in common_orders if after_assignment[o] != before_assignment[o]
        )
        reassigning_rates.append(changed_count / len(common_orders))

    return result