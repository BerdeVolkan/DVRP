"""Capacited Vehicles Routing Problem (CVRP).

Dieselbe statische Instanz (17 Knoten, 4 Fahrzeuge, Kapazitaet 15) wird mit zwei
Verfahren geloest und verglichen:

    1) OR-Tools  -- klassischer Solver, pro Instanz ~10 s Rechenzeit
    2) REINFORCE -- Policy wird offline auf einer Verteilung aehnlicher Instanzen
                    trainiert und danach in Millisekunden angewendet
                    (Vorgehen der Neural Combinatorial Optimization,
                     vgl. Bello et al. 2016, Nazari et al. 2018, Kool et al. 2019)

Die Distanzmatrix unten ist exakt die Manhattan-Distanz ueber data["locations"];
alle Punkte liegen auf einem 9x9-Raster mit dem Depot im Zentrum. Genau daraus
erzeugt sample_instance() die Trainingsverteilung, sodass die Zielinstanz eine
gewoehnliche Ziehung dieser Verteilung ist. main() prueft die Aequivalenz von
Matrix und Koordinaten per assert.
"""

import time

import numpy as np
import torch
import torch.optim as optim
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from RL_static_solution import SimplePolicy


# Hyperparameter der RL-Methode -- zum Hochdrehen hier anpassen.
NUM_FEATURES = 9          # Spalten pro (Fahrzeug, Knoten)-Kombination
UNSERVED_PENALTY = 2.0    # Strafe je unbedienter Nachfrageeinheit (normierte Distanz)
RL_SEED = 0
RL_EPOCHS = 400
RL_BATCH_SIZE = 32
RL_LEARNING_RATE = 1e-3
RL_SAMPLES = 1000         # stochastische Rollouts auf der Zielinstanz


def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        # fmt: off
      [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
      [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
      [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
      [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
      [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
      [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
      [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
      [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
      [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
      [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
      [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
      [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
      [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
      [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
      [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
      [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
      [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
        # fmt: on
    ]
    # Koordinaten hinter der Matrix: manhattan_matrix(locations) == distance_matrix.
    data["locations"] = [
        # fmt: off
      (456, 320),                        # 0 = Depot, exakt im Rasterzentrum
      (228, 0), (912, 0), (0, 80), (114, 80), (570, 160), (798, 160), (342, 240),
      (684, 240), (570, 400), (912, 400), (114, 480), (228, 480), (342, 560),
      (684, 560), (0, 640), (798, 640),
        # fmt: on
    ]
    data["grid_x"] = [114 * i for i in range(9)]
    data["grid_y"] = [80 * i for i in range(9)]
    data["demands"] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    data["vehicle_capacities"] = [15, 15, 15, 15]
    data["num_vehicles"] = 4
    data["depot"] = 0
    return data


def manhattan_matrix(locations):
    """Manhattan-Distanzmatrix ueber eine Liste von (x, y)-Punkten."""
    return [
        [abs(x_from - x_to) + abs(y_from - y_to) for x_to, y_to in locations]
        for x_from, y_from in locations
    ]


# ---------------------------------------------------------------------------
# Gemeinsame Ausgabe -- beide Verfahren nutzen dasselbe Format und dieselbe
# Distanzformel, damit die Zahlen direkt vergleichbar sind.
# ---------------------------------------------------------------------------
def print_routes(data, routes, title):
    """Prints routes on console and returns the total distance."""
    matrix = data["distance_matrix"]
    demands = data["demands"]

    print(title)
    total_distance = 0
    total_load = 0
    for vehicle_id, route in enumerate(routes):
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        for position, node in enumerate(route):
            route_load += demands[node]
            if position > 0:
                route_distance += matrix[route[position - 1]][node]
            plan_output += f" {node} Load({route_load})"
            plan_output += " -> " if position < len(route) - 1 else "\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")
    return total_distance


def print_solution(data, manager, routing, solution, title):
    """Prints solution on console and returns the total distance."""
    print(f"Objective: {solution.ObjectiveValue()}")
    routes = []
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return print_routes(data, routes, title)


# ---------------------------------------------------------------------------
# 1) OR-Tools
# ---------------------------------------------------------------------------
def solve_with_ortools(data, time_limit=10):
    """Solve the CVRP problem. Returns (total distance, wall clock seconds)."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit)

    # Solve the problem.
    started = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    elapsed = time.perf_counter() - started

    # Print solution on console.
    if not solution:
        print("OR-Tools hat keine Loesung gefunden.")
        return None, elapsed
    total_distance = print_solution(
        data, manager, routing, solution, "Touren laut OR-Tools:"
    )
    return total_distance, elapsed


# ---------------------------------------------------------------------------
# 2) Reinforcement Learning -- Instanzen
# ---------------------------------------------------------------------------
def target_instance(data):
    """Die Originalinstanz im Instanzformat der RL-Umgebung."""
    return {
        "locations": list(data["locations"]),
        "demands": list(data["demands"]),
        "distance_matrix": data["distance_matrix"],
        "vehicle_capacities": list(data["vehicle_capacities"]),
        "num_vehicles": data["num_vehicles"],
        "depot": data["depot"],
    }


def sample_instance(rng, data):
    """Zieht eine Instanz derselben Verteilung wie die Zielinstanz.

    Depot bleibt im Rasterzentrum, die Kunden werden ohne Zuruecklegen aus den
    uebrigen Rasterpunkten gezogen. Die Nachfragen sind eine Permutation des
    Original-Multisets -- die Summe bleibt damit immer 60 und die Kapazitaet
    exakt ausgelastet, also dieselbe Schwierigkeitsklasse wie das Ziel.
    """
    depot_index = data["depot"]
    depot = data["locations"][depot_index]
    grid = [(x, y) for x in data["grid_x"] for y in data["grid_y"] if (x, y) != depot]

    customer_demands = data["demands"][:depot_index] + data["demands"][depot_index + 1:]
    chosen = rng.choice(len(grid), size=len(customer_demands), replace=False)

    locations = [depot] + [grid[i] for i in chosen]
    demands = [0] + [int(d) for d in rng.permutation(customer_demands)]
    return {
        "locations": locations,
        "demands": demands,
        "distance_matrix": manhattan_matrix(locations),
        "vehicle_capacities": list(data["vehicle_capacities"]),
        "num_vehicles": data["num_vehicles"],
        "depot": 0,
    }


# ---------------------------------------------------------------------------
# 2) Reinforcement Learning -- Umgebung
# ---------------------------------------------------------------------------
class CVRPEnvironment:
    """CVRP-Umgebung: K Fahrzeuge, eine Tour pro Fahrzeug, kein Nachladen.

    Die Instanz kommt aus einem Callable -- im Training zieht es jede Epoche eine
    neue Instanz, in der Auswertung liefert es immer die Zielinstanz.
    """

    def __init__(self, instance_sampler):
        self.instance_sampler = instance_sampler

    def reset(self):
        instance = self.instance_sampler()
        self.locations = np.array(instance["locations"], dtype=np.float32)
        self.demand = np.array(instance["demands"], dtype=np.float32)
        self.matrix = np.array(instance["distance_matrix"], dtype=np.float32)
        self.depot = instance["depot"]
        self.num_vehicles = instance["num_vehicles"]

        self.dist_scale = float(self.matrix.max())
        self.capacity = float(max(instance["vehicle_capacities"]))
        coord_scale = self.locations.max(axis=0)
        self.coord_scale = np.where(coord_scale > 0, coord_scale, 1.0).astype(np.float32)

        K = self.num_vehicles
        self.vehicle_pos = np.full(K, self.depot, dtype=np.int64)
        self.cap_left = np.array(instance["vehicle_capacities"], dtype=np.float32)
        self.finished = np.zeros(K, dtype=bool)
        self.remaining = self.demand.copy()
        self.routes = [[self.depot] for _ in range(K)]
        self.total_distance = 0.0
        return self

    def feasibility_mask(self):
        """Form (K, n): mask[v, j] = 'Fahrzeug v faehrt zu Knoten j' erlaubt?"""
        mask = np.zeros((self.num_vehicles, len(self.remaining)), dtype=bool)
        for v in range(self.num_vehicles):
            if self.finished[v]:
                continue  # Tour beendet -- kein Nachladen, Fahrzeug faellt raus
            servable = (self.remaining > 0) & (self.remaining <= self.cap_left[v])
            servable[self.depot] = False
            if servable.any():
                mask[v] = servable
            else:
                # Depot erst, wenn kein Kunde mehr machbar ist. Das erzwingt
                # Weiterfahren statt fruehem Abbruch -- bei exakt ausgelasteter
                # Kapazitaet der entscheidende Hebel fuer Zulaessigkeit.
                mask[v, self.depot] = True
        return mask

    def step(self, vehicle, node):
        distance = float(self.matrix[self.vehicle_pos[vehicle], node])
        self.total_distance += distance

        if node == self.depot:
            self.finished[vehicle] = True
        else:
            # Volle Auslieferung -- die Maske hat die Kapazitaet bereits geprueft.
            self.cap_left[vehicle] -= self.remaining[node]
            self.remaining[node] = 0.0

        self.vehicle_pos[vehicle] = node
        self.routes[vehicle].append(node)
        return -distance / self.dist_scale

    def is_done(self):
        return bool(self.finished.all())

    def unserved_demand(self):
        return float(self.remaining.sum())


def build_features(env):
    """Feature-Matrix (K*n, 9): pro (Fahrzeug, Knoten)-Kombination
    [x, y, Restnachfrage, Distanz Fahrzeug->Knoten, Distanz Knoten->Depot,
     Restkapazitaet, Restkapazitaet nach Bedienung, ist_Depot, Tour_beendet]."""
    n = len(env.remaining)

    xy = env.locations / env.coord_scale
    demand_norm = env.remaining / env.capacity
    dist_to_depot = env.matrix[:, env.depot] / env.dist_scale
    is_depot = np.zeros(n, dtype=np.float32)
    is_depot[env.depot] = 1.0

    rows = []
    for v in range(env.num_vehicles):
        dist_from_vehicle = env.matrix[env.vehicle_pos[v]] / env.dist_scale
        cap_col = np.full(n, env.cap_left[v] / env.capacity, dtype=np.float32)
        # Restkapazitaet *nach* Bedienung -- macht "exakt vollmachen" sichtbar.
        cap_after = (env.cap_left[v] - env.remaining) / env.capacity
        finished_col = np.full(n, float(env.finished[v]), dtype=np.float32)
        rows.append(
            np.stack(
                [
                    xy[:, 0], xy[:, 1], demand_norm, dist_from_vehicle, dist_to_depot,
                    cap_col, cap_after, is_depot, finished_col,
                ],
                axis=1,
            )
        )

    return torch.from_numpy(np.concatenate(rows, axis=0).astype(np.float32))


def rollout(policy, env, greedy=False, track_grad=False):
    """Ein vollstaendiges Rollout. Gibt (log_probs, Distanz, unbediente Nachfrage)."""
    env.reset()
    n = len(env.remaining)
    log_probs = []

    # Jeder Schritt bedient entweder einen Kunden oder beendet eine Tour.
    max_steps = n + env.num_vehicles
    with torch.enable_grad() if track_grad else torch.no_grad():
        for _ in range(max_steps):
            mask = env.feasibility_mask()
            if not mask.any():
                break
            mask_flat = torch.from_numpy(mask.reshape(-1))
            probs = policy(build_features(env), mask_flat)

            if greedy:
                action = int(torch.argmax(probs).item())
            else:
                distribution = torch.distributions.Categorical(probs)
                sampled = distribution.sample()
                if track_grad:
                    log_probs.append(distribution.log_prob(sampled))
                action = int(sampled.item())

            env.step(action // n, action % n)
            if env.is_done():
                break

    return log_probs, env.total_distance, env.unserved_demand()


# ---------------------------------------------------------------------------
# 2) Reinforcement Learning -- Training und Anwendung
# ---------------------------------------------------------------------------
def train_rl(data, num_epochs=RL_EPOCHS, batch_size=RL_BATCH_SIZE,
             lr=RL_LEARNING_RATE, log_every=25, seed=RL_SEED):
    """REINFORCE mit Batch-Mittelwert-Baseline auf einer Instanzverteilung."""
    rng = np.random.default_rng(seed)
    # Pro Epoche eine Instanz, darauf batch_size Rollouts: so misst die Baseline
    # Policy-Qualitaet statt Instanzschwierigkeit. Die Instanz wechselt je Epoche.
    current = {}
    env = CVRPEnvironment(lambda: current["instance"])
    policy = SimplePolicy(num_features=NUM_FEATURES)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(num_epochs):
        current["instance"] = sample_instance(rng, data)
        batch_log_probs, batch_returns, distances = [], [], []
        num_feasible = 0

        for _ in range(batch_size):
            log_probs, distance, unserved = rollout(policy, env, track_grad=True)
            batch_log_probs.append(torch.stack(log_probs).sum())
            batch_returns.append(
                -distance / env.dist_scale - UNSERVED_PENALTY * unserved
            )
            distances.append(distance)
            num_feasible += unserved == 0

        returns = torch.tensor(batch_returns, dtype=torch.float32)
        advantage = returns - returns.mean()   # einfache Batch-Mittelwert-Baseline
        loss = -(torch.stack(batch_log_probs) * advantage).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:4d} | durchschn. Gesamtdistanz: {np.mean(distances):7.1f}m "
                  f"| zulaessig: {num_feasible / batch_size:6.1%} "
                  f"| loss: {loss.item():8.4f}")

    return policy


def greedy_rollout(policy, env):
    """Deterministische Anwendung: pro Schritt die wahrscheinlichste Aktion."""
    _, distance, unserved = rollout(policy, env, greedy=True)
    return [list(route) for route in env.routes], distance, unserved


def sampled_rollout(policy, env, num_samples=RL_SAMPLES):
    """Sampling at inference: viele stochastische Rollouts, bestes zulaessiges gewinnt."""
    best_routes, best_distance = None, None
    for _ in range(num_samples):
        _, distance, unserved = rollout(policy, env)
        if unserved == 0 and (best_distance is None or distance < best_distance):
            best_routes, best_distance = [list(r) for r in env.routes], distance
    return best_routes, best_distance


def print_comparison(rows, ortools_distance):
    """Vergleichstabelle: Distanz, Gap gegenueber OR-Tools, Zeit je Methode."""
    print(f"{'Methode':<26}{'Distanz':>12}{'Gap':>10}{'Zeit':>12}")
    print("-" * 60)
    for name, distance, seconds in rows:
        # Die Anwendungszeiten liegen Groessenordnungen auseinander -- unter einer
        # Sekunde in ms ausgeben, sonst faellt genau der Vorteil des RL unter den Tisch.
        elapsed = f"{seconds * 1000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"
        if distance is None:
            print(f"{name:<26}{'unzulaessig':>12}{'-':>10}{elapsed:>12}")
            continue
        gap = f"{(distance / ortools_distance - 1):+.1%}" if ortools_distance else "-"
        print(f"{name:<26}{f'{distance}m':>12}{gap:>10}{elapsed:>12}")


def main():
    data = create_data_model()
    assert manhattan_matrix(data["locations"]) == data["distance_matrix"], (
        "locations passen nicht zur distance_matrix -- beide Verfahren wuerden "
        "unterschiedliche Instanzen loesen"
    )

    print("=" * 60)
    print("1) OR-Tools")
    print("=" * 60)
    ortools_distance, ortools_time = solve_with_ortools(data)

    print()
    print("=" * 60)
    print("2) Reinforcement Learning (REINFORCE)")
    print("=" * 60)
    print(f"Training auf zufaelligen Instanzen derselben Verteilung "
          f"({RL_EPOCHS} Epochen x {RL_BATCH_SIZE} Rollouts) ...")
    torch.manual_seed(RL_SEED)
    started = time.perf_counter()
    policy = train_rl(data)
    train_time = time.perf_counter() - started
    print(f"Trainingszeit: {train_time:.1f}s (einmalige Offline-Kosten)\n")

    target_env = CVRPEnvironment(lambda: target_instance(data))

    started = time.perf_counter()
    greedy_routes, _, greedy_unserved = greedy_rollout(policy, target_env)
    greedy_time = time.perf_counter() - started
    greedy_distance = None
    if greedy_unserved == 0:
        greedy_distance = print_routes(data, greedy_routes, "Touren laut RL (greedy):")
    else:
        print(f"RL (greedy): keine zulaessige Loesung "
              f"({greedy_unserved:.0f} Nachfrageeinheiten unbedient)")

    print()
    started = time.perf_counter()
    sampled_routes, _ = sampled_rollout(policy, target_env)
    sampled_time = time.perf_counter() - started
    sampled_distance = None
    if sampled_routes:
        sampled_distance = print_routes(
            data, sampled_routes, f"Touren laut RL (bestes aus {RL_SAMPLES} Rollouts):"
        )
    else:
        print(f"RL (sampling): keine zulaessige Loesung in {RL_SAMPLES} Rollouts")

    print()
    print("=" * 60)
    print("Vergleich auf der Zielinstanz")
    print("=" * 60)
    print_comparison(
        [
            ("OR-Tools", ortools_distance, ortools_time),
            ("RL (greedy)", greedy_distance, greedy_time),
            (f"RL (sampling x{RL_SAMPLES})", sampled_distance, sampled_time),
        ],
        ortools_distance,
    )
    print(f"\nRL-Training separat: {train_time:.1f}s, einmalig und instanzunabhaengig.")


if __name__ == "__main__":
    main()
