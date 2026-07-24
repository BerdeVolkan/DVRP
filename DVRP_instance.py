"""
Instanzgenerierung, entkoppelt von der eigentlichen Simulation.

Erzeugt fuer ein (scenario_id, seed)-Paar einmalig alle Zufallsanteile, die
sonst bei jeder Skriptausfuehrung neu gezogen wuerden: Kundenpositionen
(statisch + dynamisch), die Offline-Referenzloesung (D_offline, T_offline),
T_op sowie die release_date-Werte der dynamischen Kunden. Das Ergebnis wird
als JSON-Datei gespeichert (instances/{scenario_id}_{seed}.json), damit ein
Heuristikvergleich ueber mehrere Solver-Konfigurationen (siehe
DVRP_solver_configs.SOLVER_CONFIGS) exakt dieselbe Instanz verwendet.

Wichtig: solve_offline_reference() nutzt ein Wall-Clock-Zeitlimit fuer den
OR-Tools Solver, ist also bei erneuter Ausfuehrung selbst mit gleichem Seed
nicht exakt reproduzierbar (Solver-Timing). Indem D_offline/T_offline/T_op
hier EINMAL berechnet und persistiert werden, sehen alle nachfolgenden
Simulationslaeufe (DVRP_environment.simulate_from_instance()) exakt dieselben
Werte.
"""

import json
import os
from typing import Any

from dvrpsim import Location, Order

import DVRP_algo
import DVRP_utils


def _make_offline_order(order_id: str, pickup_location: Location, depot: Location) -> Order:
    order = Order(id=order_id)
    order.pickup_location = pickup_location
    order.delivery_location = depot
    order.pickup_duration = 2
    order.delivery_duration = 3
    return order


def generate_instance(
    scenario_id: str,
    seed: int,
    total_customers: int,
    degree_of_dynamism: float,
    offline_time_limit_seconds: float,
    alpha: float = 0.75,
    num_vehicles: int = 4,
    instances_dir: str = 'instances',
) -> str:
    """
    Erzeugt eine deterministische Probleminstanz und speichert sie als JSON.

    Empfohlene offline_time_limit_seconds (siehe DVRP_algo.solve_offline_reference):
        n=20 Kunden:  15-30 Sekunden
        n=50 Kunden:  30-60 Sekunden
        n=100 Kunden: 60-120 Sekunden

    Returns:
        Pfad der geschriebenen Instanz-JSON-Datei.
    """
    DVRP_utils.set_all_seeds(seed)

    num_dynamic_customers = round(total_customers * degree_of_dynamism)
    num_static_customers = total_customers - num_dynamic_customers

    static_customers: list[dict[str, Any]] = []
    for i in range(num_static_customers):
        x = int(DVRP_utils.ENV_RNG.integers(-1000, 1001))
        y = int(DVRP_utils.ENV_RNG.integers(-1000, 1001))
        static_customers.append({'id': f'CUSTOMER {i + 1}', 'x': x, 'y': y})

    dynamic_customers: list[dict[str, Any]] = []
    for i in range(num_dynamic_customers):
        x = int(DVRP_utils.NEW_COORD_RNG.integers(-1000, 1001))
        y = int(DVRP_utils.NEW_COORD_RNG.integers(-1000, 1001))
        dynamic_customers.append({'id': f'CUSTOMER NEW {i + 1}', 'x': x, 'y': y})

    # Offline-Referenzloesung: alle (statischen + dynamischen) Kunden ab t=0
    # bekannt, keine Reoptimierung.
    depot = Location(id='DEPOT', x=0, y=0)
    offline_locations = {'DEPOT': depot}
    offline_orders = []

    for i, c in enumerate(static_customers):
        loc = Location(id=c['id'], x=c['x'], y=c['y'])
        offline_locations[loc.id] = loc
        offline_orders.append(_make_offline_order(f'O-{i + 1}', loc, depot))

    for i, c in enumerate(dynamic_customers):
        loc = Location(id=c['id'], x=c['x'], y=c['y'])
        offline_locations[loc.id] = loc
        offline_orders.append(_make_offline_order(f'O-OFFLINE-NEW-{i + 1}', loc, depot))

    offline_result = DVRP_algo.solve_offline_reference(
        offline_locations,
        offline_orders,
        num_vehicles=num_vehicles,
        time_limit_seconds=offline_time_limit_seconds,
    )

    T_op = alpha * offline_result['T_offline']

    # release_date-Werte der dynamischen Kunden ziehen (Erscheinungszeitpunkt
    # innerhalb [0, T_op)). Array-Index i <-> dynamischer Kunde i.
    release_dates = DVRP_utils.NEW_EVENT_TIME_RNG.uniform(0, T_op, size=num_dynamic_customers)
    for c, release_date in zip(dynamic_customers, release_dates):
        c['release_date'] = float(release_date)

    instance = {
        'scenario_id': scenario_id,
        'seed': seed,
        'total_customers': total_customers,
        'degree_of_dynamism': degree_of_dynamism,
        'num_vehicles': num_vehicles,
        'alpha': alpha,
        'static_customers': static_customers,
        'dynamic_customers': dynamic_customers,
        'D_offline': offline_result['D_offline'],
        'T_offline': offline_result['T_offline'],
        'T_op': T_op,
    }

    os.makedirs(instances_dir, exist_ok=True)
    instance_path = os.path.join(instances_dir, f'{scenario_id}_{seed}.json')
    with open(instance_path, 'w') as f:
        json.dump(instance, f, indent=2)

    return instance_path


def load_instance(instance_path: str) -> dict[str, Any]:
    """Laedt eine zuvor mit generate_instance() erzeugte Instanz-Datei."""
    with open(instance_path) as f:
        return json.load(f)


if __name__ == '__main__':
    path = generate_instance(
        scenario_id='n20_dod0.1',
        seed=42,
        total_customers=20,
        degree_of_dynamism=0.1,
        offline_time_limit_seconds=20,
    )
    print(f"Instanz gespeichert unter: {path}")
