"""
Regressionstest fuer die konfigurierbaren OR-Tools Solver-Parameter.

Prueft:
  1) solve_vrp_with_ortools() liefert mit manuell uebergebenen alten
     hartcodierten Werten exakt dasselbe Ergebnis wie mit den neuen
     Default-Werten (nur Pflichtargumente uebergeben).
  2) Jedes der 5 Presets aus DVRP_solver_configs.SOLVER_CONFIGS laeuft
     mit derselben Testinstanz fehlerfrei durch und liefert ein
     gueltiges (nicht-leeres) Ergebnis.
"""

from dvrpsim import Location
from ortools.constraint_solver import routing_enums_pb2

import DVRP_utils
from DVRP_algo import solve_vrp_with_ortools
from DVRP_solver_configs import SOLVER_CONFIGS


def build_test_instance(num_customers: int = 10, num_vehicles: int = 2):
    """Baut eine kleine, reduzierte Testinstanz analog zum Simulations-Setup auf."""
    DVRP_utils.set_all_seeds(123)

    locations = {'DEPOT': Location(id='DEPOT', x=0, y=0)}
    for i in range(num_customers):
        x = int(DVRP_utils.ENV_RNG.integers(-1000, 1001))
        y = int(DVRP_utils.ENV_RNG.integers(-1000, 1001))
        loc_id = f'CUSTOMER {i + 1}'
        locations[loc_id] = Location(id=loc_id, x=x, y=y)

    state = {
        'time': 0,
        'vehicles': {
            f'TRUCK-{v + 1}': {
                'status': 'IDLE',
                'current_visit': {'location': 'DEPOT'},
                'next_visits': [],
                'loaded_orders': [],
                'previous_visit': None,
            }
            for v in range(num_vehicles)
        },
    }

    return locations, state, num_vehicles


def routes_equal(routes_a, routes_b) -> bool:
    return routes_a == routes_b


def total_distance(routes: list[list[str]], locations: dict) -> float:
    from DVRP_algo import travel_time_calc

    total = 0.0
    for route in routes:
        for a, b in zip(route, route[1:]):
            total += travel_time_calc(locations[a], locations[b])
    return total


def test_defaults_match_old_hardcoded_behavior() -> None:
    locations, state, num_vehicles = build_test_instance()

    result_manual, _decision_info_manual = solve_vrp_with_ortools(
        locations,
        locations,
        state,
        num_vehicles,
        first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        time_limit_seconds=2,
        span_cost_coefficient=10,
    )

    result_default, _decision_info_default = solve_vrp_with_ortools(locations, locations, state, num_vehicles)

    assert routes_equal(result_manual, result_default), (
        f"Routen weichen ab!\nmanuell: {result_manual}\ndefault: {result_default}"
    )

    dist_manual = total_distance(result_manual, locations)
    dist_default = total_distance(result_default, locations)
    assert dist_manual == dist_default, (
        f"Gesamtdistanz weicht ab! manuell={dist_manual} default={dist_default}"
    )

    print("[OK] Default-Verhalten entspricht exakt dem alten hartcodierten Verhalten.")
    print(f"     Routen: {result_default}")
    print(f"     Gesamtdistanz: {dist_default:.2f}")


def test_all_presets_run_successfully() -> None:
    locations, state, num_vehicles = build_test_instance()

    for preset_name, config in SOLVER_CONFIGS.items():
        routes, decision_info = solve_vrp_with_ortools(
            locations,
            locations,
            state,
            num_vehicles,
            first_solution_strategy=config["first_solution_strategy"],
            local_search_metaheuristic=config["local_search_metaheuristic"],
        )
        assert decision_info["status"], f"Preset '{preset_name}' hat keinen Solver-Status geliefert!"

        assert routes, f"Preset '{preset_name}' hat keine gueltige Loesung geliefert!"
        assert len(routes) == num_vehicles, (
            f"Preset '{preset_name}' hat {len(routes)} statt {num_vehicles} Routen geliefert!"
        )

        dist = total_distance(routes, locations)
        print(f"[OK] Preset '{preset_name}' erfolgreich. Gesamtdistanz: {dist:.2f}")


if __name__ == '__main__':
    try:
        test_defaults_match_old_hardcoded_behavior()
        test_all_presets_run_successfully()
    except AssertionError as e:
        print(f"[FAILED] {e}")
        raise SystemExit(1)
    else:
        print("[SUCCESS] Alle Regressionstests bestanden.")
        raise SystemExit(0)
