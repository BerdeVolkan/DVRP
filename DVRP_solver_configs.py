"""Zentrale Presets fuer OR-Tools Solver-Konfigurationen.

Jedes Preset legt first_solution_strategy und local_search_metaheuristic fest.
time_limit_seconds und span_cost_coefficient sind bewusst NICHT Teil der Presets,
da sie unabhaengig davon variiert werden sollen.
"""

from ortools.constraint_solver import routing_enums_pb2

SOLVER_CONFIGS: dict[str, dict] = {
    "baseline": {
        "id": "baseline",
        "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    },
    "no_metaheuristic": {
        "id": "no_metaheuristic",
        "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    },
    "tabu": {
        "id": "tabu",
        "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    },
    "greedy_construction": {
        "id": "greedy_construction",
        "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    },
    "savings_construction": {
        "id": "savings_construction",
        "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "local_search_metaheuristic": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    },
}

DEFAULT_SOLVER_CONFIG = "baseline"