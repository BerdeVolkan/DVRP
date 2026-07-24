import numpy as np

# Generator für die Umwelt
ENV_RNG = np.random.default_rng(42)

# Generator für die Zeitpunkte der dynamischen Kundenereignisse
NEW_EVENT_TIME_RNG = np.random.default_rng(43)

# Generator für die Fahrzeuge
VEHICLE_RNG = np.random.default_rng(44)

# Generator für die Koordinaten der neuen dynamischen Kunden
NEW_COORD_RNG = np.random.default_rng(46)

# Fallback-Default fuer das Zeitfenster, innerhalb dessen dynamische Kunden
# ausgeloest werden duerfen (DemoModel.setup_events(horizon=...)). Der
# eigentliche Wert wird zur Laufzeit als T_op = alpha * T_offline aus einer
# Offline-Referenzloesung berechnet (siehe DVRP_algo.solve_offline_reference
# und der __main__-Block in DVRP_environment.py); diese Konstante greift nur,
# wenn setup_events() ohne horizon-Argument aufgerufen wird.
SIMULATION_HORIZON = 5000

def set_all_seeds(seed):
    """Ermöglicht es, die gesamte Simulation mit einer Basis-Zahl zu steuern"""
    global ENV_RNG, NEW_EVENT_TIME_RNG, VEHICLE_RNG, NEW_COORD_RNG
    ENV_RNG = np.random.default_rng(seed)
    NEW_EVENT_TIME_RNG = np.random.default_rng(seed + 5)
    VEHICLE_RNG = np.random.default_rng(seed + 10)
    NEW_COORD_RNG = np.random.default_rng(seed + 15)