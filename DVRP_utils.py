import numpy as np

# Generator für die Umwelt
ENV_RNG = np.random.default_rng(42)

# Generator für die Fahrzeuge
VEHICLE_RNG = np.random.default_rng(44)

# Generator für die Zeitpunkte der neuen dynamischen Kunden
NEW_TIME_RNG = np.random.default_rng(45)

# Generator für die Koordinaten der neuen dynamischen Kunden
NEW_COORD_RNG = np.random.default_rng(46)

def set_all_seeds(seed):
    """Ermöglicht es, die gesamte Simulation mit einer Basis-Zahl zu steuern"""
    global ENV_RNG, VEHICLE_RNG, NEW_TIME_RNG, NEW_COORD_RNG
    ENV_RNG = np.random.default_rng(seed)
    VEHICLE_RNG = np.random.default_rng(seed + 10)
    NEW_TIME_RNG = np.random.default_rng(seed + 20)
    NEW_COORD_RNG = np.random.default_rng(seed + 15)
