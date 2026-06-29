import numpy as np

# Generator für die Umwelt
ENV_RNG = np.random.default_rng(42)

# Generator für die Auswahl der Trigger-Kunden
NEW_CUSTOMER_RNG = np.random.default_rng(43)

# Generator für die Fahrzeuge
VEHICLE_RNG = np.random.default_rng(44)

# Generator für die Koordinaten der neuen dynamischen Kunden
NEW_COORD_RNG = np.random.default_rng(46)

def set_all_seeds(seed):
    """Ermöglicht es, die gesamte Simulation mit einer Basis-Zahl zu steuern"""
    global ENV_RNG, NEW_CUSTOMER_RNG, VEHICLE_RNG, NEW_COORD_RNG
    ENV_RNG = np.random.default_rng(seed)
    NEW_CUSTOMER_RNG = np.random.default_rng(seed + 5)
    VEHICLE_RNG = np.random.default_rng(seed + 10)
    NEW_COORD_RNG = np.random.default_rng(seed + 15)