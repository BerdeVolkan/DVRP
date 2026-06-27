import numpy as np

# Generator für die Umwelt
ENV_RNG = np.random.default_rng(42)

# Generator für neue Kunden
NEW_CUSTOMER_RNG = np.random.default_rng(43)

# Generator für die Fahrzeuge
VEHICLE_RNG = np.random.default_rng(44)

def set_all_seeds(seed):
    """Ermöglicht es, die gesamte Simulation mit einer Basis-Zahl zu steuern"""
    global ENV_RNG, NEW_CUSTOMER_RNG, VEHICLE_RNG
    ENV_RNG = np.random.default_rng(seed)
    NEW_CUSTOMER_RNG = np.random.default_rng(seed + 5)
    VEHICLE_RNG = np.random.default_rng(seed + 10)