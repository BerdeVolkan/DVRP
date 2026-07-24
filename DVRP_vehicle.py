from dvrpsim import Location, Vehicle
from dvrpsim.utils.distances import euclidean_distance
from DVRP_utils import VEHICLE_RNG
#from scipy.stats import rv_discrete, norm

class Truck(Vehicle):

    def __init__(self, id: str) -> None:
        super().__init__(id)

    def travel_time(self, origin: Location, destination: Location) -> float:
        base_distance = euclidean_distance(origin.x, origin.y, destination.x, destination.y)
        mu = 1.03
        sigma = 0.14
        factor = VEHICLE_RNG.normal(mu, sigma)

        # Clipping gegen unrealistische Ausreißer
        factor = max(0.5, factor)
        return base_distance #* factor

    def on_arrival(self) -> None:
        super().on_arrival()

        # Tatsaechlich gefahrene Strecke (Euklidisch, wie DVRP_algo.travel_time_calc)
        # erfassen - NICHT aus OR-Tools-internen Kosten uebernehmen, da
        # solve_vrp_with_ortools fuer en-route-Fahrzeuge kuenstliche
        # Zusatzkosten auf den ersten Arc addiert.
        distance = euclidean_distance(
            self.previous_location.x, self.previous_location.y,
            self.current_location.x, self.current_location.y,
        )
        vehicle_log = self.model.run_log['vehicles'].setdefault(self.id, {'total_distance': 0.0})
        vehicle_log['total_distance'] += distance

        if self.is_idle:
            self.model.request_for_routing()

    def on_departure(self):
        super().on_departure()