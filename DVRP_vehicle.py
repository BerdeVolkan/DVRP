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
        if self.is_idle:
            self.model.request_for_routing()

    def on_departure(self):
        super().on_departure()