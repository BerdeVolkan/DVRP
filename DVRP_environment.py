import json
import os
from datetime import datetime

from dvrpsim import Model, Location, Order
from dvrpsim.utils.distances import euclidean_distance
from dvrpsim.utils.order_providers import order_provider
from simpy import Resource
import DVRP_algo
import DVRP_vehicle
from DVRP_utils import ENV_RNG, NEW_COORD_RNG, NEW_TIME_RNG

class DemoModel(Model):

    def __init__(self, num_dynamic_events: int, min_event_time: float = 0, max_event_time: float = 3000) -> None:
        super().__init__()
        self.num_dynamic_events = num_dynamic_events
        self.min_event_time = min_event_time
        self.max_event_time = max_event_time

    def setup_events(self):
        # Zufällige, auf [min_event_time, max_event_time] begrenzte Zeitpunkte ziehen
        event_times = sorted(NEW_TIME_RNG.uniform(self.min_event_time, self.max_event_time, size=self.num_dynamic_events))

        dynamic_orders = []
        for i, release_time in enumerate(event_times):
            x_new = NEW_COORD_RNG.integers(-1000, 1001)
            y_new = NEW_COORD_RNG.integers(-1000, 1001)

            # Location und Order erstellen
            new_loc = Location(id=f'CUSTOMER NEW {i+1}', x=x_new, y=y_new)
            self.add_location(new_loc)

            order_new = Order(id=f'O-NEW-{i+1}')
            order_new.pickup_location = new_loc
            order_new.delivery_location = self._locations['DEPOT']

            order_new.release_date = release_time
            order_new.pickup_duration = 0
            order_new.delivery_duration = 0

            dynamic_orders.append(order_new)

        self.env.process(order_provider(self, dynamic_orders, decision_point_on_request=True))

    def routing_callback(self):
        """
        Wird bei jedem Routing-Request aufgerufen
        Nutzt OR-Tools für die Optimierung
        """
        state = self.get_state()
        #print(self._locations) # dict mit Key Location zb Depot und Value Location object
        return DVRP_algo.routing_algorithm(state, self._locations)


if __name__ == '__main__':

    num_new_customers = 4
    num_static_customers = 20

    model = DemoModel(num_dynamic_events=num_new_customers)

    # Erstelle Depot
    depot = Location(id='DEPOT', x=0, y=0)
    depot.resource = Resource(model.env, 1)
    model.add_location(depot)

    # Erstelle Kunden mit zufälligen Koordinaten
    for i in range(num_static_customers):
        x = ENV_RNG.integers(-1000, 1001)
        y = ENV_RNG.integers(-1000, 1001)

        customer_location = Location(id=f'CUSTOMER {i+1}', x=x, y=y)
        model.add_location(customer_location)
        #print(customer_location.x, customer_location.y)
        
        # Erstelle Order für diesen Kunden
        order = Order(id=f'O-{i+1}')
        order.pickup_location = customer_location
        order.delivery_location = depot
        order.release_date = 0
        order.pickup_duration = 0
        order.delivery_duration = 0
        model.request_order(order, decision_point_on_request=True)

    # Erstelle 4 Fahrzeuge
    for i in range(4):
        vehicle = DVRP_vehicle.Truck(f'TRUCK-{i+1}')
        vehicle.initial_location = depot
        model.add_vehicle(vehicle)

    print("Starte Simulation mit OR-Tools Optimierung")
    print(f"{len(model._locations) - 1} Kunden")
    print(f"{len(model.vehicles)} Fahrzeuge")
    print("-" * 80)
    
    model.setup_events()

    # Starte Simulation
    model.run()

    truck_max_delivery = {}

    for order in model.delivered_orders:
        truck_id = order.pickup_vehicle.id
        if truck_id not in truck_max_delivery or order.delivery_time > truck_max_delivery[truck_id]:
            truck_max_delivery[truck_id] = order.delivery_time

    distances = [truck_max_delivery.get(v.id, 0) for v in model.vehicles]
    n = len(distances)
    jain_fairness_index = (sum(distances) ** 2) / (n * sum(d ** 2 for d in distances))
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results.jsonl')

    result_record = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'num_static_customers': num_static_customers,
        'num_dynamic_events': num_new_customers,
        'min_event_time': model.min_event_time,
        'max_event_time': model.max_event_time,
        'num_vehicles': len(model.vehicles),
        'jain_fairness_index': jain_fairness_index,
        'total_makespan': max(truck_max_delivery.values()),
        'truck_delivery_times': truck_max_delivery,
        'solve_times': DVRP_algo.solve_times,
        'reassigning_rates': DVRP_algo.reassigning_rates,
    }

    with open(results_path, 'a') as f:
        f.write(json.dumps(result_record) + '\n')

    print("-" * 80)
    print("Simulation abgeschlossen")