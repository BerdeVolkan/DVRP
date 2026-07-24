from dvrpsim import Model, Location, Order
from simpy import Resource
from typing import Any, Generator
import DVRP_algo
import DVRP_vehicle
from DVRP_utils import ENV_RNG, NEW_EVENT_TIME_RNG, NEW_COORD_RNG, SIMULATION_HORIZON

class DemoModel(Model):

    def __init__(self, num_dynamic_events: int, solver_config: str = "baseline") -> None:
        super().__init__()
        self.num_dynamic_events = num_dynamic_events
        self.dynamic_events_count = 0
        self.solver_config = solver_config
        # Positionen der dynamischen Kunden, vorab gezogen via prepare_dynamic_customers()
        self.dynamic_customer_locations: list[Location] = []

    def prepare_dynamic_customers(self) -> list[Location]:
        """
        Zieht die Positionen aller dynamischen Kunden VORAB (noch ohne
        Erscheinungszeiten), damit dieselben Positionen sowohl in die Offline-
        Referenzloesung (solve_offline_reference) als auch in die spaeteren,
        zeitbasierten Kundenereignisse (setup_events) einfliessen.
        """
        self.dynamic_customer_locations = []

        for i in range(self.num_dynamic_events):
            x_new = int(NEW_COORD_RNG.integers(-1000, 1001))
            y_new = int(NEW_COORD_RNG.integers(-1000, 1001))
            self.dynamic_customer_locations.append(
                Location(id=f'CUSTOMER NEW {i + 1}', x=x_new, y=y_new)
            )

        return self.dynamic_customer_locations

    def setup_events(self, horizon: float = SIMULATION_HORIZON) -> None:
        """Plant die Zeitpunkte der dynamischen Kundenereignisse innerhalb von [0, horizon)."""
        event_times = NEW_EVENT_TIME_RNG.uniform(0, horizon, size=self.num_dynamic_events)

        for idx, trigger_time in enumerate(event_times):
            self.env.process(self._dynamic_customer_proc(trigger_time, idx))

    def _dynamic_customer_proc(self, trigger_time: float, idx: int) -> Generator[Any, Any, None]:
        delay = trigger_time - self.env.now
        if delay > 0:
            yield self.env.timeout(delay)

        self._spawn_dynamic_customer(idx)

    def _spawn_dynamic_customer(self, idx: int) -> None:
        self.dynamic_events_count += 1

        # bereits in prepare_dynamic_customers() gezogene Position wiederverwenden
        new_loc = self.dynamic_customer_locations[idx]
        self.add_location(new_loc)

        order_new = Order(id=f'O-NEW-{idx + 1}')
        order_new.pickup_location = new_loc
        order_new.delivery_location = self._locations['DEPOT']

        order_new.release_date = self.env.now
        order_new.pickup_duration = 2
        order_new.delivery_duration = 3

        self.request_order(order_new, decision_point_on_request=True)

    def routing_callback(self):
        """
        Wird bei jedem Routing-Request aufgerufen
        Nutzt OR-Tools für die Optimierung
        """
        state = self.get_state()
        #print(self._locations) # dict mit Key Location zb Depot und Value Location object
        return DVRP_algo.routing_algorithm(state, self._locations, solver_config=self.solver_config)


if __name__ == '__main__':

    total_customers = 20
    degree_of_dynamism = 0.1 # Anteil der Kunden, die dynamisch (über die Zeit) hinzugefügt werden

    num_dynamic_customers = round(total_customers * degree_of_dynamism)
    num_static_customers = total_customers - num_dynamic_customers

    model = DemoModel(num_dynamic_events=num_dynamic_customers)

    # Erstelle Depot
    depot = Location(id='DEPOT', x=0, y=0)
    depot.resource = Resource(model.env, 1)
    model.add_location(depot)

    # Erstelle Kunden mit zufälligen Koordinaten
    static_orders = []
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
        order.pickup_duration = 2
        order.delivery_duration = 3
        static_orders.append(order)
        model.request_order(order, decision_point_on_request=True)

    # Erstelle 4 Fahrzeuge
    num_vehicles = 4
    for i in range(num_vehicles):
        vehicle = DVRP_vehicle.Truck(f'TRUCK-{i+1}')
        vehicle.initial_location = depot
        model.add_vehicle(vehicle)

    # Positionen der dynamischen Kunden VORAB ziehen (noch ohne Erscheinungszeiten),
    # damit sie in die Offline-Referenzloesung einfliessen koennen.
    dynamic_locations = model.prepare_dynamic_customers()

    # Offline-Referenzloesung: alle (statischen + dynamischen) Kunden sind ab
    # t=0 bekannt, keine Reoptimierung, grosszuegiges Zeitlimit. Liefert
    # D_offline (fuer eine spaetere Gap-Metrik) und T_offline (Makespan), aus
    # dem T_op = alpha * T_offline abgeleitet wird (ersetzt SIMULATION_HORIZON).
    offline_locations = {'DEPOT': depot}
    for order in static_orders:
        offline_locations[order.pickup_location.id] = order.pickup_location

    offline_orders = list(static_orders)
    for i, loc in enumerate(dynamic_locations):
        offline_locations[loc.id] = loc

        offline_order = Order(id=f'O-OFFLINE-NEW-{i+1}')
        offline_order.pickup_location = loc
        offline_order.delivery_location = depot
        offline_order.pickup_duration = 2
        offline_order.delivery_duration = 3
        offline_orders.append(offline_order)

    # Empfohlenes time_limit_seconds fuer den Offline-Solve (siehe
    # DVRP_algo.solve_offline_reference): n=20 -> 15-30s, n=50 -> 30-60s,
    # n=100 -> 60-120s. Hier: 20 Kunden -> 20s.
    offline_result = DVRP_algo.solve_offline_reference(
        offline_locations,
        offline_orders,
        num_vehicles=num_vehicles,
        time_limit_seconds=20,
    )

    alpha = 0.75
    T_op = alpha * offline_result['T_offline']

    print(f"Offline-Referenz: D_offline={offline_result['D_offline']:.2f}, T_offline={offline_result['T_offline']:.2f}")
    print(f"T_op (alpha={alpha}) = {T_op:.2f}")

    print("Starte Simulation mit OR-Tools Optimierung")
    print(f"{len(model._locations) - 1} Kunden")
    print(f"{len(model.vehicles)} Fahrzeuge")
    print("-" * 80)

    model.setup_events(horizon=T_op)

    # Starte Simulation
    model.run()
    
    print("-" * 80)
    print("Simulation abgeschlossen")