from dvrpsim import Model, Location, Order, Vehicle
from simpy import Resource
import DVRP_algo
import DVRP_vehicle
from DVRP_utils import ENV_RNG, NEW_CUSTOMER_RNG

#first git change

class DemoModel(Model):

    def on_vehicle_arrival(self, vehicle: Vehicle) -> None:
        loc_id = vehicle.current_location.id
        
        if loc_id in self.trigger_customers:
            # Event auslösen
            self.dynamic_events_count += 1
            # Den Trigger entfernen
            self.trigger_customers.remove(loc_id)
            
            new_id = f'CUSTOMER NEW {self.dynamic_events_count}'
            
            x_new = NEW_CUSTOMER_RNG.integers(-1000, 1001)
            y_new = NEW_CUSTOMER_RNG.integers(-1000, 1001)
            
            # Location und Order erstellen
            new_loc = Location(id=new_id, x=x_new, y=y_new)
            self.add_location(new_loc)
            
            order_new = Order(id=f'O-NEW-{self.dynamic_events_count}')
            order_new.pickup_location = new_loc
            order_new.delivery_location = self._locations['DEPOT']

            order_new.release_date = self.env.now
            order_new.pickup_duration = 2
            order_new.delivery_duration = 3
            
            self.request_order(order_new, decision_point_on_request=True)

    def __init__(self, num_dynamic_events: int) -> None:
        super().__init__()
        self.num_dynamic_events = num_dynamic_events
        self.trigger_customers = [] # Liste der IDs, die ein Event auslösen
        self.dynamic_events_count = 0

    def setup_events(self):
        # Alle vorhandenen Kunden IDs
        all_customer_ids = [cid for cid in self._locations.keys() if cid != 'DEPOT']
        
        # Wählen zufällig n Kunden aus, die ein Event auslösen sollen
        if len(all_customer_ids) >= self.num_dynamic_events:
            self.trigger_customers = NEW_CUSTOMER_RNG.choice(
                all_customer_ids, 
                size=self.num_dynamic_events, 
                replace=False
            ).tolist()

    def routing_callback(self):
        """
        Wird bei jedem Routing-Request aufgerufen
        Nutzt OR-Tools für die Optimierung
        """
        state = self.get_state()
        #print(self._locations) # dict mit Key Location zb Depot und Value Location object
        return DVRP_algo.routing_algorithm(state, self._locations)


if __name__ == '__main__':

    num_new_customers = 3

    model = DemoModel(num_dynamic_events=num_new_customers)

    # Erstelle Depot
    depot = Location(id='DEPOT', x=0, y=0)
    depot.resource = Resource(model.env, 1)
    model.add_location(depot)

    # Erstelle Kunden mit zufälligen Koordinaten
    for i in range(20):
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
    
    print("-" * 80)
    print("Simulation abgeschlossen")