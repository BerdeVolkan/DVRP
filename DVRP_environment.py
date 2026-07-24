import json
import os
from dvrpsim import Model, Location
from simpy import Resource
from typing import Any, Generator
import DVRP_algo
import DVRP_instance
import DVRP_vehicle
from DVRP_order import LoggingOrder

class DemoModel(Model):

    def __init__(self, num_dynamic_events: int, solver_config: str = "baseline") -> None:
        super().__init__()
        self.num_dynamic_events = num_dynamic_events
        self.dynamic_events_count = 0
        self.solver_config = solver_config
        # Positionen der dynamischen Kunden, uebernommen aus der Instanz-Datei
        # via setup_events_from_instance()
        self.dynamic_customer_locations: list[Location] = []

        # Zentrales, maschinenlesbares Log fuer die spaetere Auswertung.
        # HINWEIS: dvrpsim.Model belegt das Attribut "self.log" bereits fuer
        # sein eigenes Text-/Dateilogging (dvrp.log, siehe dvrpsim.utils.logging).
        # Ein Ueberschreiben von self.log wuerde die Simulation zum Absturz
        # bringen (interne Aufrufe wie self.log.on_order_pickup(...) im
        # dvrpsim-Quellcode). Daher wird hier stattdessen self.run_log verwendet.
        self.run_log: dict[str, Any] = {
            "meta": {},
            "orders": {},
            "vehicles": {},
            "decision_points": [],
        }

    def setup_events_from_instance(self, dynamic_customers: list[dict]) -> None:
        """
        Plant die dynamischen Kundenereignisse anhand einer bereits generierten
        Instanz (siehe DVRP_instance.generate_instance()). Positionen UND
        release_date-Zeitpunkte sind darin bereits fixiert - es wird HIER
        NICHT erneut aus NEW_COORD_RNG/NEW_EVENT_TIME_RNG gezogen, damit alle
        Solver-Konfigurationen fuer denselben (scenario_id, seed) exakt
        dieselbe Instanz sehen (siehe Modul-Docstring von DVRP_instance).

        :param dynamic_customers: Liste von {'id', 'x', 'y', 'release_date'}
            Dicts, wie in der Instanz-Datei unter 'dynamic_customers' gespeichert.
        """
        self.dynamic_customer_locations = [
            Location(id=c['id'], x=c['x'], y=c['y']) for c in dynamic_customers
        ]

        for idx, c in enumerate(dynamic_customers):
            self.env.process(self._dynamic_customer_proc(c['release_date'], idx))

    def _dynamic_customer_proc(self, trigger_time: float, idx: int) -> Generator[Any, Any, None]:
        delay = trigger_time - self.env.now
        if delay > 0:
            yield self.env.timeout(delay)

        self._spawn_dynamic_customer(idx)

    def _spawn_dynamic_customer(self, idx: int) -> None:
        self.dynamic_events_count += 1

        # bereits aus der Instanz-Datei uebernommene Position wiederverwenden
        new_loc = self.dynamic_customer_locations[idx]
        self.add_location(new_loc)

        order_new = LoggingOrder(id=f'O-NEW-{idx + 1}')
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
        decision, decision_info = DVRP_algo.routing_algorithm(state, self._locations, solver_config=self.solver_config)

        # solve_vrp_with_ortools()/routing_algorithm() haben keinen direkten
        # Zugriff auf das Model-Objekt, daher werden die Diagnosedaten hier
        # (mit Zugriff auf self.run_log und self.env.now) geloggt.
        if decision_info is not None:
            self.run_log["decision_points"].append({
                "time": self.env.now,
                **decision_info,
            })

        return decision


def simulate_from_instance(instance_path: str, solver_config: str, results_dir: str = "results") -> str:
    """
    Baut eine Simulation aus einer bereits generierten Instanz-Datei
    (DVRP_instance.generate_instance()) auf und fuehrt sie mit der gegebenen
    solver_config aus. Zieht KEINE neuen Kundenpositionen und ruft
    solve_offline_reference() NICHT erneut auf - D_offline/T_offline/T_op und
    alle Positionen/release_date-Werte werden 1:1 aus der Datei uebernommen,
    damit alle 5 Solver-Konfigurationen im Heuristikvergleich exakt dieselbe
    Instanz sehen.

    HINWEIS zu set_all_seeds(): wird hier bewusst NICHT aufgerufen.
    DVRP_instance.generate_instance() hat set_all_seeds(instance['seed'])
    bereits einmalig aufgerufen, um Positionen/Offline-Referenz/release_date
    deterministisch zu erzeugen - das sind ab hier nur noch feste, aus der
    Datei gelesene Zahlen, kein erneuter RNG-Zugriff noetig. Ein zusaetzlicher
    set_all_seeds()-Aufruf haette aktuell keinen beobachtbaren Effekt (der
    VEHICLE_RNG-Stochastikfaktor in DVRP_vehicle.Truck.travel_time() ist
    auskommentiert, siehe dortiger Kommentar), wuerde aber bei dessen
    Reaktivierung dazu fuehren, dass alle 5 Solver-Config-Laeufe fuer
    dieselbe Instanz unbeabsichtigt exakt denselben (oder falsch
    synchronisierten) VEHICLE_RNG-Ausgangszustand erhalten, statt jeweils
    unabhaengig vom Standard-Seed aus weiterzulaufen.

    Returns:
        Pfad der geschriebenen run_log-JSON-Datei.
    """
    instance = DVRP_instance.load_instance(instance_path)

    model = DemoModel(num_dynamic_events=len(instance['dynamic_customers']), solver_config=solver_config)

    depot = Location(id='DEPOT', x=0, y=0)
    depot.resource = Resource(model.env, 1)
    model.add_location(depot)

    for i, c in enumerate(instance['static_customers']):
        customer_location = Location(id=c['id'], x=c['x'], y=c['y'])
        model.add_location(customer_location)

        order = LoggingOrder(id=f'O-{i + 1}')
        order.pickup_location = customer_location
        order.delivery_location = depot
        order.release_date = 0
        order.pickup_duration = 2
        order.delivery_duration = 3
        model.request_order(order, decision_point_on_request=True)

    num_vehicles = instance.get('num_vehicles', 4)
    for i in range(num_vehicles):
        vehicle = DVRP_vehicle.Truck(f'TRUCK-{i + 1}')
        vehicle.initial_location = depot
        model.add_vehicle(vehicle)

    model.run_log["meta"] = {
        "solver_config_id": model.solver_config,
        "seed": instance['seed'],
        "scenario_id": instance['scenario_id'],
        "D_offline": instance['D_offline'],
        "T_offline": instance['T_offline'],
        "T_op": instance['T_op'],
        "alpha": instance['alpha'],
        "num_static": len(instance['static_customers']),
        "num_dynamic": len(instance['dynamic_customers']),
    }

    print(f"Instanz geladen: {instance_path}")
    print(f"Offline-Referenz: D_offline={instance['D_offline']:.2f}, T_offline={instance['T_offline']:.2f}")
    print(f"T_op (alpha={instance['alpha']}) = {instance['T_op']:.2f}")
    print("Starte Simulation mit OR-Tools Optimierung")
    print(f"{len(model._locations) - 1} Kunden")
    print(f"{len(model.vehicles)} Fahrzeuge")
    print("-" * 80)

    model.setup_events_from_instance(instance['dynamic_customers'])

    # Starte Simulation
    model.run()

    print("-" * 80)
    print("Simulation abgeschlossen")

    # Strukturiertes Log als JSON exportieren (ergaenzt das bestehende
    # Text-/Konsolenlogging, ersetzt es nicht).
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir, f"{instance['scenario_id']}_{model.solver_config}_{instance['seed']}.json"
    )

    with open(results_path, "w") as f:
        json.dump(model.run_log, f, indent=2)

    print(f"Strukturiertes Log gespeichert unter: {results_path}")

    return results_path


if __name__ == '__main__':
    # Instanz vorher einmalig erzeugen, z.B.:
    #   python DVRP_instance.py
    # oder DVRP_instance.generate_instance(...) direkt aufrufen. Alle 5
    # Heuristik-Laeufe fuer denselben (scenario_id, seed) nutzen dieselbe
    # Instanz-Datei, nur solver_config variiert (siehe DVRP_solver_configs.py).
    instance_path = os.path.join("instances", "n20_dod0.1_42.json")
    solver_config = "baseline"

    simulate_from_instance(instance_path, solver_config)
