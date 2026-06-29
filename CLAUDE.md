# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
python DVRP_environment.py
```

No test suite or linter is configured. The project uses a **Conda** Python environment (configured in `.vscode/settings.json`).

## Dependencies

- `dvrpsim` — discrete-event simulation framework (provides `Model`, `Location`, `Order`, `Vehicle`)
- `ortools` — Google OR-Tools for VRP solving (`pywrapcp`, `routing_enums_pb2`)
- `numpy` — seeded RNGs for reproducibility
- `simpy` — resource management (depot `Resource`)

## Architecture

The project simulates a **Dynamic Vehicle Routing Problem** where new customer orders appear mid-simulation when vehicles reach certain locations.

### Data flow

1. `DVRP_environment.py` — entry point and `dvrpsim.Model` subclass. Sets up depot, 20 static customers, 4 trucks, and `num_dynamic_events` trigger customers. On each vehicle arrival at a trigger location, `on_vehicle_arrival()` spawns a new customer and calls `request_order(..., decision_point_on_request=True)`, which fires `routing_callback()`.
2. `routing_callback()` calls `DVRP_algo.routing_algorithm(state, self._locations)` with the full simulation state snapshot and location dict.
3. `DVRP_algo.py` — OR-Tools solver pipeline:
   - `routing_algorithm()` — entry point; filters to only unpicked orders and relevant locations before solving.
   - `solve_vrp_with_ortools()` — builds OR-Tools `RoutingModel` with per-vehicle distance callbacks. For en-route vehicles, adds remaining travel distance as an extra cost on the first arc so re-optimization accounts for vehicles already mid-trip.
   - OR-Tools config: `PARALLEL_CHEAPEST_INSERTION` initial solution + `GUIDED_LOCAL_SEARCH` metaheuristic, 2-second time limit, global span cost coefficient = 10.
   - `convert_ortools_solution_to_dvrp()` — translates OR-Tools index routes back to dvrpsim's `{vehicles: {next_visits: [...]}, orders: {...}}` format.
4. `DVRP_vehicle.py` — `Truck` subclass. Overrides `travel_time()` with a stochastic factor (Normal(1.03, 0.14), clipped at 0.5). The factor is currently commented out — actual travel time is pure Euclidean distance. `on_arrival()` triggers `request_for_routing()` when a truck becomes idle.
5. `DVRP_utils.py` — three seeded `numpy.random.default_rng` instances (seeds 42, 43, 44) for environment layout, new customer placement, and vehicle stochasticity. `set_all_seeds(seed)` reseeds all three from a single base.

### Key state structure

`model.get_state()` returns a dict with:
- `state['time']` — current simulation time
- `state['open_orders']` — undelivered orders, keyed by order ID, with `pickup_location`, `pickup_vehicle`, etc.
- `state['vehicles'][vehicle_id]` — per-vehicle dict with `status` (`'EN_ROUTE'` or `'IDLE'`), `current_visit`, `next_visits`, `loaded_orders`, `previous_visit`

Vehicle IDs are hardcoded as `'TRUCK-1'` through `'TRUCK-4'` in `DVRP_algo.py`; changing the number of vehicles requires updating `create_data_model` accordingly.
