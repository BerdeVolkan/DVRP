# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Simulation

```bash
python DVRP_environment.py
```

No test suite or linter is configured. The project uses a **Conda** Python environment (configured in `.vscode/settings.json`).

## Dependencies

- `dvrpsim` — discrete-event simulation framework (provides `Model`, `Location`, `Order`, `Vehicle`, plus utilities like `order_provider`)
- `ortools` — Google OR-Tools for VRP solving (`pywrapcp`, `routing_enums_pb2`)
- `numpy` — seeded RNGs for reproducibility
- `simpy` — resource management (depot `Resource`)

## Architecture

The project simulates a **Dynamic Vehicle Routing Problem** where new customer orders appear mid-simulation at random, but bounded, points in simulation time — independent of where vehicles currently are.

### Data flow

1. `DVRP_environment.py` — entry point and `dvrpsim.Model` subclass (`DemoModel`). Sets up a depot, 20 static customers (all requested immediately at `t=0` via `request_order(..., decision_point_on_request=True)`), and 4 trucks.
   - `DemoModel.setup_events()` pre-generates `num_dynamic_events` dynamic customers *before* `model.run()` is called: for each one it draws a random `release_date` from `NEW_TIME_RNG.uniform(min_event_time, max_event_time)` (constructor params, default `[0, 3000]`) and random coordinates from `NEW_COORD_RNG`, builds the `Location`/`Order`, and hands the whole batch to dvrpsim's `order_provider(model, orders, decision_point_on_request=True)` utility (`dvrpsim.utils.order_providers`) via `model.env.process(...)`. `order_provider` releases each order at its `release_date` as an independent SimPy process, which fires `request_for_routing()` → `routing_callback()` for each one — no dependency on `on_vehicle_arrival` anymore (that callback has been removed; it previously spawned a new customer when a vehicle happened to arrive at a pre-selected "trigger" location).
2. `routing_callback()` calls `DVRP_algo.routing_algorithm(state, self._locations)` with the full simulation state snapshot and location dict.
3. `DVRP_algo.py` — OR-Tools solver pipeline:
   - `routing_algorithm()` — entry point; filters to only unpicked orders and relevant locations before solving. Also computes the **Reassigning Rate** metric (see below) before and after solving.
   - `solve_vrp_with_ortools()` — builds OR-Tools `RoutingModel` with per-vehicle distance callbacks. For en-route vehicles, adds remaining travel distance as an extra cost on the first arc so re-optimization accounts for vehicles already mid-trip. Appends each call's wall-clock solve duration to the module-level `solve_times` list.
   - OR-Tools config: `PARALLEL_CHEAPEST_INSERTION` initial solution + `GUIDED_LOCAL_SEARCH` metaheuristic, 2-second time limit, global span cost coefficient = 10.
   - `convert_ortools_solution_to_dvrp()` — translates OR-Tools index routes back to dvrpsim's `{vehicles: {next_visits: [...]}, orders: {...}}` format. For a vehicle that's already `EN_ROUTE` to `DEPOT` with `loaded_orders` (OR-Tools' route for it necessarily starts with `DEPOT`, since dvrpsim forbids diverting an en-route vehicle's already-committed next stop), the leading `DEPOT` entry is kept as-is as a delivery-only stop for the currently loaded orders, and the remaining route is appended as further pickups followed by a second, final `DEPOT` delivery stop for those newly picked orders.
4. `DVRP_vehicle.py` — `Truck` subclass. Overrides `travel_time()` with a stochastic factor (Normal(1.03, 0.14), clipped at 0.5). The factor is currently commented out — actual travel time is pure Euclidean distance. `on_arrival()` triggers `request_for_routing()` when a truck becomes idle.
5. `DVRP_utils.py` — four seeded `numpy.random.default_rng` instances: `ENV_RNG` (42, static customer layout), `VEHICLE_RNG` (44, currently-unused travel-time stochasticity), `NEW_TIME_RNG` (45, dynamic customer arrival times), `NEW_COORD_RNG` (46, dynamic customer coordinates). `set_all_seeds(seed)` reseeds all four from a single base (`seed`, `seed+10`, `seed+20`, `seed+15` respectively) — use this whenever running repeated/comparable experiments, since it's the single source of reproducibility for the whole simulation.

### Key state structure

`model.get_state()` (dvrpsim library method) returns a dict with:
- `state['time']` — current simulation time
- `state['open_orders']` — undelivered orders, keyed by order ID, with `pickup_location`, `pickup_vehicle` (set only once physically picked up, else `None`), `assigned_vehicle` (dvrpsim-computed: the vehicle currently planned/loaded to handle this order, covers already-picked-up, currently-servicing, and future-`next_visits` cases — `None` if no vehicle is assigned yet), etc.
- `state['vehicles'][vehicle_id]` — per-vehicle dict with `status` (`'EN_ROUTE'` or `'IDLE'`), `current_visit`, `next_visits`, `loaded_orders`, `previous_visit`

Vehicle IDs are hardcoded as `'TRUCK-1'` through `'TRUCK-4'` in `DVRP_algo.py`; changing the number of vehicles requires updating `create_data_model` accordingly.

### Experiment metrics (module-level lists in `DVRP_algo.py`)

- `solve_times` — wall-clock duration (seconds) of each `routing.SolveWithParameters()` call.
- `reassigning_rates` — one value per `routing_algorithm()` call (after the first), in `[0, 1]`: the fraction of orders that switched to a different truck between the plan *before* this re-optimization and the plan *right after* it. Only orders that (a) were assigned to some truck both before and after, and (b) had **not yet been picked up** at the time of comparison are counted — already-loaded orders can't switch trucks and would otherwise water down the rate. Computed self-contained within a single `routing_algorithm()` call (before-state from `state['open_orders'][...]['assigned_vehicle']`, after-state from the freshly built `result['vehicles'][...]['next_visits']`), so there's no lag between when a customer arrives and when its impact is measured.
- `truck_max_delivery` / Jain's Fairness Index (computed in `DVRP_environment.py`'s `__main__` block, not stored on `DVRP_algo`) — for each truck, the highest `delivery_time` among its delivered orders is used as a proxy for total distance driven (valid since travel time ≈ Euclidean distance and pickup/delivery durations are 0), then fed into the standard Jain's fairness formula `(Σxᵢ)² / (n · Σxᵢ²)` to gauge how evenly workload is distributed across the fleet.
