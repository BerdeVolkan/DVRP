"""
REINFORCE fuer ein unkapazitiertes Multi-Vehicle-VRP mit Global-Span-Kosten
(hier: 4 Fahrzeuge) -- also exakt das Problem, das auch die Simulation in
DVRP_environment.py und der OR-Tools-Baseline-Solver in DVRP_algo.py loesen.

Keine Kapazitaet: die Simulation kennt keine Kapazitaetsdimension, Trucks
sammeln Auftraege bei den Kunden ein und liefern sie im Depot ab. Ohne
Kapazitaet ist die reine Gesamtdistanz aber minimal, wenn ein einziger Truck
alles faehrt (Dreiecksungleichung -- jede zusaetzliche Tour kostet extra
Depot-Kanten). Erst der Span-Term macht die Flotte sinnvoll. Deshalb wird -- wie
in DVRP_algo.py per distance_dimension.SetGlobalSpanCostCoefficient(10) --
folgende Zielfunktion minimiert:

    Kosten = Summe aller Routenlaengen + span_coefficient * laengste Route

Das ist die exakte Entsprechung des OR-Tools-Terms: OR-Tools bestraft
coef * (max_v cumul_end_v - min_v cumul_start_v); da die Dimension mit
slack_max=0 angelegt ist, gilt cumul_end_v = cumul_start_v + Routenlaenge_v,
und der Minimierer setzt alle Start-Cumuls gleich -> uebrig bleibt
coef * max_v Routenlaenge_v.

Aufbau, identisch zur Logik aus den vorherigen Beispielen:
    MultiVehicleVRPEnvironment -> die Umgebung: Zustand, Feasibility-Maske,
                                   Reward, step()/reset() -- jetzt fuer K Fahrzeuge
    SimplePolicy                -> der Agent: bewertet jede (Fahrzeug, Knoten)-
                                    Kombination, maskiert Unzulaessiges, softmax
    train()                     -> REINFORCE-Trainingsschleife mit
                                    Batch-Mittelwert-Baseline

Aktionsraum: statt "welcher Kunde als naechstes" entscheidet die Policy jetzt
"welches Fahrzeug faehrt als naechstes zu welchem Knoten" -> Aktionsraum ist
K * n (Fahrzeuge x Knoten) statt nur n.

Wichtig: eine Aktion darf nie zum aktuellen Standort des jeweiligen Fahrzeugs
selbst fuehren (j != current) -- sonst koennte die Policy lernen, durch
endloses "im Depot bleiben" (Distanz 0) den Reward zu hacken, statt echte
Routen zu fahren.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import DVRP_utils


# Gleiche Geometrie wie die Simulation in DVRP_environment.py:
# Depot im Ursprung, Kunden ganzzahlig in [-5000, 5000] auf beiden Achsen.
COORD_LIMIT = 5000
MAX_DIST = float(np.hypot(2 * COORD_LIMIT, 2 * COORD_LIMIT))  # ~14142.14, groesste moegliche Distanz

# Normierung fuer die Routenlaengen-Features (beobachtete Routen: ~14k-48k m)
ROUTE_DIST_NORM = MAX_DIST
NUM_FEATURES = 7
# Nur als Absicherung: Episoden koennen bei max_steps abgeschnitten werden, ohne
# dass alle Kunden bedient sind -- ein unbedienter Kunde darf sich nie lohnen.
UNSERVED_PENALTY = MAX_DIST


def dvrp_instance_coords(seed=1, num_customers=20):
    """Exakt die Kundenkoordinaten aus DVRP_environment.py (Depot (0,0) + ENV_RNG).

    Die Ziehreihenfolge (pro Kunde erst x, dann y) muss identisch zur Schleife in
    DVRP_environment.py sein, sonst weichen die Koordinaten ab.
    """
    DVRP_utils.set_all_seeds(seed)
    coords = np.zeros((num_customers + 1, 2), dtype=np.float32)  # Index 0 = Depot bei (0,0)
    for i in range(1, num_customers + 1):
        coords[i, 0] = DVRP_utils.ENV_RNG.integers(-COORD_LIMIT, COORD_LIMIT + 1)
        coords[i, 1] = DVRP_utils.ENV_RNG.integers(-COORD_LIMIT, COORD_LIMIT + 1)
    return coords


# ---------------------------------------------------------------------------
# 1) Umgebung: K Fahrzeuge, keine Kapazitaet, Global-Span-Kosten
# ---------------------------------------------------------------------------
class MultiVehicleVRPEnvironment:
    def __init__(self, num_customers=20, num_vehicles=4, span_coefficient=10.0,
                 seed=None, fixed_coords=None):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        # Pendant zu distance_dimension.SetGlobalSpanCostCoefficient(10) in
        # DVRP_algo.py -- parametrisiert, damit eine Ablation ueber
        # {0, 1, 10, 50} ohne Codeaenderung moeglich ist.
        self.span_coefficient = span_coefficient
        self.rng = np.random.default_rng(seed)
        # fixed_coords: feste Instanz (z.B. die aus DVRP_environment.py) statt
        # einer neuen Zufallsinstanz pro Episode
        self.fixed_coords = fixed_coords

    def reset(self):
        n = self.num_customers + 1  # Index 0 = Depot
        if self.fixed_coords is not None:
            self.coords = self.fixed_coords.copy()
        else:
            self.coords = self.rng.integers(
                -COORD_LIMIT, COORD_LIMIT + 1, size=(n, 2)
            ).astype(np.float32)
            self.coords[0] = 0.0  # Depot im Ursprung, wie in DVRP_environment.py

        self.unvisited = np.ones(n, dtype=bool)
        self.unvisited[0] = False  # Depot ist kein zu bedienender Kunde

        K = self.num_vehicles
        self.vehicle_pos = np.zeros(K, dtype=np.int64)          # alle starten im Depot
        # Entspricht der Cumul-Variable der OR-Tools-Distance-Dimension
        self.vehicle_route_dist = np.zeros(K, dtype=np.float32)
        self.vehicle_done = np.zeros(K, dtype=bool)             # Tour beendet?
        self.total_distance = 0.0
        self.routes = [[0] for _ in range(K)]
        return self._state()

    def _state(self):
        return {
            "coords": self.coords,
            "unvisited": self.unvisited,
            "vehicle_pos": self.vehicle_pos,
            "vehicle_route_dist": self.vehicle_route_dist,
        }

    def feasibility_mask(self):
        """Form (K, n): mask[v, j] = 'Fahrzeug v faehrt zu Knoten j' erlaubt?

        Ohne Kapazitaet gibt es keinen Grund fuer Zwischenstopps im Depot (kein
        Nachladen mehr) -- das Depot ist damit das Tourende, genau wie bei
        OR-Tools (starts -> Kunden -> ends=DEPOT).
        """
        n = len(self.unvisited)
        K = self.num_vehicles
        mask = np.zeros((K, n), dtype=bool)

        open_customers = bool(self.unvisited[1:].any())
        num_active = int((~self.vehicle_done).sum())

        for v in range(K):
            if self.vehicle_done[v]:
                continue  # Tour beendet -> keine Aktion mehr
            cur = self.vehicle_pos[v]
            for j in range(1, n):
                if j != cur and self.unvisited[j]:  # Selbst-Schleife verboten
                    mask[v, j] = True
            # Depot = Tourende. Nur erlaubt, wenn das Fahrzeug unterwegs ist UND
            # danach noch ein anderes Fahrzeug die offenen Kunden uebernehmen
            # kann -- sonst koennten alle Fahrzeuge heimfahren, waehrend Kunden
            # offen sind, und die Maske waere komplett False (softmax -> NaN).
            if cur != 0 and (not open_customers or num_active > 1):
                mask[v, 0] = True
        return mask

    def step(self, vehicle, node):
        cur = self.vehicle_pos[vehicle]
        dist = float(np.linalg.norm(self.coords[cur] - self.coords[node]))
        reward = -dist
        self.total_distance += dist
        self.vehicle_route_dist[vehicle] += dist

        if node == 0:
            self.vehicle_done[vehicle] = True  # zurueck im Depot -> Tour beendet
        else:
            self.unvisited[node] = False

        self.vehicle_pos[vehicle] = node
        self.routes[vehicle].append(node)
        return self._state(), reward

    def is_done(self):
        all_served = not bool(self.unvisited[1:].any())
        all_at_depot = bool(np.all(self.vehicle_pos == 0))
        return all_served and all_at_depot

    def max_route_distance(self):
        return float(self.vehicle_route_dist.max())

    def total_cost(self):
        """Identisch zur OR-Tools-Zielfunktion: Gesamtdistanz + coef * laengste Route."""
        cost = self.total_distance + self.span_coefficient * self.max_route_distance()
        return cost + UNSERVED_PENALTY * int(self.unvisited[1:].sum())


def build_features(state):
    """Feature-Matrix (K*n, NUM_FEATURES): pro (Fahrzeug, Knoten)-Kombination
    [x, y, noch_offen, Distanz zu diesem Fahrzeug, dessen bisherige Routenlaenge,
     Span-Marge, ist_Depot].

    Die beiden Routenlaengen-Features sind fuer die Span-Kosten essenziell: ohne
    sie kann die Policy gar nicht wissen, welches Fahrzeug gerade der Engpass
    ist. Sie sind das RL-Pendant zur Cumul-Variable, ueber die OR-Tools den
    Span-Term rechnet.
    """
    coords = state["coords"]
    unvisited = state["unvisited"]
    vehicle_pos = state["vehicle_pos"]
    vehicle_route_dist = state["vehicle_route_dist"]

    n = coords.shape[0]     #Anzahl der Knoten (Depot + Kunden)
    K = len(vehicle_pos)    #Anzahl der Fahrzeuge

    is_depot = np.zeros(n, dtype=np.float32)
    is_depot[0] = 1.0
    unvisited_f = unvisited.astype(np.float32)
    # Koordinaten und Distanzen liegen im Meter-Bereich (bis ~14000) und muessen
    # normalisiert werden, sonst saettigt das Netz sofort
    coords_norm = coords / COORD_LIMIT  # [-1, 1]
    max_route = float(vehicle_route_dist.max())

    rows = []
    for v in range(K):
        dist_to_vehicle = np.linalg.norm(coords - coords[vehicle_pos[v]], axis=1)
        route_col = np.full(n, vehicle_route_dist[v] / ROUTE_DIST_NORM, dtype=np.float32)
        # Um wie viel waechst die laengste Route (= der Span-Term), wenn v nach j
        # faehrt? > 0 heisst: dieser Zug verschiebt das Maximum nach oben.
        span_margin = (
            (vehicle_route_dist[v] + dist_to_vehicle) - max_route
        ) / ROUTE_DIST_NORM
        feats = np.stack(
            [coords_norm[:, 0], coords_norm[:, 1], unvisited_f,
             dist_to_vehicle / MAX_DIST, route_col,
             span_margin.astype(np.float32), is_depot],
            axis=1,
        )
        rows.append(feats)

    return torch.from_numpy(np.concatenate(rows, axis=0).astype(np.float32))  # (K*n, NUM_FEATURES)


# ---------------------------------------------------------------------------
# 2) Policy: bewertet jede (Fahrzeug, Knoten)-Kombination
# ---------------------------------------------------------------------------
class SimplePolicy(nn.Module):
    def __init__(self, num_features=6, hidden_dim=64):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask_flat):
        scores = self.scorer(features).squeeze(-1)
        scores = scores.masked_fill(~mask_flat, float("-inf"))
        return torch.softmax(scores, dim=-1)


# ---------------------------------------------------------------------------
# 3) REINFORCE-Trainingsschleife
# ---------------------------------------------------------------------------
def train(num_epochs=150, batch_size=16, num_customers=20, num_vehicles=4,
          span_coefficient=10.0, lr=1e-3, max_steps=200, log_every=25):
    env = MultiVehicleVRPEnvironment(num_customers=num_customers, num_vehicles=num_vehicles,
                                      span_coefficient=span_coefficient)
    policy = SimplePolicy(num_features=NUM_FEATURES)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    n = num_customers + 1

    for epoch in range(num_epochs):
        batch_log_probs, batch_rewards = [], []
        batch_distances, batch_max_routes = [], []

        for _ in range(batch_size):
            state = env.reset()
            log_probs = []

            for _ in range(max_steps):
                mask = env.feasibility_mask()                 # (K, n)
                mask_flat = torch.from_numpy(mask.reshape(-1))
                features = build_features(state)               # (K*n, NUM_FEATURES)

                probs = policy(features, mask_flat)
                dist = torch.distributions.Categorical(probs)
                action_flat = dist.sample()
                log_probs.append(dist.log_prob(action_flat))

                a = int(action_flat.item())
                vehicle, node = a // n, a % n
                state, _ = env.step(vehicle, node)

                if env.is_done():
                    break

            # Return direkt aus der Zielfunktion, nicht aus den Step-Rewards
            # aufsummiert: der Span-Term ist ein Episoden-Term. Wuerde man ihn
            # nur beim is_done()-Schritt addieren, bekaeme eine bei max_steps
            # abgeschnittene Episode faelschlich einen besseren Return.
            batch_log_probs.append(torch.stack(log_probs).sum())
            batch_rewards.append(-env.total_cost())
            batch_distances.append(env.total_distance)
            batch_max_routes.append(env.max_route_distance())

        rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        baseline = rewards.mean()                # einfache Batch-Mittelwert-Baseline
        # Standardisieren, damit die Gradienten unabhaengig von der Meter-Skala
        # der Distanzen (~1e5 pro Episode) bleiben
        advantage = (rewards - baseline) / (rewards.std() + 1e-8)

        loss = -(torch.stack(batch_log_probs) * advantage.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            avg_distance = float(np.mean(batch_distances))
            avg_max_route = float(np.mean(batch_max_routes))
            avg_cost = -rewards.mean().item()
            print(f"Epoch {epoch:4d} | Gesamtdistanz: {avg_distance:10.1f} m "
                  f"| laengste Route: {avg_max_route:9.1f} m "
                  f"| Zielfunktion: {avg_cost:11.1f} "
                  f"| loss: {loss.item():7.4f}")

    return policy


# ---------------------------------------------------------------------------
# 4) Trainierte Policy greedy auf einer neuen Instanz anwenden
# ---------------------------------------------------------------------------
def greedy_rollout(policy, env, max_steps=200):
    state = env.reset()
    n = env.num_customers + 1
    for _ in range(max_steps):
        mask = env.feasibility_mask()
        mask_flat = torch.from_numpy(mask.reshape(-1))
        features = build_features(state)
        with torch.no_grad():
            probs = policy(features, mask_flat)
        a = int(torch.argmax(probs).item())
        vehicle, node = a // n, a % n
        state, _ = env.step(vehicle, node)
        if env.is_done():
            break
    return env.routes, env.vehicle_route_dist, env.total_distance


if __name__ == "__main__":
    # Hinweis: fuer bessere Ergebnisse num_epochs/batch_size erhoehen
    # (z.B. num_epochs=500, batch_size=64) -- dauert dann entsprechend laenger.
    torch.manual_seed(0)
    SPAN_COEFFICIENT = 10.0  # wie SetGlobalSpanCostCoefficient(10) in DVRP_algo.py
    #besser num_epochs=500, batch_size=64, dauert dann entsprechend laenger
    trained_policy = train(num_epochs=150, batch_size=16, num_customers=20,
                            num_vehicles=4, span_coefficient=SPAN_COEFFICIENT)

    # Test auf exakt der Instanz aus DVRP_environment.py (seed=1)
    test_env = MultiVehicleVRPEnvironment(num_customers=20, num_vehicles=4,
                                           span_coefficient=SPAN_COEFFICIENT, seed=123,
                                           fixed_coords=dvrp_instance_coords(seed=1))
    routes, route_dists, total_distance = greedy_rollout(trained_policy, test_env)

    print("\nGefundene Routen je Fahrzeug (0 = Depot):")
    for v, route in enumerate(routes):
        print(f"  Fahrzeug {v}: {route}")
        print(f"    Distanz der Route: {route_dists[v]:.1f} m")
    print(f"Alle Kunden beliefert und zurueck im Depot: {test_env.is_done()}")
    print(f"Gesamtdistanz ueber alle Fahrzeuge: {total_distance:.1f} m")
    print(f"Laengste Route (Span-Term): {test_env.max_route_distance():.1f} m")
    print(f"Zielfunktion (Distanz + {SPAN_COEFFICIENT:g} * laengste Route): "
          f"{test_env.total_cost():.1f}")