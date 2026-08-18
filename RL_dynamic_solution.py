"""
REINFORCE fuer ein unkapazitiertes Multi-Vehicle-VRP mit Global-Span-Kosten
(hier: 4 Fahrzeuge) -- trainiert auf **dynamischen Entscheidungspunkten**, wie sie
in der Simulation tatsaechlich auftreten.

Unterschied zu RL_static_solution.py: dort starten alle Fahrzeuge im Depot mit
Restdistanz 0. dvrpsim ruft routing_callback aber mitten im Lauf auf -- Fahrzeuge
sind unterwegs, haben einen fest zugesagten naechsten Halt und eine Restdistanz
dorthin. Genau solche Zustaende erzeugt _reset_dynamic() im Training, und
DVRP_rl_algo.solve_vrp_with_rl setzt beim Deployment ueber dieselbe Methode
(reset_from) den echten Zustand aus dvrpsim ein.

Die Features bleiben unveraendert: der Fahrzeugstandort steckt bereits in
dist_to_vehicle (Feature 4), die Restdistanz in vehicle_route_dist (Feature 5)
und in der Span-Marge (Feature 6). Neu ist nur, womit diese Groessen zu
Episodenbeginn initialisiert werden.

Trainiert wird die Policy einmalig ueber __main__ und als rl_policy.pt
exportiert; das Deployment laedt sie und rechnet nur noch vorwaerts.

Es ist exakt das Problem, das auch die Simulation in DVRP_environment.py und der
OR-Tools-Baseline-Solver in DVRP_algo.py loesen.

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

import os

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

HIDDEN_DIM = 64
# Trainierte Policy fuer das Deployment in DVRP_rl_algo.py
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl_policy.pt")


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
                 seed=None, fixed_coords=None, dynamic_start=True,
                 p_at_depot=0.25, p_to_depot=0.2):
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
        # dynamic_start=False -> statischer Fall wie in RL_static_solution.py
        self.dynamic_start = dynamic_start
        self.p_at_depot = p_at_depot    # Anteil Fahrzeuge, die im Depot stehen
        self.p_to_depot = p_to_depot    # Anteil der Fahrenden, die zum Depot unterwegs sind

    def _init_instance(self):
        """Koordinaten und offene Kunden setzen -- gemeinsam fuer alle reset-Varianten."""
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

    def _apply_start_state(self, start_nodes, remaining_dists):
        """Fahrzeugzustand aus Startknoten und Restdistanzen aufbauen."""
        K = self.num_vehicles
        self.vehicle_pos = np.asarray(start_nodes, dtype=np.int64).copy()
        # Entspricht der Cumul-Variable der OR-Tools-Distance-Dimension: sie startet
        # bei 0 und bekommt die Restdistanz als Zuschlag auf die erste Kante.
        self.vehicle_route_dist = np.asarray(remaining_dists, dtype=np.float32).copy()
        self.vehicle_done = np.zeros(K, dtype=bool)             # Tour beendet?
        # Die Restdistanzen sind bei OR-Tools Teil der Kantenkosten und zaehlen
        # damit in die Gesamtdistanz -- hier genauso, sonst weichen die
        # Zielfunktionen am selben Entscheidungspunkt voneinander ab.
        self.total_distance = float(np.sum(self.vehicle_route_dist))
        self.routes = [[int(j)] for j in self.vehicle_pos]

        # Ein Kunden-Startknoten ist fest zugesagt: das Fahrzeug bedient ihn bei
        # Ankunft, also ist er kein offener Kunde mehr.
        for j in self.vehicle_pos:
            if j != 0:
                self.unvisited[j] = False

        return self._state()

    def reset_from(self, start_nodes, remaining_dists):
        """Setzt die Umgebung auf einen konkreten Entscheidungspunkt-Zustand.

        start_nodes[v]      Knotenindex, den Fahrzeug v fest ansteuert (bzw. wo es
                            steht) -- Pendant zu data["starts"] in DVRP_algo.py
        remaining_dists[v]  noch zu fahrende Distanz bis dorthin -- Pendant zu
                            extra_start_distance (Zuschlag auf die erste Kante)

        Wird von DVRP_rl_algo beim Deployment benutzt und im Training ueber
        _reset_dynamic mit zufaelligen Werten gefuellt -- beide Pfade teilen sich
        damit garantiert dieselbe Zustandssemantik.
        """
        self._init_instance()
        return self._apply_start_state(start_nodes, remaining_dists)

    def _reset_dynamic(self):
        """Zufaelliger Entscheidungspunkt-Zustand fuer das Training.

        Die Restdistanz wird nicht frei gezogen, sondern aus der Geometrie
        erzeugt: das Fahrzeug steht auf der Kante zwischen zwei Knoten, also ist
        die Restdistanz der noch ungefahrene Anteil einer echten Kante. Damit ist
        sie automatisch sinnvoll begrenzt (Mittel ~2600 m, praktisch nie ueber
        10000 m) statt von einer willkuerlichen Obergrenze abzuhaengen.
        """
        self._init_instance()
        n = self.num_customers + 1
        K = self.num_vehicles

        start_nodes = np.zeros(K, dtype=np.int64)
        remaining = np.zeros(K, dtype=np.float32)

        # Ziele ohne Zuruecklegen: zwei Trucks koennen nicht auf denselben offenen
        # Auftrag zusteuern. Aufs Depot dagegen schon, das bleibt immer waehlbar.
        free_customers = list(self.rng.permutation(np.arange(1, n)))

        for v in range(K):
            if self.rng.random() < self.p_at_depot or not free_customers:
                continue  # steht im Depot, Restdistanz 0 -- deckt auch t=0 ab

            if self.rng.random() < self.p_to_depot:
                target = 0                       # en route zum Depot (z.B. mit Ladung)
            else:
                target = int(free_customers.pop())

            origin = int(self.rng.integers(0, n))
            edge = float(np.linalg.norm(self.coords[origin] - self.coords[target]))
            remaining[v] = (1.0 - self.rng.random()) * edge  # ungefahrener Anteil
            start_nodes[v] = target

        return self._apply_start_state(start_nodes, remaining)

    def reset(self):
        if self.dynamic_start:
            return self._reset_dynamic()
        # Statischer Fall: alle im Depot, nichts vorgefahren
        K = self.num_vehicles
        return self.reset_from(np.zeros(K, dtype=np.int64), np.zeros(K, dtype=np.float32))

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
# 2b) Export/Import: Training und Deployment sind getrennte Laeufe
# ---------------------------------------------------------------------------
def save_policy(policy, path=MODEL_PATH, **extra):
    """Speichert Gewichte samt allem, was zum Nachbau noetig ist.

    Die Normierungskonstanten muessen mit: weichen sie beim Deployment ab,
    rechnet das Netz still auf einer anderen Skala als im Training.
    """
    config = {
        "num_features": NUM_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "coord_limit": COORD_LIMIT,
        "max_dist": MAX_DIST,
        "route_dist_norm": ROUTE_DIST_NORM,
    }
    config.update(extra)
    torch.save({"model_state": policy.state_dict(), "config": config}, path)
    print(f"Policy gespeichert: {path}")
    return path


def load_policy(path=MODEL_PATH):
    """Laedt eine exportierte Policy. Returns (policy, config)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    for name, value in (("coord_limit", COORD_LIMIT), ("max_dist", MAX_DIST),
                        ("route_dist_norm", ROUTE_DIST_NORM)):
        if not np.isclose(config[name], value):
            raise ValueError(
                f"Normierung weicht ab: {name} war beim Training {config[name]}, "
                f"ist hier {value}. Die Policy wuerde auf falscher Skala rechnen."
            )

    policy = SimplePolicy(num_features=config["num_features"],
                          hidden_dim=config["hidden_dim"])
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return policy, config


# ---------------------------------------------------------------------------
# 3) REINFORCE-Trainingsschleife
# ---------------------------------------------------------------------------
def train(num_epochs=150, batch_size=16, num_customers=20, num_vehicles=4,
          span_coefficient=10.0, lr=1e-3, max_steps=200, log_every=25,
          dynamic_start=True):
    env = MultiVehicleVRPEnvironment(num_customers=num_customers, num_vehicles=num_vehicles,
                                      span_coefficient=span_coefficient,
                                      dynamic_start=dynamic_start)
    policy = SimplePolicy(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM)
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
def greedy_rollout(policy, env, max_steps=200, initial_state=None):
    """initial_state: Rueckgabe von env.reset_from(...), wenn der Startzustand
    schon feststeht (Deployment) statt neu gezogen zu werden (Training)."""
    state = env.reset() if initial_state is None else initial_state
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
    torch.manual_seed(0)
    SPAN_COEFFICIENT = 10.0  # wie SetGlobalSpanCostCoefficient(10) in DVRP_algo.py
    TRAIN_CUSTOMERS = 150     # bleibt bei 20: dvrpsim triggert das Routing im
    NUM_VEHICLES = 4         # Wesentlichen beim Eintreffen neuer Kunden

    # Training auf zufaelligen Entscheidungspunkt-Zustaenden (Fahrzeuge unterwegs,
    # mit Restdistanz) -- nicht mehr nur auf dem t=0-Zustand.
    trained_policy = train(num_epochs=500, batch_size=64, num_customers=TRAIN_CUSTOMERS,
                            num_vehicles=NUM_VEHICLES, span_coefficient=SPAN_COEFFICIENT,
                            dynamic_start=True)

    save_policy(trained_policy, num_customers=TRAIN_CUSTOMERS,
                num_vehicles=NUM_VEHICLES, span_coefficient=SPAN_COEFFICIENT,
                dynamic_start=True)

    # Kontrolllauf auf dem t=0-Zustand der Instanz aus DVRP_environment.py (seed=1):
    # alle Fahrzeuge im Depot, Restdistanz 0 -- damit direkt mit den Zahlen aus
    # RL_static_solution.py vergleichbar.
    test_env = MultiVehicleVRPEnvironment(num_customers=TRAIN_CUSTOMERS,
                                           num_vehicles=NUM_VEHICLES,
                                           span_coefficient=SPAN_COEFFICIENT,
                                           dynamic_start=False,
                                           fixed_coords=dvrp_instance_coords(
                                               seed=1, num_customers=TRAIN_CUSTOMERS))
    routes, route_dists, total_distance = greedy_rollout(trained_policy, test_env)

    print(f"\nDynamisch trainiert | Kontrolle auf dem t=0-Zustand ({TRAIN_CUSTOMERS} Kunden)")
    print("\nGefundene Routen je Fahrzeug (0 = Depot):")
    for v, route in enumerate(routes):
        print(f"  Fahrzeug {v}: {route}")
        print(f"    Distanz der Route: {route_dists[v]:.1f} m")
    print(f"Alle Kunden beliefert und zurueck im Depot: {test_env.is_done()}")
    print(f"Gesamtdistanz ueber alle Fahrzeuge: {total_distance:.1f} m")
    print(f"Laengste Route (Span-Term): {test_env.max_route_distance():.1f} m")
    print(f"Zielfunktion (Distanz + {SPAN_COEFFICIENT:g} * laengste Route): "
          f"{test_env.total_cost():.1f}")