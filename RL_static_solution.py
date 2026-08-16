"""
REINFORCE fuer ein Capacitated VRP (CVRP) mit mehreren Fahrzeugen (hier: 4).

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


# ---------------------------------------------------------------------------
# 1) Umgebung: K Fahrzeuge, feste Kapazitaet
# ---------------------------------------------------------------------------
class MultiVehicleVRPEnvironment:
    def __init__(self, num_customers=15, num_vehicles=4, capacity=20, seed=None):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

    def reset(self):
        n = self.num_customers + 1  # Index 0 = Depot
        self.coords = self.rng.uniform(0, 1, size=(n, 2)).astype(np.float32)

        demand = self.rng.integers(1, 10, size=n).astype(np.float32)
        demand[0] = 0.0  # Depot hat keine Nachfrage
        self.demand = demand

        K = self.num_vehicles
        self.vehicle_pos = np.zeros(K, dtype=np.int64)          # alle starten im Depot
        self.vehicle_cap_left = np.full(K, self.capacity, dtype=np.float32)
        self.total_distance = 0.0
        self.routes = [[0] for _ in range(K)]
        return self._state()

    def _state(self):
        return {
            "coords": self.coords,
            "demand": self.demand,
            "capacity": self.capacity,
            "vehicle_pos": self.vehicle_pos,
            "vehicle_cap_left": self.vehicle_cap_left,
        }

    def feasibility_mask(self):
        """Form (K, n): mask[v, j] = 'Fahrzeug v faehrt zu Knoten j' erlaubt?"""
        n = len(self.demand)
        K = self.num_vehicles
        mask = np.zeros((K, n), dtype=bool)
        for v in range(K):
            cur = self.vehicle_pos[v]
            for j in range(n):
                if j == cur:
                    continue  # Selbst-Schleife verboten (Reward-Hacking-Fix)
                if j == 0:
                    mask[v, j] = True  # Rueckkehr zum Depot ist immer erlaubt
                elif self.demand[j] > 0 and self.demand[j] <= self.vehicle_cap_left[v]:
                    mask[v, j] = True
        return mask

    def step(self, vehicle, node):
        cur = self.vehicle_pos[vehicle]
        dist = float(np.linalg.norm(self.coords[cur] - self.coords[node]))
        reward = -dist
        self.total_distance += dist

        if node == 0:
            self.vehicle_cap_left[vehicle] = self.capacity  # Auffuellen im Depot
        else:
            served = min(self.demand[node], self.vehicle_cap_left[vehicle])
            self.demand[node] -= served
            self.vehicle_cap_left[vehicle] -= served

        self.vehicle_pos[vehicle] = node
        self.routes[vehicle].append(node)
        return self._state(), reward

    def is_done(self):
        all_served = self.demand[1:].sum() <= 1e-6
        all_at_depot = bool(np.all(self.vehicle_pos == 0))
        return all_served and all_at_depot


def build_features(state):
    """Feature-Matrix (K*n, 6): pro (Fahrzeug, Knoten)-Kombination
    [x, y, Restnachfrage, Distanz zu diesem Fahrzeug, dessen Restkapazitaet, ist_Depot]."""
    coords = state["coords"]
    demand = state["demand"]
    capacity = state["capacity"]
    vehicle_pos = state["vehicle_pos"]
    vehicle_cap_left = state["vehicle_cap_left"]

    n = coords.shape[0]
    K = len(vehicle_pos)

    is_depot = np.zeros(n, dtype=np.float32)
    is_depot[0] = 1.0
    demand_norm = demand / capacity

    rows = []
    for v in range(K):
        dist_to_vehicle = np.linalg.norm(coords - coords[vehicle_pos[v]], axis=1)
        cap_col = np.full(n, vehicle_cap_left[v] / capacity, dtype=np.float32)
        feats = np.stack(
            [coords[:, 0], coords[:, 1], demand_norm, dist_to_vehicle, cap_col, is_depot],
            axis=1,
        )
        rows.append(feats)

    return torch.from_numpy(np.concatenate(rows, axis=0).astype(np.float32))  # (K*n, 6)


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
def train(num_epochs=150, batch_size=16, num_customers=15, num_vehicles=4,
          capacity=20, lr=1e-3, max_steps=150, log_every=25):
    env = MultiVehicleVRPEnvironment(num_customers=num_customers, num_vehicles=num_vehicles,
                                      capacity=capacity)
    policy = SimplePolicy(num_features=6)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    n = num_customers + 1

    for epoch in range(num_epochs):
        batch_log_probs, batch_rewards = [], []

        for _ in range(batch_size):
            state = env.reset()
            log_probs = []
            total_reward = 0.0

            for _ in range(max_steps):
                mask = env.feasibility_mask()                 # (K, n)
                mask_flat = torch.from_numpy(mask.reshape(-1))
                features = build_features(state)               # (K*n, 6)

                probs = policy(features, mask_flat)
                dist = torch.distributions.Categorical(probs)
                action_flat = dist.sample()
                log_probs.append(dist.log_prob(action_flat))

                a = int(action_flat.item())
                vehicle, node = a // n, a % n
                state, reward = env.step(vehicle, node)
                total_reward += reward

                if env.is_done():
                    break

            batch_log_probs.append(torch.stack(log_probs).sum())
            batch_rewards.append(total_reward)

        rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        baseline = rewards.mean()                # einfache Batch-Mittelwert-Baseline
        advantage = rewards - baseline

        loss = -(torch.stack(batch_log_probs) * advantage.detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == num_epochs - 1:
            avg_distance = -rewards.mean().item()
            print(f"Epoch {epoch:4d} | durchschn. Gesamtdistanz: {avg_distance:6.3f} "
                  f"| loss: {loss.item():7.4f}")

    return policy


# ---------------------------------------------------------------------------
# 4) Trainierte Policy greedy auf einer neuen Instanz anwenden
# ---------------------------------------------------------------------------
def greedy_rollout(policy, env, max_steps=150):
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
    return env.routes, env.vehicle_cap_left, env.total_distance


if __name__ == "__main__":
    # Hinweis: fuer bessere Ergebnisse num_epochs/batch_size erhoehen
    # (z.B. num_epochs=500, batch_size=64) -- dauert dann entsprechend laenger.
    torch.manual_seed(0)
    #besser num_epochs=500, batch_size=64, dauert dann entsprechend laenger
    trained_policy = train(num_epochs=150, batch_size=16, num_customers=15,
                            num_vehicles=4, capacity=20)

    test_env = MultiVehicleVRPEnvironment(num_customers=15, num_vehicles=4,
                                           capacity=20, seed=123)
    routes, cap_left, total_distance = greedy_rollout(trained_policy, test_env)

    print("\nGefundene Routen je Fahrzeug (0 = Depot):")
    for v, route in enumerate(routes):
        print(f"  Fahrzeug {v}: {route}")
    print(f"Gesamtdistanz ueber alle Fahrzeuge: {total_distance:.3f}")