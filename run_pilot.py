"""
Pilotlauf zur Validierung der Gesamtpipeline (Instanzgenerierung, Solver-
Konfigurationen, strukturiertes Logging) fuer den geplanten Heuristikvergleich.

Erzeugt 2 Instanzen (seed=1, seed=2) fuer das Baseline-Szenario aus Kapitel 4
(total_customers=50, degree_of_dynamism=0.4, uniform in [-1000, 1000]) und
simuliert jede mit allen 5 Solver-Presets aus DVRP_solver_configs.SOLVER_CONFIGS
(10 Laeufe insgesamt). Danach: Laufzeit-Uebersicht, Sanity-Checks (Fairness
zwischen solver_configs, Solver-Status, vollstaendige Zustellung) und eine
grobe Zeit-Hochrechnung fuer die geplante Vollkampagne.
"""

import json
import os
import shutil
import statistics
import time

import DVRP_instance
from DVRP_environment import simulate_from_instance
from DVRP_solver_configs import SOLVER_CONFIGS

SCENARIO_ID = "n50_dod0.4"
TOTAL_CUSTOMERS = 50
DEGREE_OF_DYNAMISM = 0.4
# Empfehlung fuer n=50 laut DVRP_algo.solve_offline_reference: 30-60s
OFFLINE_TIME_LIMIT_SECONDS = 30
SEEDS = [1, 2]
RESULTS_DIR = "results"


def generate_pilot_instances() -> dict[int, str]:
    instance_paths = {}
    for seed in SEEDS:
        print(f"Generiere Instanz fuer seed={seed} ...")
        path = DVRP_instance.generate_instance(
            scenario_id=SCENARIO_ID,
            seed=seed,
            total_customers=TOTAL_CUSTOMERS,
            degree_of_dynamism=DEGREE_OF_DYNAMISM,
            offline_time_limit_seconds=OFFLINE_TIME_LIMIT_SECONDS,
        )
        instance_paths[seed] = path
        print(f"  -> {path}")
    return instance_paths


def run_pilot_simulations(instance_paths: dict[int, str]) -> list[dict]:
    runs = []
    for seed, instance_path in instance_paths.items():
        for solver_config_id in SOLVER_CONFIGS:
            print(f"Simuliere seed={seed}, solver_config={solver_config_id} ...")
            start = time.perf_counter()
            result_path = simulate_from_instance(instance_path, solver_config_id, results_dir=RESULTS_DIR)
            elapsed = time.perf_counter() - start

            pilot_path = os.path.join(RESULTS_DIR, f"pilot_{seed}_{solver_config_id}.json")
            shutil.move(result_path, pilot_path)

            runs.append({
                "seed": seed,
                "solver_config": solver_config_id,
                "elapsed_seconds": elapsed,
                "result_path": pilot_path,
            })
            print(f"  -> {pilot_path} ({elapsed:.1f}s)")

    return runs


def report_timing(runs: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("Laufzeiten (Wall-Clock, gesamter Lauf inkl. Instanz-Laden und JSON-Export)")
    print("=" * 80)
    for r in runs:
        print(f"  seed={r['seed']} solver_config={r['solver_config']:<22} {r['elapsed_seconds']:8.2f}s")

    times = [r["elapsed_seconds"] for r in runs]
    print(f"\n  min={min(times):.2f}s  mean={statistics.mean(times):.2f}s  "
          f"max={max(times):.2f}s  (n={len(times)})")


def sanity_check_meta_consistency(runs: list[dict]) -> list[str]:
    """
    Fairness-Voraussetzung: alle 5 solver_config-Varianten fuer denselben Seed
    muessen exakt dieselbe Instanz sehen. run_log speichert keine Kunden-
    Koordinaten direkt, daher wird Positionsgleichheit INDIREKT ueber
    D_offline/T_offline (direkte Funktionen der Positionen) sowie DIREKT ueber
    release_date (t_request der dynamischen Auftraege) geprueft.
    """
    warnings = []
    logs_by_seed: dict[int, list] = {}
    for r in runs:
        with open(r["result_path"]) as f:
            logs_by_seed.setdefault(r["seed"], []).append((r["solver_config"], json.load(f)))

    for seed, entries in logs_by_seed.items():
        reference_config, reference_log = entries[0]
        ref_meta = reference_log["meta"]

        for solver_config, log in entries[1:]:
            for key in ("D_offline", "T_offline", "T_op", "num_static", "num_dynamic"):
                if log["meta"].get(key) != ref_meta.get(key):
                    warnings.append(
                        f"seed={seed}: meta['{key}'] weicht ab zwischen '{reference_config}' "
                        f"({ref_meta.get(key)}) und '{solver_config}' ({log['meta'].get(key)})"
                    )

            ref_release = {
                oid: o.get("t_request") for oid, o in reference_log["orders"].items()
                if oid.startswith("O-NEW-")
            }
            cur_release = {
                oid: o.get("t_request") for oid, o in log["orders"].items()
                if oid.startswith("O-NEW-")
            }
            if ref_release != cur_release:
                warnings.append(
                    f"seed={seed}: release_date (t_request) der dynamischen Auftraege weicht ab "
                    f"zwischen '{reference_config}' und '{solver_config}'"
                )

    if not warnings:
        print("[OK] meta (D_offline/T_offline/T_op) und release_date sind fuer beide Seeds "
              "ueber alle 5 solver_config-Varianten identisch.")
    else:
        print("[FAIRNESS-VERLETZUNG]")
        for w in warnings:
            print(f"  {w}")

    return warnings


def sanity_check_solver_status(runs: list[dict]) -> list[str]:
    """
    Meldet den Anteil der Entscheidungspunkte je Lauf, deren Solver-Status
    NICHT ROUTING_SUCCESS ist. HINWEIS: OR-Tools meldet ROUTING_SUCCESS bereits
    dann, wenn irgendeine gueltige Loesung gefunden wurde - auch wenn die
    Guided-Local-Search-Metaheuristik das volle Zeitlimit ausgeschoepft hat,
    ohne weiter zu konvergieren. D.h. dieser Status unterscheidet NICHT
    zuverlaessig zwischen "schnell konvergiert" und "Zeitlimit voll
    ausgeschoepft". Als zusaetzliches, aussagekraeftigeres Signal wird deshalb
    auch der Anteil der Solves gemeldet, deren solve_time_seconds nahe am
    konfigurierten Zeitlimit (2s, Default von solve_vrp_with_ortools) liegt.
    """
    notes = []
    for r in runs:
        with open(r["result_path"]) as f:
            log = json.load(f)

        decision_points = log["decision_points"]
        n = len(decision_points)
        non_success = [dp for dp in decision_points if dp["status"] != "ROUTING_SUCCESS"]
        share_non_success = len(non_success) / n if n else 0.0

        near_time_limit = [dp for dp in decision_points if dp["solve_time_seconds"] >= 1.9]
        share_near_limit = len(near_time_limit) / n if n else 0.0

        print(f"  seed={r['seed']} solver_config={r['solver_config']:<22} "
              f"decision_points={n:3d}  nicht-ROUTING_SUCCESS={len(non_success):3d} ({share_non_success:.0%})  "
              f"solve_time>=1.9s={len(near_time_limit):3d} ({share_near_limit:.0%})")

        if share_non_success > 0.5:
            note = (f"seed={r['seed']} solver_config={r['solver_config']}: {share_non_success:.0%} der "
                    f"Entscheidungspunkte liefen NICHT mit ROUTING_SUCCESS - 2s Zeitlimit fuer n=50 "
                    f"evtl. knapp bemessen.")
            notes.append(note)

        if share_near_limit > 0.5:
            note = (f"seed={r['seed']} solver_config={r['solver_config']}: {share_near_limit:.0%} der "
                    f"Entscheidungspunkte haben das 2s-Zeitlimit praktisch voll ausgeschoepft "
                    f"(solve_time_seconds>=1.9s) - 2s fuer n=50 evtl. knapp bemessen.")
            notes.append(note)

    if notes:
        print("\n[HINWEIS]")
        for note in notes:
            print(f"  {note}")

    return notes


def sanity_check_all_orders_delivered(runs: list[dict]) -> list[str]:
    issues = []
    for r in runs:
        with open(r["result_path"]) as f:
            log = json.load(f)

        undelivered = [oid for oid, o in log["orders"].items() if "t_delivery" not in o]
        print(f"  seed={r['seed']} solver_config={r['solver_config']:<22} "
              f"orders={len(log['orders']):3d}  ohne t_delivery={len(undelivered):3d}")

        if undelivered:
            issue = (f"seed={r['seed']} solver_config={r['solver_config']}: "
                     f"{len(undelivered)} Auftraege ohne t_delivery: {undelivered}")
            issues.append(issue)

    if issues:
        print("\n[AUFFAELLIG]")
        for issue in issues:
            print(f"  {issue}")

    return issues


def report_extrapolation(runs: list[dict]) -> None:
    times = [r["elapsed_seconds"] for r in runs]
    mean_time = statistics.mean(times)

    full_campaign_runs = 8 * 5 * 20
    full_campaign_hours = full_campaign_runs * mean_time / 3600

    fastest_run = min(runs, key=lambda r: r["elapsed_seconds"])
    sweep_runs = 4 * 8 * 20
    sweep_hours = sweep_runs * fastest_run["elapsed_seconds"] / 3600

    print("\n" + "=" * 80)
    print(f"Zeit-Hochrechnung (Basis: mean={mean_time:.2f}s ueber {len(times)} Piloten-Laeufe)")
    print("=" * 80)
    print(f"  Vollkampagne: 8 Szenarien x 5 Solver-Konfigurationen x 20 Seeds = {full_campaign_runs} Laeufe")
    print(f"    -> ca. {full_campaign_hours:.1f} Stunden")
    print(f"  Zeitbudget-Sweep (schnellste Piloten-Konfiguration '{fastest_run['solver_config']}', "
          f"{fastest_run['elapsed_seconds']:.1f}s/Lauf): "
          f"4 Zeitlimits x 8 Szenarien x 20 Seeds = {sweep_runs} Laeufe")
    print(f"    -> ca. {sweep_hours:.1f} Stunden")


if __name__ == '__main__':
    pilot_start = time.perf_counter()

    instance_paths = generate_pilot_instances()
    runs = run_pilot_simulations(instance_paths)

    report_timing(runs)

    print("\n" + "=" * 80)
    print("Sanity-Check 1: Fairness (meta/release_date identisch ueber solver_configs je Seed)")
    print("=" * 80)
    fairness_warnings = sanity_check_meta_consistency(runs)

    print("\n" + "=" * 80)
    print("Sanity-Check 2: Solver-Status je Lauf")
    print("=" * 80)
    status_notes = sanity_check_solver_status(runs)

    print("\n" + "=" * 80)
    print("Sanity-Check 3: alle Auftraege zugestellt (t_delivery vorhanden)")
    print("=" * 80)
    delivery_issues = sanity_check_all_orders_delivered(runs)

    report_extrapolation(runs)

    pilot_elapsed = time.perf_counter() - pilot_start
    print(f"\nGesamtdauer Pilotlauf-Skript: {pilot_elapsed / 60:.1f} min")

    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"  Fairness-Verletzungen:   {len(fairness_warnings)}")
    print(f"  Zeitlimit-Hinweise:      {len(status_notes)}")
    print(f"  Unzugestellte Auftraege: {len(delivery_issues)}")
