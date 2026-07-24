"""
Test fuer die entkoppelte Instanzgenerierung (DVRP_instance.py):

  1) Determinismus: zweimalige Generierung mit identischem (scenario_id, seed)
     liefert exakt dieselbe Instanz (Positionen, release_date-Werte,
     D_offline/T_offline/T_op).
  2) Zwei Simulationen (simulate_from_instance) derselben geladenen Instanz mit
     unterschiedlichen solver_config-Presets verwenden dieselben Kunden-
     positionen und release_date-Werte, ablesbar aus den resultierenden
     run_log-JSONs (meta + orders t_request).
"""

import shutil

import DVRP_instance
from DVRP_environment import simulate_from_instance


SCENARIO_ID = "test_repro"
SEED = 999
TOTAL_CUSTOMERS = 8
DEGREE_OF_DYNAMISM = 0.25
OFFLINE_TIME_LIMIT_SECONDS = 5


def test_generation_is_deterministic() -> str:
    dir_a = "/tmp/dvrp_instance_test_a"
    dir_b = "/tmp/dvrp_instance_test_b"
    shutil.rmtree(dir_a, ignore_errors=True)
    shutil.rmtree(dir_b, ignore_errors=True)

    path_a = DVRP_instance.generate_instance(
        SCENARIO_ID, SEED, TOTAL_CUSTOMERS, DEGREE_OF_DYNAMISM,
        offline_time_limit_seconds=OFFLINE_TIME_LIMIT_SECONDS, instances_dir=dir_a,
    )
    path_b = DVRP_instance.generate_instance(
        SCENARIO_ID, SEED, TOTAL_CUSTOMERS, DEGREE_OF_DYNAMISM,
        offline_time_limit_seconds=OFFLINE_TIME_LIMIT_SECONDS, instances_dir=dir_b,
    )

    instance_a = DVRP_instance.load_instance(path_a)
    instance_b = DVRP_instance.load_instance(path_b)

    assert instance_a["static_customers"] == instance_b["static_customers"], \
        "Statische Kundenpositionen weichen zwischen zwei Generierungen ab!"
    assert instance_a["dynamic_customers"] == instance_b["dynamic_customers"], \
        "Dynamische Kunden (Position + release_date) weichen ab!"
    assert instance_a["D_offline"] == instance_b["D_offline"], "D_offline weicht ab!"
    assert instance_a["T_offline"] == instance_b["T_offline"], "T_offline weicht ab!"
    assert instance_a["T_op"] == instance_b["T_op"], "T_op weicht ab!"

    print("[OK] generate_instance() ist deterministisch bei identischem (scenario_id, seed).")
    print(f"     static_customers: {instance_a['static_customers']}")
    print(f"     dynamic_customers: {instance_a['dynamic_customers']}")
    print(f"     D_offline={instance_a['D_offline']:.2f} T_offline={instance_a['T_offline']:.2f} T_op={instance_a['T_op']:.2f}")

    return path_a


def test_two_solver_configs_share_same_instance(instance_path: str) -> None:
    results_dir = "/tmp/dvrp_instance_test_results"
    shutil.rmtree(results_dir, ignore_errors=True)

    path_baseline = simulate_from_instance(instance_path, "baseline", results_dir=results_dir)
    path_tabu = simulate_from_instance(instance_path, "tabu", results_dir=results_dir)

    import json
    with open(path_baseline) as f:
        log_baseline = json.load(f)
    with open(path_tabu) as f:
        log_tabu = json.load(f)

    # meta enthaelt D_offline/T_offline/T_op/seed - muss fuer beide Laeufe
    # identisch sein, da beide dieselbe Instanz-Datei laden.
    for key in ("seed", "scenario_id", "D_offline", "T_offline", "T_op", "alpha"):
        assert log_baseline["meta"][key] == log_tabu["meta"][key], \
            f"meta['{key}'] weicht zwischen solver_configs ab: {log_baseline['meta'][key]} != {log_tabu['meta'][key]}"

    # release_date der dynamischen Kunden aeussert sich in orders[order_id]['t_request'].
    # Fuer dynamische Auftraege (O-NEW-*) muss t_request in beiden Laeufen identisch sein,
    # da beide dieselben release_date-Werte aus der Instanz-Datei uebernehmen.
    dynamic_order_ids = [oid for oid in log_baseline["orders"] if oid.startswith("O-NEW-")]
    assert dynamic_order_ids, "Keine dynamischen Auftraege im Log gefunden - Testinstanz zu klein?"

    for order_id in dynamic_order_ids:
        t_baseline = log_baseline["orders"][order_id]["t_request"]
        t_tabu = log_tabu["orders"][order_id]["t_request"]
        assert t_baseline == t_tabu, (
            f"t_request von {order_id} weicht zwischen solver_configs ab: {t_baseline} != {t_tabu}"
        )

    print("[OK] Beide solver_config-Laeufe (baseline, tabu) verwenden dieselbe Instanz:")
    print(f"     meta uebereinstimmend: seed={log_baseline['meta']['seed']}, "
          f"T_op={log_baseline['meta']['T_op']:.2f}")
    print(f"     release_date (t_request) uebereinstimmend fuer: {dynamic_order_ids}")


if __name__ == '__main__':
    try:
        instance_path = test_generation_is_deterministic()
        test_two_solver_configs_share_same_instance(instance_path)
    except AssertionError as e:
        print(f"[FAILED] {e}")
        raise SystemExit(1)
    else:
        print("[SUCCESS] Alle Reproduzierbarkeits-Tests bestanden.")
        raise SystemExit(0)
