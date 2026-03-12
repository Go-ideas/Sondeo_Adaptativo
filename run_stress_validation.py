import json

from autoplay_lab import run_lab_validation_suite


def main() -> None:
    with open("lab_validation_suite_stress.json", "r", encoding="utf-8") as f:
        suite = json.load(f)

    cases = suite["cases"]
    seeds = [123, 456, 789]
    n_runs = 10

    print("Iniciando stress test")
    print("Casos:", len(cases))
    print("Seeds:", seeds)
    print("Runs por seed:", n_runs)

    def _on_progress(payload: dict) -> None:
        completed = int(payload.get("completed_batches", 0))
        total = int(payload.get("total_batches_estimated", 0))
        case_name = str(payload.get("case_name", ""))
        seed = int(payload.get("seed", 0))
        replicate = int(payload.get("replicate", 1))
        print(
            f"[{completed}/{total}] Batch completado - case={case_name} seed={seed} replicate={replicate}"
        )

    result = run_lab_validation_suite(
        cases=cases,
        seeds=seeds,
        n_runs=n_runs,
        progress_callback=_on_progress,
    )

    with open("stress_validation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Stress validation completada")
    print("stress_validation_result.json")


if __name__ == "__main__":
    main()
