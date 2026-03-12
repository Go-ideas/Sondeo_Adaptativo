import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from autoplay_lab import run_lab_validation_suite


def _load_case(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    brief = data.get("brief", {})
    plan = data.get("plan", data.get("plan_v3_2", {}))
    if not isinstance(brief, dict) or not isinstance(plan, dict):
        raise ValueError(f"Caso invalido en {path.name}: requiere 'brief' y 'plan'.")
    return {
        "case_name": path.stem,
        "brief": brief,
        "plan": plan,
    }


def _collect_cases(input_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(input_dir.glob("Caso *.json"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos 'Caso *.json' en {input_dir}")
    return [_load_case(p) for p in files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validador formal de laboratorio autoplay (Fase 1).")
    parser.add_argument(
        "--input-dir",
        default="Aprendizaje",
        help="Directorio con casos JSON (default: Aprendizaje).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Corridas por batch (default: 20).",
    )
    parser.add_argument(
        "--seeds",
        default="123,456",
        help="Semillas separadas por coma (default: 123,456).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Ruta de salida JSON (opcional).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    cases = _collect_cases(input_dir)
    result = run_lab_validation_suite(cases=cases, seeds=seeds, n_runs=int(max(args.n_runs, 1)))

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = Path(f"lab_validation_suite_{ts}.json")

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
