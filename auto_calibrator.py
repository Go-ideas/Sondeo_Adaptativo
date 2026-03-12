from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoplay_lab import _case_identity, run_lab_validation_suite
from calibration_storage import evaluate_category_publishability as evaluate_category_publishability


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SUITE_PATHS = [
    BASE_DIR / "lab_validation_suite_stress.json",
    BASE_DIR / "Aprendizaje" / "lab_validation_suite_stress.json",
]
DEFAULT_GUIDES_PATHS = [
    BASE_DIR / "calibration_guides.json",
    BASE_DIR / "Aprendizaje" / "calibration_guides.json",
]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _resolve_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _default_guides() -> Dict[str, Any]:
    return {
        "categories": {
            "universidad": {"accepted_min": 0.30, "coverage_util_min": 0.40, "human_like_min": 0.70},
            "soda": {"accepted_min": 0.20, "coverage_util_min": 0.35, "human_like_min": 0.70},
            "auto": {"accepted_min": 0.20, "coverage_util_min": 0.35, "human_like_min": 0.70},
            "banco": {"accepted_min": 0.25, "coverage_util_min": 0.40, "human_like_min": 0.70},
            "seguro": {"accepted_min": 0.25, "coverage_util_min": 0.40, "human_like_min": 0.70},
            "retail": {"accepted_min": 0.25, "coverage_util_min": 0.40, "human_like_min": 0.70},
            "tecnologia": {"accepted_min": 0.25, "coverage_util_min": 0.40, "human_like_min": 0.70},
            "producto_generico": {"accepted_min": 0.25, "coverage_util_min": 0.35, "human_like_min": 0.70},
        }
    }


def _load_guides() -> Dict[str, Any]:
    found = _resolve_existing(DEFAULT_GUIDES_PATHS)
    if not found:
        return _default_guides()
    guides = _read_json(found, {})
    if not isinstance(guides, dict) or not guides:
        return _default_guides()
    return guides


def _extract_category(case: Dict[str, Any]) -> str:
    brief = case.get("brief", {}) if isinstance(case, dict) else {}
    ident = _case_identity(brief if isinstance(brief, dict) else {})
    return str(ident.get("categoria", "producto_generico")).strip().lower() or "producto_generico"


def _load_cases() -> List[Dict[str, Any]]:
    suite_path = _resolve_existing(DEFAULT_SUITE_PATHS)
    if not suite_path:
        return []
    payload = _read_json(suite_path, {})
    cases = payload.get("cases", []) if isinstance(payload, dict) else []
    return [c for c in cases if isinstance(c, dict)]


def _avg(values: List[float]) -> float:
    return round(sum(values) / max(len(values), 1), 3)


def _aggregate_batch_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "accepted_count": 0.0,
            "review_count": 0.0,
            "rejected_count": 0.0,
            "avg_coverage": 0.0,
            "avg_coverage_util": 0.0,
            "avg_naturalidad": 0.0,
            "avg_human_like_index": 0.0,
            "avg_question_diversity_index": 0.0,
            "best_quality_score": 0.0,
            "median_quality_score": 0.0,
        }
    return {
        "accepted_count": _avg([float(r.get("accepted_count", 0.0)) for r in rows]),
        "review_count": _avg([float(r.get("review_count", 0.0)) for r in rows]),
        "rejected_count": _avg([float(r.get("rejected_count", 0.0)) for r in rows]),
        "avg_coverage": _avg([float(r.get("avg_coverage", 0.0)) for r in rows]),
        "avg_coverage_util": _avg([float(r.get("avg_coverage_util", 0.0)) for r in rows]),
        "avg_naturalidad": _avg([float(r.get("avg_naturalidad", 0.0)) for r in rows]),
        "avg_human_like_index": _avg([float(r.get("avg_human_like_index", 0.0)) for r in rows]),
        "avg_question_diversity_index": _avg([float(r.get("avg_question_diversity_index", 0.0)) for r in rows]),
        "best_quality_score": _avg([float(r.get("best_quality_score", 0.0)) for r in rows]),
        "median_quality_score": _avg([float(r.get("median_quality_score", 0.0)) for r in rows]),
    }


def _collect_runs(validation_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for batch in validation_payload.get("batches", []) if isinstance(validation_payload, dict) else []:
        if not isinstance(batch, dict):
            continue
        accepted.extend([x for x in batch.get("accepted_runs", []) if isinstance(x, dict)])
        review.extend([x for x in batch.get("review_runs", []) if isinstance(x, dict)])
        rejected.extend([x for x in batch.get("rejected_runs", []) if isinstance(x, dict)])

    def _sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda x: float((x.get("quality_scores", {}) if isinstance(x.get("quality_scores", {}), dict) else {}).get("quality_score_total", 0.0)),
            reverse=True,
        )

    return {"accepted_runs": _sort(accepted), "review_runs": _sort(review), "rejected_runs": _sort(rejected)}


def _check_goals(categoria: str, summary: Dict[str, Any], guides: Dict[str, Any]) -> Dict[str, Any]:
    categories = guides.get("categories", {}) if isinstance(guides, dict) else {}
    target = categories.get(categoria, categories.get("producto_generico", {})) if isinstance(categories, dict) else {}
    accepted_min = float(target.get("accepted_min", 0.25))
    coverage_util_min = float(target.get("coverage_util_min", 0.35))
    human_like_min = float(target.get("human_like_min", 0.70))

    accepted = float(summary.get("accepted", 0.0))
    coverage_util = float(summary.get("avg_coverage_util", 0.0))
    human_like = float(summary.get("avg_human_like_index", 0.0))

    fails: List[str] = []
    recs: List[str] = []
    if accepted < accepted_min:
        fails.append("accepted_bajo")
        recs.append("Aumentar deteccion semantica util y seguimiento por foco.")
    if coverage_util < coverage_util_min:
        fails.append("coverage_util_baja")
        recs.append("Forzar profundidad minima antes de rotar atributo.")
    if human_like < human_like_min:
        fails.append("human_like_bajo")
        recs.append("Variar plantillas y reforzar continuidad contextual.")

    return {
        "cumple_metas": len(fails) == 0,
        "diagnostico": fails,
        "recomendaciones": recs or ["Sin recomendaciones criticas."],
        "metas": {
            "accepted_min": accepted_min,
            "coverage_util_min": coverage_util_min,
            "human_like_min": human_like_min,
        },
    }


def derive_codex_action(diagnostico: List[str]) -> tuple[str, str, str]:
    diag = [str(x).strip().lower() for x in (diagnostico or []) if str(x).strip()]
    if "accepted_bajo" in diag:
        return (
            "accepted_bajo",
            "interview_engine.py",
            "Aumentar deteccion semantica util y seguimiento por foco",
        )
    if "coverage_util_baja" in diag:
        return (
            "coverage_util_baja",
            "interview_engine.py",
            "Mejorar profundidad util y evidencia por atributo",
        )
    if "human_like_bajo" in diag:
        return (
            "human_like_bajo",
            "interview_engine.py",
            "Mejorar naturalidad de preguntas",
        )
    return (
        "sin_diagnostico",
        "interview_engine.py",
        "Revisar reglas del moderador",
    )


def build_codex_prompt(
    categoria: str,
    diagnostico: List[str],
    recomendaciones: List[str],
    metricas: Dict[str, Any],
    metas: Dict[str, Any],
    archivo_objetivo: str,
    cambio_sugerido: str,
) -> str:
    accepted = metricas.get("accepted")
    coverage = metricas.get("coverage_util")
    human = metricas.get("human_like")

    accepted_meta = metas.get("accepted_min")
    coverage_meta = metas.get("coverage_util_min")
    human_meta = metas.get("human_like_min")

    diagnostico_txt = "\n".join([f"- {d}" for d in diagnostico])
    recomendaciones_txt = "\n".join([f"- {r}" for r in recomendaciones])

    prompt = f"""
Necesitamos ajustar {archivo_objetivo}.

Categoria:
{categoria}

Metricas actuales:
accepted={accepted}
coverage_util={coverage}
human_like={human}

Metas:
accepted>={accepted_meta}
coverage_util>={coverage_meta}
human_like>={human_meta}

Problema principal:
{diagnostico[0] if diagnostico else "sin_diagnostico"}

Diagnostico detectado:
{diagnostico_txt}

Cambio sugerido:
{cambio_sugerido}

Recomendaciones actuales:
{recomendaciones_txt}

Requisitos:
- no romper categorias existentes
- no bajar human_like
- mejorar accepted

Haz cambios en:

SEMANTIC_MAP
semantic_match()
seguimiento por foco
validacion respuestas cortas

No modificar:

autoplay_lab.py
stress_case_generator.py
run_stress_validation.py

Validar con:

python run_stress_validation.py
python export_stress_report.py
"""
    return prompt.strip()


def run_category_calibration(
    categoria: str,
    max_iterations: int = 5,
    seeds: Optional[List[int]] = None,
    n_runs: int = 20,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    seeds = [int(s) for s in (seeds or [123, 456])]
    categoria_slug = str(categoria or "producto_generico").strip().lower()

    all_cases = _load_cases()
    selected_cases = [case for case in all_cases if _extract_category(case) == categoria_slug]
    if not selected_cases:
        raise RuntimeError(f"No hay casos para categoria '{categoria_slug}' en lab_validation_suite_stress.json.")

    guides = _load_guides()
    iterations: List[Dict[str, Any]] = []
    runs_payload: Dict[str, List[Dict[str, Any]]] = {"accepted_runs": [], "review_runs": [], "rejected_runs": []}

    total_iterations = max(int(max_iterations), 1)
    total_batches_per_iter = len(selected_cases) * (len(seeds) + 1)
    total_batches = max(total_iterations * total_batches_per_iter, 1)
    completed_batches = 0
    total_units = max(total_iterations * total_batches_per_iter * max(int(n_runs), 1), 1)

    for it in range(1, total_iterations + 1):
        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "iteration_start",
                    "iteration": it,
                    "max_iterations": total_iterations,
                    "completed_batches": completed_batches,
                    "total_batches": total_batches,
                    "progress_ratio": round(completed_batches / total_batches, 4),
                }
            )

        def _on_validation_progress(evt: Dict[str, Any]) -> None:
            nonlocal completed_batches
            if not isinstance(evt, dict):
                return
            local_done = int(evt.get("completed_batches", 0) or 0)
            local_total = int(evt.get("total_batches_estimated", total_batches_per_iter) or total_batches_per_iter)
            base_done = (it - 1) * total_batches_per_iter
            completed_batches = min(base_done + local_done, total_batches)
            run_index = int(evt.get("run_index", 0) or 0)
            total_runs = int(evt.get("total_runs", max(int(n_runs), 1)) or max(int(n_runs), 1))
            unit_base = base_done * max(int(n_runs), 1)
            units_from_batches = completed_batches * max(int(n_runs), 1)
            units_from_run = unit_base + max(local_done - 1, 0) * max(int(n_runs), 1) + min(max(run_index, 0), total_runs)
            completed_units = max(units_from_batches, units_from_run) if run_index > 0 else units_from_batches
            completed_units = min(completed_units, total_units)
            ratio = round(completed_units / total_units, 4)
            if callable(progress_callback):
                progress_callback(
                    {
                        "phase": "validation_running",
                        "iteration": it,
                        "max_iterations": total_iterations,
                        "local_completed_batches": local_done,
                        "local_total_batches": local_total,
                        "completed_batches": completed_batches,
                        "total_batches": total_batches,
                        "progress_ratio": ratio,
                        "run_index": run_index,
                        "total_runs": total_runs,
                        "case_name": evt.get("case_name"),
                        "seed": evt.get("seed"),
                        "replicate": evt.get("replicate"),
                    }
                )

        validation = run_lab_validation_suite(
            cases=selected_cases,
            seeds=seeds,
            n_runs=int(max(n_runs, 1)),
            progress_callback=_on_validation_progress,
        )
        rows = [
            row
            for row in (validation.get("batch_comparison", []) if isinstance(validation, dict) else [])
            if str(row.get("categoria", "")).strip().lower() == categoria_slug
        ]
        summary = _aggregate_batch_rows(rows)
        summary["accepted"] = summary["accepted_count"]
        summary["review"] = summary["review_count"]
        summary["rejected"] = summary["rejected_count"]
        goals = _check_goals(categoria_slug, summary, guides)
        principal, archivo_objetivo, cambio_sugerido = derive_codex_action(goals.get("diagnostico", []))
        metricas_actuales = {
            "accepted": float(summary.get("accepted", 0.0)),
            "coverage_util": float(summary.get("avg_coverage_util", 0.0)),
            "human_like": float(summary.get("avg_human_like_index", 0.0)),
        }
        codex_prompt = build_codex_prompt(
            categoria=categoria_slug,
            diagnostico=goals.get("diagnostico", []),
            recomendaciones=goals.get("recomendaciones", []),
            metricas=metricas_actuales,
            metas=goals.get("metas", {}),
            archivo_objetivo=archivo_objetivo,
            cambio_sugerido=cambio_sugerido,
        )

        iter_payload = {
            "iteracion": it,
            "categoria": categoria_slug,
            "summary": summary,
            "cumple_metas": goals["cumple_metas"],
            "diagnostico": goals["diagnostico"],
            "recomendaciones": goals["recomendaciones"],
            "metas": goals["metas"],
            "principal_cuello_botella": principal,
            "archivo_objetivo": archivo_objetivo,
            "cambio_sugerido": cambio_sugerido,
            "codex_prompt_sugerido": codex_prompt,
            "raw_validation": validation,
        }
        iterations.append(iter_payload)

        runs = _collect_runs(validation)
        runs_payload["accepted_runs"].extend(runs["accepted_runs"][:20])
        runs_payload["review_runs"].extend(runs["review_runs"][:20])
        runs_payload["rejected_runs"].extend(runs["rejected_runs"][:20])

        if callable(progress_callback):
            progress_callback(
                {
                    "phase": "iteration_done",
                    "iteration": it,
                    "max_iterations": total_iterations,
                    "completed_batches": min(it * total_batches_per_iter, total_batches),
                    "total_batches": total_batches,
                    "progress_ratio": round(min(it * total_batches_per_iter, total_batches) / total_batches, 4),
                }
            )

    last = iterations[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = {
        "version": "auto_calibration_v1",
        "categoria": categoria_slug,
        "max_iterations": int(max_iterations),
        "seeds": seeds,
        "n_runs": int(n_runs),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "summary": last.get("summary", {}),
        "cumple_metas": bool(last.get("cumple_metas", False)),
        "diagnostico": last.get("diagnostico", []),
        "recomendaciones": last.get("recomendaciones", []),
        "metas": last.get("metas", {}),
        "principal_cuello_botella": last.get("principal_cuello_botella", "sin_diagnostico"),
        "archivo_objetivo": last.get("archivo_objetivo", "interview_engine.py"),
        "cambio_sugerido": last.get("cambio_sugerido", ""),
        "codex_prompt_sugerido": last.get("codex_prompt_sugerido", ""),
        "iterations": iterations,
        "accepted_runs": runs_payload["accepted_runs"][:50],
        "review_runs": runs_payload["review_runs"][:50],
        "rejected_runs": runs_payload["rejected_runs"][:50],
        "export_filename_sugerido": f"{categoria_slug}_auto_calibration_{ts}.json",
    }
    if callable(progress_callback):
        progress_callback(
            {
                "phase": "done",
                "iteration": total_iterations,
                "max_iterations": total_iterations,
                "completed_batches": total_batches,
                "total_batches": total_batches,
                "progress_ratio": 1.0,
            }
        )
    return out


if __name__ == "__main__":
    print(json.dumps({"ok": True, "module": "auto_calibrator"}, ensure_ascii=False))
