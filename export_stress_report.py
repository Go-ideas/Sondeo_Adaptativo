import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _safe_mean(values: List[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def _group_rows(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        bucket = str(row.get(key, "desconocido")).strip() or "desconocido"
        grouped.setdefault(bucket, []).append(row)
    return grouped


def _summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "avg_accepted": _safe_mean([float(r.get("accepted_count", 0.0)) for r in rows]),
        "avg_review": _safe_mean([float(r.get("review_count", 0.0)) for r in rows]),
        "avg_rejected": _safe_mean([float(r.get("rejected_count", 0.0)) for r in rows]),
        "avg_quality_score_total": _safe_mean([float(r.get("median_quality_score", 0.0)) for r in rows]),
        "avg_coverage": _safe_mean([float(r.get("avg_coverage", 0.0)) for r in rows]),
        "avg_human_like": _safe_mean([float(r.get("avg_human_like_index", 0.0)) for r in rows]),
        "avg_coverage_util": _safe_mean([float(r.get("avg_coverage_util", 0.0)) for r in rows]),
    }


def _build_diagnostics(rows: List[Dict[str, Any]], stability_rows: List[Dict[str, Any]]) -> List[str]:
    notes: List[str] = []
    if not rows:
        return ["No hay datos para diagnostico."]

    by_profile = _group_rows(rows, "perfil_respuesta_simulada")
    by_category = _group_rows(rows, "categoria")
    profile_summary = {k: _summarize_group(v) for k, v in by_profile.items()}
    category_summary = {k: _summarize_group(v) for k, v in by_category.items()}

    claro = profile_summary.get("claro")
    vago = profile_summary.get("vago")
    confuso = profile_summary.get("confuso")
    redundante = profile_summary.get("redundante")
    if claro and vago and claro["avg_quality_score_total"] > vago["avg_quality_score_total"] + 0.08:
        notes.append("El motor funciona mejor con perfiles claros que con perfiles vagos.")
    if confuso and confuso["avg_review"] > confuso["avg_accepted"]:
        notes.append("Los perfiles confusos disparan mayor review que accepted.")
    if redundante and redundante["avg_rejected"] >= 4:
        notes.append("Los perfiles redundantes generan mayor rechazo y posible repeticion.")

    if category_summary:
        top_cat = max(category_summary.items(), key=lambda kv: kv[1]["avg_coverage_util"])
        low_cat = min(category_summary.items(), key=lambda kv: kv[1]["avg_coverage_util"])
        if top_cat[0] != low_cat[0]:
            notes.append(
                f"La categoria {low_cat[0]} tiene menor coverage_util que {top_cat[0]}."
            )

    stability_avg = _safe_mean([float(r.get("stability_index", 0.0)) for r in stability_rows])
    if stability_avg >= 0.85:
        notes.append("La estabilidad con misma seed es alta.")
    elif stability_avg >= 0.65:
        notes.append("La estabilidad con misma seed es moderada.")
    else:
        notes.append("La estabilidad con misma seed es baja y requiere revisar aleatoriedad.")

    if not notes:
        notes.append("Sin hallazgos fuertes; se recomienda ampliar la bateria.")
    return notes


def _recommendation(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "requiere ajustes antes de Fase 2"
    accepted_total = sum(float(r.get("accepted_count", 0.0)) for r in rows)
    review_total = sum(float(r.get("review_count", 0.0)) for r in rows)
    rejected_total = sum(float(r.get("rejected_count", 0.0)) for r in rows)
    total = max(accepted_total + review_total + rejected_total, 1.0)
    accepted_rate = accepted_total / total
    avg_human_like = _safe_mean([float(r.get("avg_human_like_index", 0.0)) for r in rows])
    avg_coverage_util = _safe_mean([float(r.get("avg_coverage_util", 0.0)) for r in rows])
    if accepted_rate >= 0.70 and avg_human_like >= 0.75 and avg_coverage_util >= 0.60:
        return "listo para Fase 2"
    return "requiere ajustes antes de Fase 2"


def main() -> None:
    in_path = Path("stress_validation_result.json")
    if not in_path.exists():
        raise FileNotFoundError("No existe stress_validation_result.json. Ejecuta run_stress_validation.py primero.")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    rows = data.get("batch_comparison", [])
    stability_rows = data.get("stability_rows", [])
    if not isinstance(rows, list):
        rows = []
    if not isinstance(stability_rows, list):
        stability_rows = []

    stability_map = {str(r.get("case_name", "")): float(r.get("stability_index", 0.0)) for r in stability_rows if isinstance(r, dict)}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row["stability_index"] = stability_map.get(str(row.get("case_name", "")), 0.0)

    # CSV por batch
    csv_path = Path("stress_validation_report.csv")
    fieldnames = [
        "case_name",
        "categoria",
        "perfil_respuesta_simulada",
        "seed",
        "accepted_count",
        "review_count",
        "rejected_count",
        "avg_coverage",
        "avg_coverage_util",
        "avg_naturalidad",
        "avg_human_like_index",
        "avg_question_diversity_index",
        "best_quality_score",
        "median_quality_score",
        "stability_index",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Agregados
    by_profile = _group_rows(rows, "perfil_respuesta_simulada")
    by_category = _group_rows(rows, "categoria")
    profile_summary = {k: _summarize_group(v) for k, v in by_profile.items()}
    category_summary = {k: _summarize_group(v) for k, v in by_category.items()}
    diagnostics = _build_diagnostics(rows, stability_rows)
    recommendation = _recommendation(rows)

    # Top/Bottom casos por score mediano
    sorted_rows = sorted(rows, key=lambda r: float(r.get("median_quality_score", 0.0)), reverse=True)
    best_cases = sorted_rows[:5]
    worst_cases = list(reversed(sorted_rows[-5:])) if sorted_rows else []

    # Markdown legible
    md_path = Path("stress_validation_report.md")
    lines: List[str] = []
    lines.append("# Stress Test del Laboratorio")
    lines.append("## Resumen general")
    lines.append(f"- batches evaluados: {len(rows)}")
    lines.append(f"- casos evaluados: {int(data.get('n_cases', 0))}")
    lines.append(f"- seeds: {data.get('seeds', [])}")
    lines.append(f"- n_runs por batch: {data.get('n_runs_per_batch', 0)}")
    lines.append(f"- stability_index_promedio: {data.get('stability_index_promedio', 0.0)}")

    lines.append("## Resultados por categoria")
    for cat, summary in sorted(category_summary.items(), key=lambda kv: kv[0]):
        lines.append(
            f"- {cat}: accepted={summary['avg_accepted']}, review={summary['avg_review']}, "
            f"rejected={summary['avg_rejected']}, coverage={summary['avg_coverage']}, "
            f"coverage_util={summary['avg_coverage_util']}, human_like={summary['avg_human_like']}"
        )

    lines.append("## Resultados por perfil de respuesta")
    for profile, summary in sorted(profile_summary.items(), key=lambda kv: kv[0]):
        lines.append(
            f"- {profile}: accepted={summary['avg_accepted']}, review={summary['avg_review']}, "
            f"rejected={summary['avg_rejected']}, quality={summary['avg_quality_score_total']}, "
            f"coverage={summary['avg_coverage']}, human_like={summary['avg_human_like']}"
        )

    lines.append("## Casos con mejor desempeno")
    for row in best_cases:
        lines.append(
            f"- {row.get('case_name')} (seed={row.get('seed')}): median_quality={row.get('median_quality_score')}, "
            f"best_quality={row.get('best_quality_score')}, coverage_util={row.get('avg_coverage_util')}"
        )

    lines.append("## Casos con peor desempeno")
    for row in worst_cases:
        lines.append(
            f"- {row.get('case_name')} (seed={row.get('seed')}): median_quality={row.get('median_quality_score')}, "
            f"best_quality={row.get('best_quality_score')}, coverage_util={row.get('avg_coverage_util')}"
        )

    lines.append("## Diagnostico automatico")
    for note in diagnostics:
        lines.append(f"- {note}")

    lines.append("## Recomendacion final:")
    lines.append(f"- {recommendation}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(csv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
