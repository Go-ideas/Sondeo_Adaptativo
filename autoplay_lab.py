import json
import random
import re
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from interview_engine import run_full, start_session


CONFUSION_MARKERS = [
    "no entendi",
    "puedes explicarlo mejor",
    "no estoy seguro",
    "a que te refieres",
]

BAD_EMERGENT_TOKENS = {"estoy", "seguro", "entiendo", "entiendas", "entienda", "refieres"}
GENERIC_RESPONSE_MARKERS = [
    "si eso falla",
    "no me late",
    "me gusta cuando",
    "pues sepa bien",
    "pues sepa rico",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", _normalize(value)).strip("_")
    return base[:80] if base else "proyecto"


def _case_identity(brief: Dict[str, Any]) -> Dict[str, str]:
    b = brief.get("brief", {}) if isinstance(brief, dict) else {}
    categoria_override_raw = ""
    if isinstance(brief, dict):
        categoria_override_raw = str(brief.get("categoria_estudio", "")).strip()
    if not categoria_override_raw and isinstance(b, dict):
        categoria_override_raw = str(b.get("categoria_estudio", "")).strip()
    categoria_override = _slugify(categoria_override_raw) if categoria_override_raw else ""

    text = _normalize(
        f"{b.get('antecedente', '')} {b.get('objetivo_principal', '')} "
        f"{brief.get('contexto', '') if isinstance(brief, dict) else ''} {categoria_override}"
    )
    if categoria_override:
        cat = categoria_override
    elif "universidad" in text or "universidades" in text:
        cat = "universidad"
    elif any(x in text for x in ["soda", "refresco", "refrescos", "bebida", "bebidas"]):
        cat = "soda" if any(x in text for x in ["soda", "refresco", "refrescos"]) else "bebida"
    elif any(x in text for x in ["auto", "vehiculo", "vehiculos"]):
        cat = "auto"
    elif any(x in text for x in ["banco", "bancario", "banca"]):
        cat = "banco"
    elif any(x in text for x in ["seguro", "aseguradora", "poliza"]):
        cat = "seguro"
    elif any(x in text for x in ["retail", "tienda", "supermercado", "autoservicio"]):
        cat = "retail"
    elif any(x in text for x in ["tecnologia", "tecnologico", "software", "app", "plataforma"]):
        cat = "tecnologia"
    else:
        cat = "producto_generico"
    objetivo = str(b.get("objetivo_principal", "")).strip() or str(b.get("antecedente", "")).strip()
    return {
        "categoria": cat,
        "slug_categoria": _slugify(cat),
        "slug_objetivo": _slugify(objetivo),
    }


def _extract_batch_compare_row(batch: Dict[str, Any], case_name: str, seed: int) -> Dict[str, Any]:
    summary = batch.get("summary", {}) if isinstance(batch, dict) else {}
    case_identity = batch.get("case_identity", {}) if isinstance(batch.get("case_identity", {}), dict) else {}
    return {
        "case_name": case_name,
        "seed": int(seed),
        "categoria": str(case_identity.get("categoria", "")),
        "perfil_respuesta_simulada": str(batch.get("perfil_respuesta_simulada", "")),
        "accepted_count": int(summary.get("accepted_count", 0)),
        "review_count": int(summary.get("review_count", 0)),
        "rejected_count": int(summary.get("rejected_count", 0)),
        "avg_coverage": float(summary.get("avg_coverage", 0.0)),
        "avg_coverage_util": float(summary.get("avg_coverage_util", 0.0)),
        "avg_naturalidad": float(summary.get("avg_naturalidad", 0.0)),
        "avg_human_like_index": float(summary.get("avg_human_like_index", 0.0)),
        "avg_question_diversity_index": float(summary.get("avg_question_diversity_index", 0.0)),
        "best_quality_score": float(summary.get("best_quality_score", 0.0)),
        "median_quality_score": float(summary.get("median_quality_score", 0.0)),
        "failure_counts": summary.get("failure_counts", {}),
    }


def _preview_runs(batch: Dict[str, Any]) -> Dict[str, Any]:
    reviews = batch.get("review_runs", []) if isinstance(batch, dict) else []
    rejected = batch.get("rejected_runs", []) if isinstance(batch, dict) else []

    def _to_preview(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            scores = item.get("quality_scores", {}) if isinstance(item.get("quality_scores", {}), dict) else {}
            out.append(
                {
                    "run_id": str(item.get("run_id", "")),
                    "quality_score_total": float(scores.get("quality_score_total", 0.0)),
                    "quality_failures": item.get("quality_failures", []),
                }
            )
        return out

    return {
        "top_3_review": _to_preview(reviews, 3),
        "top_2_rejected": _to_preview(rejected, 2),
    }


def _calc_stability_index(batch_a: Dict[str, Any], batch_b: Dict[str, Any]) -> float:
    a = batch_a.get("summary", {}) if isinstance(batch_a, dict) else {}
    b = batch_b.get("summary", {}) if isinstance(batch_b, dict) else {}

    n_runs = max(
        int(a.get("n_runs", 0) or 0),
        int(b.get("n_runs", 0) or 0),
        1,
    )

    diffs = [
        abs(float(a.get("accepted_rate", 0.0)) - float(b.get("accepted_rate", 0.0))),
        abs(float(a.get("avg_coverage", 0.0)) - float(b.get("avg_coverage", 0.0))),
        abs(float(a.get("avg_coverage_util", 0.0)) - float(b.get("avg_coverage_util", 0.0))),
        abs(float(a.get("avg_naturalidad", 0.0)) - float(b.get("avg_naturalidad", 0.0))),
        abs(float(a.get("avg_human_like_index", 0.0)) - float(b.get("avg_human_like_index", 0.0))),
        abs(float(a.get("avg_question_diversity_index", 0.0)) - float(b.get("avg_question_diversity_index", 0.0))),
        abs(float(a.get("best_quality_score", 0.0)) - float(b.get("best_quality_score", 0.0))),
        abs(float(a.get("median_quality_score", 0.0)) - float(b.get("median_quality_score", 0.0))),
        abs(int(a.get("accepted_count", 0)) - int(b.get("accepted_count", 0))) / float(n_runs),
        abs(int(a.get("review_count", 0)) - int(b.get("review_count", 0))) / float(n_runs),
        abs(int(a.get("rejected_count", 0)) - int(b.get("rejected_count", 0))) / float(n_runs),
    ]
    avg_diff = sum(diffs) / max(len(diffs), 1)
    return round(max(0.0, 1.0 - avg_diff), 3)


def _get_params(result: Dict[str, Any]) -> Dict[str, Any]:
    state = result.get("state", {}) if isinstance(result, dict) else {}
    plan = state.get("plan", {}) if isinstance(state, dict) else {}
    bp = plan.get("engine_blueprint", {}) if isinstance(plan, dict) else {}
    params = bp.get("parametros", {}) if isinstance(bp, dict) else {}
    return params if isinstance(params, dict) else {}


def score_autoplay_run(result: Dict[str, Any]) -> Dict[str, Any]:
    transcript = result.get("transcript", [])
    traces = result.get("traces", [])
    resumen = result.get("resumen_final", {})
    state = result.get("state", {})

    turnos_totales = len(transcript) if isinstance(transcript, list) else 0
    coverage_ratio_final = float(resumen.get("coverage_ratio", result.get("coverage_ratio", 0.0)) or 0.0)
    attrs_unicos = resumen.get("atributos_detectados_unicos", result.get("atributos_detectados_unicos", []))
    if not isinstance(attrs_unicos, list):
        attrs_unicos = []

    prof_map = resumen.get("profundidad_por_atributo", state.get("_profundidad_por_atributo", {}))
    if not isinstance(prof_map, dict):
        prof_map = {}
    attrs_profundos = [a for a, d in prof_map.items() if int(d) >= 2]

    nat_vals = [float(t.get("naturalidad_turno", 0.0)) for t in traces if isinstance(t, dict)]
    human_vals = [float(t.get("human_like_turno", 0.0)) for t in traces if isinstance(t, dict)]
    naturalidad_promedio = round(sum(nat_vals) / len(nat_vals), 3) if nat_vals else 0.0
    human_like_index = round(sum(human_vals) / len(human_vals), 3) if human_vals else 0.0

    confusion_turns = 0
    for item in transcript if isinstance(transcript, list) else []:
        if not isinstance(item, dict):
            continue
        ans = _normalize(str(item.get("respuesta", "")))
        if any(m in ans for m in CONFUSION_MARKERS):
            confusion_turns += 1
    porcentaje_turnos_confusion = round((confusion_turns / max(turnos_totales, 1)), 3)

    repeated_turns = 0
    similarities: List[float] = []
    prev_q = ""
    for item in transcript if isinstance(transcript, list) else []:
        if not isinstance(item, dict):
            continue
        q = _normalize(str(item.get("pregunta", "")))
        if prev_q and q:
            ratio = SequenceMatcher(None, q, prev_q).ratio()
            similarities.append(ratio)
            if ratio >= 0.95 and abs(len(q) - len(prev_q)) < (max(len(q), len(prev_q)) * 0.25):
                repeated_turns += 1
        prev_q = q
    porcentaje_turnos_repetidos = round((repeated_turns / max(turnos_totales - 1, 1)), 3)
    if len(similarities) == 0:
        question_diversity_index = 0.5
    else:
        question_diversity_index = round(
            1.0 - (sum(similarities) / len(similarities)),
            3,
        )
    avg_similarity = round(sum(similarities) / len(similarities), 3) if similarities else 0.0

    generic_turns = 0
    response_signatures: Dict[str, int] = {}
    for item in transcript if isinstance(transcript, list) else []:
        if not isinstance(item, dict):
            continue
        ans = _normalize(str(item.get("respuesta", "")))
        if any(m in ans for m in GENERIC_RESPONSE_MARKERS):
            generic_turns += 1
        sig = " ".join(ans.split()[:5])
        if sig:
            response_signatures[sig] = int(response_signatures.get(sig, 0)) + 1
    repeated_response_turns = sum((v - 1) for v in response_signatures.values() if v >= 3)
    generic_penalty = max(generic_turns, repeated_response_turns)
    porcentaje_respuestas_genericas = round(
        min(generic_penalty / max(turnos_totales, 1), 1.0),
        3,
    )

    params = _get_params(result)
    umbral = float(params.get("umbral_cierre", 0.8) or 0.8)
    min_turnos = int(params.get("min_turnos_antes_cierre", 6) or 6)
    min_attrs = int(params.get("min_atributos_confirmados", 2) or 2)
    motivo_cierre = str(resumen.get("motivo_cierre", result.get("motivo_cierre", ""))).strip().lower()
    closure_by_max = motivo_cierre == "max_preguntas"

    premature_closure = (
        (turnos_totales < min_turnos and not closure_by_max)
        or (coverage_ratio_final >= umbral and len(attrs_profundos) < min_attrs and not closure_by_max)
    )

    emergentes_invalidos = False
    for tr in traces if isinstance(traces, list) else []:
        if not isinstance(tr, dict):
            continue
        foco = _normalize(str(tr.get("foco", "")))
        if foco.startswith("emergente:"):
            emergentes_invalidos = True
            break
        if foco in BAD_EMERGENT_TOKENS:
            emergentes_invalidos = True
            break

    cobertura = resumen.get("cobertura", {}) if isinstance(resumen, dict) else {}
    target_operativo = int(cobertura.get("target_operativo", max(len(attrs_unicos), 1)) or max(len(attrs_unicos), 1))
    profundidad_relativa = round(min(len(attrs_profundos) / max(target_operativo, 1), 1.0), 3)
    penalizacion_baja_confusion_y_repeticion = round(max(0.0, 1.0 - ((porcentaje_turnos_confusion + porcentaje_turnos_repetidos) / 2.0)), 3)

    weak_counts: Dict[str, int] = {}
    strong_counts: Dict[str, int] = {}
    for tr in traces if isinstance(traces, list) else []:
        if not isinstance(tr, dict):
            continue
        for item in tr.get("atributos_items", []) if isinstance(tr.get("atributos_items", []), list) else []:
            if not isinstance(item, dict):
                continue
            attr = str(item.get("atributo", "")).strip()
            if not attr:
                continue
            if bool(item.get("depth_counted", False)):
                strong_counts[attr] = int(strong_counts.get(attr, 0)) + 1
            elif bool(item.get("weak_evidence", False)):
                weak_counts[attr] = int(weak_counts.get(attr, 0)) + 1

    atributos_con_profundidad_debil = 0
    for attr, depth in prof_map.items():
        if int(depth) <= 0:
            continue
        if int(weak_counts.get(attr, 0)) >= int(strong_counts.get(attr, 0)):
            atributos_con_profundidad_debil += 1

    atributos_utiles = [a for a, c in strong_counts.items() if int(c) >= 2]
    coverage_util = round(min(len(atributos_utiles) / max(target_operativo, 1), 1.0), 3)

    quality_score_total = round(
        (0.30 * max(min((coverage_ratio_final + coverage_util) / 2.0, 1.0), 0.0))
        + (0.25 * max(min(human_like_index, 1.0), 0.0))
        + (0.15 * max(min(naturalidad_promedio, 1.0), 0.0))
        + (0.10 * max(min(question_diversity_index, 1.0), 0.0))
        + (0.15 * profundidad_relativa)
        + (0.05 * min(max(penalizacion_baja_confusion_y_repeticion - porcentaje_respuestas_genericas, 0.0), 1.0)),
        3,
    )

    return {
        "coverage_ratio_final": round(coverage_ratio_final, 3),
        "naturalidad_promedio": naturalidad_promedio,
        "human_like_index": human_like_index,
        "turnos_totales": turnos_totales,
        "atributos_detectados_unicos": attrs_unicos,
        "atributos_con_profundidad_suficiente": len(attrs_profundos),
        "porcentaje_turnos_confusion": porcentaje_turnos_confusion,
        "porcentaje_turnos_repetidos": porcentaje_turnos_repetidos,
        "porcentaje_respuestas_genericas": porcentaje_respuestas_genericas,
        "generic_turns": int(generic_turns),
        "repeated_response_turns": int(repeated_response_turns),
        "question_diversity_index": question_diversity_index,
        "question_similarity_promedio": avg_similarity,
        "premature_closure": bool(premature_closure),
        "emergentes_invalidos": bool(emergentes_invalidos),
        "atributos_con_profundidad_debil": int(atributos_con_profundidad_debil),
        "coverage_util": coverage_util,
        "profundidad_relativa": profundidad_relativa,
        "penalizacion_baja_confusion_y_repeticion": penalizacion_baja_confusion_y_repeticion,
        "quality_score_total": quality_score_total,
    }


def classify_autoplay_run(scores: Dict[str, Any], state_or_result: Dict[str, Any]) -> str:
    coverage = float(scores.get("coverage_ratio_final", 0.0))
    naturalidad = float(scores.get("naturalidad_promedio", 0.0))
    human_like = float(scores.get("human_like_index", 0.0))
    prof_suf = int(scores.get("atributos_con_profundidad_suficiente", 0))
    confusion = float(scores.get("porcentaje_turnos_confusion", 1.0))
    repeated = float(scores.get("porcentaje_turnos_repetidos", 1.0))
    generic = float(scores.get("porcentaje_respuestas_genericas", 1.0))
    premature = bool(scores.get("premature_closure", True))
    emerg_bad = bool(scores.get("emergentes_invalidos", True))
    coverage_util = float(scores.get("coverage_util", 0.0))
    weak_depth = int(scores.get("atributos_con_profundidad_debil", 99))

    if (
        coverage >= 0.75
        and coverage_util >= 0.50
        and human_like >= 0.70
        and naturalidad >= 0.70
        and prof_suf >= 2
        and weak_depth <= 1
        and not premature
        and not emerg_bad
        and confusion <= 0.20
        and repeated <= 0.20
        and generic <= 0.35
    ):
        return "accepted"

    if (
        coverage >= 0.50
        and human_like >= 0.55
        and naturalidad >= 0.55
        and weak_depth <= 4
        and generic <= 0.75
        and confusion <= 0.40
        and repeated <= 0.40
        and not emerg_bad
        and not premature
    ):
        return "review"

    return "rejected"


def explain_run_failures(scores: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if float(scores.get("coverage_ratio_final", 0.0)) < 0.50:
        reasons.append("coverage baja")
    if float(scores.get("coverage_util", 0.0)) < 0.35:
        reasons.append("coverage util baja")
    if float(scores.get("naturalidad_promedio", 0.0)) < 0.55:
        reasons.append("naturalidad baja")
    if float(scores.get("human_like_index", 0.0)) < 0.55:
        reasons.append("human_like_index bajo")
    if int(scores.get("atributos_con_profundidad_suficiente", 0)) < 2:
        reasons.append("profundidad insuficiente")
    if int(scores.get("atributos_con_profundidad_debil", 0)) > 1:
        reasons.append("profundidad debil")
    if bool(scores.get("premature_closure", False)):
        reasons.append("cierre prematuro")
    if bool(scores.get("emergentes_invalidos", False)):
        reasons.append("emergentes invalidos")
    if float(scores.get("porcentaje_turnos_confusion", 0.0)) > 0.20:
        reasons.append("demasiada confusion")
    if float(scores.get("porcentaje_turnos_repetidos", 0.0)) > 0.20:
        reasons.append("preguntas repetidas")
    if float(scores.get("porcentaje_respuestas_genericas", 0.0)) > 0.35:
        reasons.append("demasiadas respuestas genericas")
    if float(scores.get("coverage_ratio_final", 0.0)) - float(scores.get("coverage_util", 0.0)) > 0.35:
        reasons.append("deteccion muy optimista")
    if float(scores.get("human_like_index", 0.0)) < 0.65 and float(scores.get("naturalidad_promedio", 0.0)) < 0.7:
        reasons.append("calidad conversacional artificial")
    return reasons


def _run_once(brief: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    state = start_session(deepcopy(brief), deepcopy(plan))
    return run_full(state)


def run_batch_autoplay(
    brief: Dict[str, Any],
    plan: Dict[str, Any],
    n_runs: int = 20,
    seed: Optional[int] = None,
    case_name: Optional[str] = None,
    perfil_respuesta_simulada: Optional[str] = None,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    accepted_runs: List[Dict[str, Any]] = []
    review_runs: List[Dict[str, Any]] = []
    rejected_runs: List[Dict[str, Any]] = []
    all_scores: List[Dict[str, Any]] = []
    case_identity = _case_identity(brief)
    slug_categoria = str(case_identity.get("slug_categoria", "proyecto")).strip() or "proyecto"
    resolved_case_name = str(case_name or "").strip() or str((brief or {}).get("case_name", "")).strip() or f"{slug_categoria}_case"
    resolved_profile = (
        str(perfil_respuesta_simulada or "").strip()
        or str((brief or {}).get("perfil_respuesta_simulada", "")).strip()
        or "default"
    )

    old_state = random.getstate()
    try:
        if seed is not None:
            random.seed(int(seed))

        total_runs = int(max(n_runs, 1))
        for i in range(1, total_runs + 1):
            if callable(progress_callback):
                progress_callback(
                    {
                        "phase": "run_start",
                        "run_index": i,
                        "total_runs": total_runs,
                        "progress_ratio": round((i - 1) / max(total_runs, 1), 4),
                    }
                )
            if seed is not None:
                random.seed(int(seed) + i)
            result = _run_once(brief, plan)
            scores = score_autoplay_run(result)
            quality_class = classify_autoplay_run(scores, result)
            failures = explain_run_failures(scores)

            run_payload = {
                "run_id": f"run_{i:03d}",
                "export_filename_sugerido": f"{slug_categoria}_autoplay_lab_v1_run_{i:03d}.json",
                "case_name": resolved_case_name,
                "seed": int(seed) if seed is not None else None,
                "perfil_respuesta_simulada": resolved_profile,
                "transcript": result.get("transcript", []),
                "traces": result.get("traces", []),
                "resumen_final": result.get("resumen_final", {}),
                "quality_scores": scores,
                "quality_class": quality_class,
                "quality_failures": failures,
            }
            all_scores.append(scores)
            if quality_class == "accepted":
                accepted_runs.append(run_payload)
            elif quality_class == "review":
                review_runs.append(run_payload)
            else:
                rejected_runs.append(run_payload)
            if callable(progress_callback):
                progress_callback(
                    {
                        "phase": "run_done",
                        "run_index": i,
                        "total_runs": total_runs,
                        "quality_class": quality_class,
                        "progress_ratio": round(i / max(total_runs, 1), 4),
                    }
                )
    finally:
        random.setstate(old_state)

    avg_coverage = round(
        sum(float(s.get("coverage_ratio_final", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_naturalidad = round(
        sum(float(s.get("naturalidad_promedio", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_human_like = round(
        sum(float(s.get("human_like_index", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_coverage_util = round(
        sum(float(s.get("coverage_util", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_generic = round(
        sum(float(s.get("porcentaje_respuestas_genericas", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_generic_turns = round(
        sum(float(s.get("generic_turns", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_repeated_response_turns = round(
        sum(float(s.get("repeated_response_turns", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )
    avg_question_diversity_index = round(
        sum(float(s.get("question_diversity_index", 0.0)) for s in all_scores) / max(len(all_scores), 1),
        3,
    )

    summary = {
        "n_runs": int(max(n_runs, 1)),
        "accepted_count": len(accepted_runs),
        "review_count": len(review_runs),
        "rejected_count": len(rejected_runs),
        "accepted_rate": round(len(accepted_runs) / max(int(max(n_runs, 1)), 1), 3),
        "avg_coverage": avg_coverage,
        "avg_naturalidad": avg_naturalidad,
        "avg_human_like_index": avg_human_like,
        "avg_coverage_util": avg_coverage_util,
        "avg_porcentaje_respuestas_genericas": avg_generic,
        "avg_generic_turns": avg_generic_turns,
        "avg_repeated_response_turns": avg_repeated_response_turns,
        "avg_question_diversity_index": avg_question_diversity_index,
    }

    accepted_runs.sort(key=lambda r: float((r.get("quality_scores", {}) or {}).get("quality_score_total", 0.0)), reverse=True)
    review_runs.sort(key=lambda r: float((r.get("quality_scores", {}) or {}).get("quality_score_total", 0.0)), reverse=True)
    rejected_runs.sort(key=lambda r: float((r.get("quality_scores", {}) or {}).get("quality_score_total", 0.0)), reverse=True)

    quality_vals = sorted(float((s or {}).get("quality_score_total", 0.0)) for s in all_scores)
    if quality_vals:
        mid = len(quality_vals) // 2
        if len(quality_vals) % 2 == 1:
            median_quality = quality_vals[mid]
        else:
            median_quality = round((quality_vals[mid - 1] + quality_vals[mid]) / 2.0, 3)
        best_quality = quality_vals[-1]
    else:
        median_quality = 0.0
        best_quality = 0.0

    failure_counts: Dict[str, int] = {}
    for run in accepted_runs + review_runs + rejected_runs:
        for reason in run.get("quality_failures", []) if isinstance(run, dict) else []:
            key = str(reason).strip()
            if not key:
                continue
            failure_counts[key] = int(failure_counts.get(key, 0)) + 1

    summary["best_quality_score"] = round(float(best_quality), 3)
    summary["median_quality_score"] = round(float(median_quality), 3)
    summary["failure_counts"] = dict(sorted(failure_counts.items(), key=lambda x: (-x[1], x[0])))

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return {
        "version": "autoplay_lab_v1",
        "case_identity": case_identity,
        "case_name": resolved_case_name,
        "seed": int(seed) if seed is not None else None,
        "perfil_respuesta_simulada": resolved_profile,
        "export_filename_sugerido": f"{slug_categoria}_autoplay_lab_v1_{ts}.json",
        "brief": brief,
        "plan": plan,
        "summary": summary,
        "accepted_runs": accepted_runs,
        "review_runs": review_runs,
        "rejected_runs": rejected_runs,
    }


def run_lab_validation_suite(
    cases: List[Dict[str, Any]],
    seeds: Optional[List[int]] = None,
    n_runs: int = 20,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    seeds = seeds or [123, 456]
    valid_cases = [c for c in cases if isinstance(c, dict)]
    outputs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(valid_cases, start=1):
        case_name = str(case.get("case_name", f"case_{idx}")).strip() or f"case_{idx}"
        brief = case.get("brief", {})
        plan = case.get("plan", {})
        perfil_respuesta_simulada = str(case.get("perfil_respuesta_simulada", "default")).strip() or "default"
        if not isinstance(brief, dict) or not isinstance(plan, dict):
            continue

        same_seed_runs: Dict[int, Dict[str, Any]] = {}
        for seed in seeds:
            def _on_batch_progress(evt: Dict[str, Any]) -> None:
                if not callable(progress_callback):
                    return
                payload = dict(evt) if isinstance(evt, dict) else {}
                payload["case_name"] = case_name
                payload["seed"] = int(seed)
                payload["replicate"] = 1
                payload["phase"] = payload.get("phase", "run_progress")
                progress_callback(payload)

            batch = run_batch_autoplay(
                brief=brief,
                plan=plan,
                n_runs=n_runs,
                seed=int(seed),
                case_name=case_name,
                perfil_respuesta_simulada=perfil_respuesta_simulada,
                progress_callback=_on_batch_progress,
            )
            batch["case_name"] = case_name
            batch["seed"] = int(seed)
            batch["perfil_respuesta_simulada"] = perfil_respuesta_simulada
            batch["run_preview"] = _preview_runs(batch)
            outputs.append(batch)
            comparisons.append(_extract_batch_compare_row(batch, case_name, int(seed)))
            if callable(progress_callback):
                progress_callback(
                    {
                        "case_name": case_name,
                        "seed": int(seed),
                        "replicate": 1,
                        "completed_batches": len(outputs),
                        "total_batches_estimated": len(valid_cases) * (len(seeds) + 1),
                    }
                )
            if int(seed) == int(seeds[0]):
                same_seed_runs[1] = batch

        # misma seed repetida para estabilidad
        repeat_seed = int(seeds[0])
        def _on_repeat_progress(evt: Dict[str, Any]) -> None:
            if not callable(progress_callback):
                return
            payload = dict(evt) if isinstance(evt, dict) else {}
            payload["case_name"] = case_name
            payload["seed"] = repeat_seed
            payload["replicate"] = 2
            payload["phase"] = payload.get("phase", "run_progress")
            progress_callback(payload)

        repeat_batch = run_batch_autoplay(
            brief=brief,
            plan=plan,
            n_runs=n_runs,
            seed=repeat_seed,
            case_name=case_name,
            perfil_respuesta_simulada=perfil_respuesta_simulada,
            progress_callback=_on_repeat_progress,
        )
        repeat_batch["case_name"] = case_name
        repeat_batch["seed"] = repeat_seed
        repeat_batch["perfil_respuesta_simulada"] = perfil_respuesta_simulada
        repeat_batch["replicate"] = 2
        repeat_batch["run_preview"] = _preview_runs(repeat_batch)
        outputs.append(repeat_batch)
        comparisons.append(_extract_batch_compare_row(repeat_batch, case_name, repeat_seed))
        if callable(progress_callback):
            progress_callback(
                {
                    "case_name": case_name,
                    "seed": repeat_seed,
                    "replicate": 2,
                    "completed_batches": len(outputs),
                    "total_batches_estimated": len(valid_cases) * (len(seeds) + 1),
                }
            )
        same_seed_runs[2] = repeat_batch

        stability = _calc_stability_index(same_seed_runs.get(1, {}), same_seed_runs.get(2, {}))
        stability_rows.append(
            {
                "case_name": case_name,
                "seed": repeat_seed,
                "stability_index": stability,
                "batch_a": {
                    "accepted_count": int((same_seed_runs.get(1, {}).get("summary", {}) or {}).get("accepted_count", 0)),
                    "review_count": int((same_seed_runs.get(1, {}).get("summary", {}) or {}).get("review_count", 0)),
                    "rejected_count": int((same_seed_runs.get(1, {}).get("summary", {}) or {}).get("rejected_count", 0)),
                },
                "batch_b": {
                    "accepted_count": int((same_seed_runs.get(2, {}).get("summary", {}) or {}).get("accepted_count", 0)),
                    "review_count": int((same_seed_runs.get(2, {}).get("summary", {}) or {}).get("review_count", 0)),
                    "rejected_count": int((same_seed_runs.get(2, {}).get("summary", {}) or {}).get("rejected_count", 0)),
                },
            }
        )

    avg_stability = round(
        sum(float(r.get("stability_index", 0.0)) for r in stability_rows) / max(len(stability_rows), 1),
        3,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return {
        "version": "autoplay_lab_validation_v1",
        "export_filename_sugerido": f"lab_validation_suite_{ts}.json",
        "n_cases": len(valid_cases),
        "n_batches": len(outputs),
        "seeds": [int(s) for s in seeds],
        "n_runs_per_batch": int(max(n_runs, 1)),
        "stability_index_promedio": avg_stability,
        "batch_comparison": comparisons,
        "stability_rows": stability_rows,
        "batches": outputs,
    }


if __name__ == "__main__":
    print(json.dumps({"ok": True, "module": "autoplay_lab"}, ensure_ascii=False))
