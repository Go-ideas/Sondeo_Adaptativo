import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
try:
    import pandas as pd
except Exception:
    pd = None

from interview_engine import run_full, start_session, step as step_turn, step_with_human_answer
from autoplay_lab import run_batch_autoplay
from auto_calibrator import run_category_calibration
from calibration_storage import load_guides as load_calibration_guides
from calibration_storage import load_history as load_calibration_history
from calibration_storage import (
    ensure_trained_categories_structure,
    evaluate_category_publishability,
    is_category_available_for_client,
    load_published_categories,
    load_trained_category,
    save_published_categories,
)
from calibration_storage import save_iteration_result as save_calibration_iteration
from plan_visual_trabajo import (
    crear_categoria_proyecto,
    generate_definiciones_operativas_ai,
    generar_atributos_iniciales,
    generar_plan_visual,
    generar_primera_pregunta_inicio,
    generar_suite_laboratorio,
    identidad_caso,
)


def load_api_key_from_file(path: str = "tokenkey.txt") -> str:
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"No se encontro {path}")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"{path} esta vacio")
    return key


def limpiar_texto_input(texto: str) -> str:
    value = str(texto or "").strip()
    if not value:
        return ""
    value = value.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    value = re.sub(r"^\s*\\?['\"]+", "", value)
    value = re.sub(r"\\?['\"]+\s*$", "", value)
    value = re.sub(r"[ \t]+", " ", value).strip()
    return value


def parse_definiciones_operativas(raw_text: str) -> Dict[str, Dict[str, str]]:
    """
    Formato esperado por linea:
    atributo | que_cuenta | que_no_cuenta
    """
    definiciones: Dict[str, Dict[str, str]] = {}
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split("|")]
        if len(parts) != 3:
            raise ValueError(
                f"Linea {idx} invalida en definiciones operativas. Usa: atributo | que_cuenta | que_no_cuenta"
            )
        atributo, que_cuenta, que_no_cuenta = [limpiar_texto_input(p) for p in parts]
        if not atributo or not que_cuenta or not que_no_cuenta:
            raise ValueError(f"Linea {idx} tiene campos vacios en definiciones operativas")
        definiciones[atributo] = {"que_cuenta": que_cuenta, "que_no_cuenta": que_no_cuenta}
    return definiciones


def definiciones_to_raw_lines(definiciones: Dict[str, Dict[str, str]]) -> str:
    lines: List[str] = []
    for atributo, item in definiciones.items():
        if not isinstance(item, dict):
            continue
        attr = limpiar_texto_input(atributo)
        qc = limpiar_texto_input(item.get("que_cuenta", ""))
        qnc = limpiar_texto_input(item.get("que_no_cuenta", ""))
        if not attr:
            continue
        lines.append(f"{attr} | {qc} | {qnc}")
    return "\n".join(lines)


def build_brief(
    antecedente: str,
    objetivo_principal: str,
    tipo_sesion: str,
    target_atributos: int,
    lista_atributos: List[str],
    definiciones_operativas: Dict[str, Dict[str, str]],
    definiciones_operativas_fuente: str,
    max_preguntas: int,
    numero_entrevistas_planeadas: int,
    profundidad: str,
    permitir_atributos_emergentes: bool,
    max_atributos_emergentes: int,
    evitar_sugestivas: bool,
    no_preguntas_temporales: bool,
    rango_edad: str,
    rangos_edad_objetivo: List[str],
    genero: str,
    perfil_participante: str,
    perfiles_objetivo: List[str],
    nivel_conocimiento: str,
    nivel_profundizacion: str,
    probabilidad_confusion: float,
    primera_pregunta_modo: str,
    primera_pregunta_manual: str,
) -> Dict[str, Any]:
    antecedente = limpiar_texto_input(antecedente)
    objetivo_principal = limpiar_texto_input(objetivo_principal)
    tipo_sesion = limpiar_texto_input(tipo_sesion)
    primera_pregunta_manual = limpiar_texto_input(primera_pregunta_manual)
    lista_atributos = [limpiar_texto_input(x) for x in lista_atributos if limpiar_texto_input(x)]
    definiciones_clean: Dict[str, Dict[str, str]] = {}
    for atributo, item in (definiciones_operativas or {}).items():
        if not isinstance(item, dict):
            continue
        attr = limpiar_texto_input(atributo)
        if not attr:
            continue
        definiciones_clean[attr] = {
            "que_cuenta": limpiar_texto_input(item.get("que_cuenta", "")),
            "que_no_cuenta": limpiar_texto_input(item.get("que_no_cuenta", "")),
        }

    return {
        "brief": {
            "antecedente": antecedente,
            "objetivo_principal": objetivo_principal,
            "tipo_sesion": tipo_sesion,
        },
        "atributos_objetivo": {
            "target": target_atributos,
            "lista": lista_atributos,
        },
        "definiciones_operativas": definiciones_clean,
        "config": {
            "max_preguntas": max_preguntas,
            "numero_entrevistas_planeadas": numero_entrevistas_planeadas,
            "profundidad": profundidad,
            "permitir_atributos_emergentes": permitir_atributos_emergentes,
            "max_atributos_emergentes": max_atributos_emergentes,
            "definiciones_operativas_fuente": definiciones_operativas_fuente,
            "probabilidad_confusion": float(probabilidad_confusion),
            "primera_pregunta_modo": primera_pregunta_modo,
            "primera_pregunta_manual": primera_pregunta_manual,
        },
        "target_participante": {
            "rango_edad": rango_edad,
            "rangos_edad": rangos_edad_objetivo,
            "genero": genero,
            "perfil": perfil_participante,
            "perfiles": perfiles_objetivo,
            "nivel_conocimiento": nivel_conocimiento,
            "nivel_profundizacion": nivel_profundizacion,
        },
        "guardrails": {
            "evitar_sugestivas": evitar_sugestivas,
            "no_preguntas_temporales": no_preguntas_temporales,
        },
    }


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return base[:60] if base else "proyecto"


def _project_slug(brief: Dict[str, Any]) -> str:
    try:
        ident = identidad_caso(brief if isinstance(brief, dict) else {})
        slug = str(ident.get("slug_caso", "")).strip()
        if slug:
            return _slugify(slug)
        cat = str(ident.get("slug_categoria", "")).strip()
        if cat:
            return _slugify(cat)
    except Exception:
        pass
    b = brief.get("brief", {}) if isinstance(brief, dict) else {}
    objetivo = str(b.get("objetivo_principal", "")).strip()
    tipo = str(b.get("tipo_sesion", "")).strip()
    if objetivo:
        return _slugify(objetivo)
    if tipo:
        return _slugify(tipo)
    return "proyecto"


def _build_semantic_seed_preview(categoria: str, atributos: List[str]) -> Dict[str, Any]:
    def norm(txt: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (txt or "").lower())).strip()

    stop = {"que", "como", "con", "por", "para", "una", "uno", "las", "los", "del", "de", "la", "el"}
    by_attr: Dict[str, List[str]] = {}
    flat: Dict[str, str] = {}
    for attr in atributos:
        toks = [t for t in norm(attr).split() if len(t) >= 3 and t not in stop]
        toks = list(dict.fromkeys(toks))[:8]
        by_attr[attr] = toks
        for tk in toks:
            flat.setdefault(tk, norm(attr))
    return {"categoria": _slugify(categoria), "keywords_por_atributo": by_attr, "semantic_map_seed": flat}


def _df(rows: List[Dict[str, Any]]):
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def _parse_seeds(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out or [123, 456]


def _calib_latest(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return sorted(rows, key=lambda r: int(r.get("iteracion", 0)))[-1]


def _calib_prev(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("iteracion", 0)))
    if len(ordered) < 2:
        return {}
    return ordered[-2]


def _delta(curr: Dict[str, Any], prev: Dict[str, Any], key: str) -> str:
    if not curr or not prev:
        return "n/a"
    try:
        d = float(curr.get(key, 0.0)) - float(prev.get(key, 0.0))
        return f"{d:+.3f}"
    except Exception:
        return "n/a"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _delta_num(curr: Dict[str, Any], prev: Dict[str, Any], key: str) -> float:
    if not curr or not prev:
        return 0.0
    return _to_float(curr.get(key, 0.0)) - _to_float(prev.get(key, 0.0))


def _get_calib_targets(guides: Dict[str, Any], categoria: str) -> Dict[str, float]:
    cats = guides.get("categories", {}) if isinstance(guides, dict) else {}
    if not isinstance(cats, dict):
        return {"accepted_min": 0.25, "coverage_util_min": 0.35, "human_like_min": 0.70}
    current = cats.get(str(categoria).strip().lower(), cats.get("producto_generico", {}))
    if not isinstance(current, dict):
        current = {}
    return {
        "accepted_min": _to_float(current.get("accepted_min", 0.25), 0.25),
        "coverage_util_min": _to_float(current.get("coverage_util_min", 0.35), 0.35),
        "human_like_min": _to_float(current.get("human_like_min", 0.70), 0.70),
    }


def _metric_actual(current: Dict[str, Any], result: Dict[str, Any], key: str) -> float:
    if isinstance(current, dict) and key in current:
        return _to_float(current.get(key, 0.0), 0.0)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    if not isinstance(summary, dict):
        return 0.0
    if key == "coverage_util":
        return _to_float(summary.get("avg_coverage_util", summary.get("coverage_util", 0.0)), 0.0)
    if key == "human_like":
        return _to_float(summary.get("avg_human_like_index", summary.get("human_like", 0.0)), 0.0)
    return _to_float(summary.get(key, 0.0), 0.0)


def _build_calib_summary(current: Dict[str, Any], previous: Dict[str, Any], targets: Dict[str, float]) -> Dict[str, Any]:
    accepted_gap = _to_float(current.get("accepted", 0.0)) - _to_float(targets.get("accepted_min", 0.0))
    cov_gap = _to_float(current.get("coverage_util", 0.0)) - _to_float(targets.get("coverage_util_min", 0.0))
    human_gap = _to_float(current.get("human_like", 0.0)) - _to_float(targets.get("human_like_min", 0.0))
    gap_pairs = {
        "Accepted": accepted_gap,
        "Coverage util": cov_gap,
        "Human like": human_gap,
    }
    bottleneck = min(gap_pairs.items(), key=lambda kv: kv[1])[0] if gap_pairs else "-"
    bottleneck = str(current.get("principal_cuello_botella", bottleneck) or bottleneck)
    deltas = {
        "accepted": _delta_num(current, previous, "accepted"),
        "coverage_util": _delta_num(current, previous, "coverage_util"),
        "human_like": _delta_num(current, previous, "human_like"),
    }
    best_improvement = max(deltas.items(), key=lambda kv: kv[1])[0] if deltas else "-"
    recommendations = current.get("recomendaciones", [])
    if not isinstance(recommendations, list):
        recommendations = []
    next_recommendation = str(recommendations[0]) if recommendations else "Sin recomendacion critica."
    raw_result = current.get("raw_result", {}) if isinstance(current, dict) else {}
    if not isinstance(raw_result, dict):
        raw_result = {}
    archivo_objetivo = str(current.get("archivo_objetivo", raw_result.get("archivo_objetivo", "interview_engine.py")) or "interview_engine.py")
    cambio_sugerido = str(current.get("cambio_sugerido", raw_result.get("cambio_sugerido", "")) or "")
    codex_prompt_sugerido = str(current.get("codex_prompt_sugerido", raw_result.get("codex_prompt_sugerido", "")) or "")
    return {
        "estado_final": "calibrado" if bool(current.get("cumple_metas", False)) else "pendiente",
        "cumple_metas": bool(current.get("cumple_metas", False)),
        "principal_cuello_botella": bottleneck,
        "mejoria_vs_previa": (
            f"Accepted {deltas['accepted']:+.3f} | Coverage util {deltas['coverage_util']:+.3f} | Human like {deltas['human_like']:+.3f}"
        ),
        "siguiente_recomendacion": next_recommendation,
        "archivo_objetivo": archivo_objetivo,
        "cambio_sugerido": cambio_sugerido,
        "codex_prompt_disponible": bool(codex_prompt_sugerido.strip()),
        "deltas": deltas,
    }


def _case_score(run: Dict[str, Any], key: str) -> float:
    qs = run.get("quality_scores", {}) if isinstance(run, dict) else {}
    if not isinstance(qs, dict):
        qs = {}
    return _to_float(qs.get(key, 0.0), 0.0)


def _case_row(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": run.get("run_id", "-"),
        "case_name": run.get("case_name", "-"),
        "seed": run.get("seed", "-"),
        "perfil_respuesta_simulada": run.get("perfil_respuesta_simulada", "-"),
        "quality_score": round(_case_score(run, "quality_score_total"), 3),
        "coverage_util": round(_case_score(run, "coverage_util"), 3),
        "human_like": round(_case_score(run, "human_like_index"), 3),
    }


def _run_transcript_lines(run: Dict[str, Any]) -> List[str]:
    transcript = run.get("transcript", []) if isinstance(run, dict) else []
    if not isinstance(transcript, list):
        return []
    lines: List[str] = []
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("pregunta", "")).strip()
        a = str(turn.get("respuesta", "")).strip()
        idx = int(turn.get("turn_index", 0) or 0)
        if q or a:
            lines.append(f"Turno {idx} | P: {q} | R: {a}")
    return lines


def _run_trace_rows(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    traces = run.get("traces", []) if isinstance(run, dict) else []
    if not isinstance(traces, list):
        return []
    rows: List[Dict[str, Any]] = []
    for tr in traces:
        if not isinstance(tr, dict):
            continue
        attrs = tr.get("atributos_detectados", [])
        if not isinstance(attrs, list):
            attrs = []
        rows.append(
            {
                "turn_index": tr.get("turn_index", "-"),
                "accion": tr.get("accion", "-"),
                "regla_disparada": tr.get("regla_disparada", "-"),
                "foco": tr.get("foco", "-"),
                "atributos_detectados": ", ".join([str(x) for x in attrs]) or "-",
            }
        )
    return rows


def _diagnostico_meaning(code: str) -> str:
    mapping = {
        "accepted_bajo": "Pocas corridas estan cumpliendo criterio objetivo.",
        "coverage_util_baja": "Se cubren atributos pero sin profundidad util suficiente.",
        "human_like_bajo": "La entrevista suena poco natural o rigida.",
        "sin_diagnostico": "No se detecto un cuello de botella claro.",
    }
    return mapping.get(str(code), "Falla detectada por criterios de calibracion.")


def _pick_calib_field(current: Dict[str, Any], result_payload: Dict[str, Any], key: str, default: Any = "") -> Any:
    if isinstance(current, dict) and current.get(key) not in [None, ""]:
        return current.get(key)
    raw = current.get("raw_result", {}) if isinstance(current, dict) else {}
    if isinstance(raw, dict) and raw.get(key) not in [None, ""]:
        return raw.get(key)
    if isinstance(result_payload, dict) and result_payload.get(key) not in [None, ""]:
        return result_payload.get(key)
    return default


def render_stress_validation() -> None:
    st.subheader("Stress Validation")
    default_path = "stress_validation_result.json"
    path = st.text_input("Archivo de resultado", value=default_path, key="stress_result_path")
    if not Path(path).exists():
        st.warning("No se encontro el archivo de stress validation.")
        return
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        return

    st.write(f"Version: {payload.get('version', '-')}")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Casos", int(payload.get("n_cases", 0) or 0))
    s2.metric("Batches", int(payload.get("n_batches", 0) or 0))
    s3.metric("Runs por batch", int(payload.get("n_runs_per_batch", 0) or 0))
    s4.metric("Stability promedio", float(payload.get("stability_index_promedio", 0.0) or 0.0))

    batch_cmp = payload.get("batch_comparison", [])
    st.markdown("### Comparativo por batch")
    st.dataframe(_df(batch_cmp), use_container_width=True)


def render_calibration_lab(
    categoria: str,
    max_iterations: int,
    seeds_text: str,
    n_runs: int,
    run_clicked: bool,
    load_clicked: bool,
) -> None:
    st.subheader("Calibration Lab")
    if "calib_execution_log" not in st.session_state:
        st.session_state["calib_execution_log"] = []
    if "calib_last_summary" not in st.session_state:
        st.session_state["calib_last_summary"] = {}

    st.markdown("### Como funciona esta calibracion")
    st.info(
        "\n".join(
            [
                "1) corre autoplay para la categoria",
                "2) evalua resultados contra guias",
                "3) detecta fallas",
                "4) genera recomendaciones",
                "5) guarda historico",
                "6) compara contra la iteracion previa",
            ]
        )
    )

    guides = load_calibration_guides()

    if load_clicked:
        st.session_state["calib_history_rows"] = load_calibration_history(categoria)
        st.success("Historico cargado.")

    rows = st.session_state.get("calib_history_rows", [])
    current = _calib_latest(rows)
    previous = _calib_prev(rows)
    result = st.session_state.get("calib_result", {})
    targets = _get_calib_targets(guides, categoria)

    accepted_actual = _metric_actual(current, result, "accepted")
    coverage_util_actual = _metric_actual(current, result, "coverage_util")
    human_like_actual = _metric_actual(current, result, "human_like")

    st.markdown("### Objetivo actual de entrenamiento")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Categoria", categoria or "-")
    g2.metric("Accepted actual", f"{accepted_actual:.3f}")
    g3.metric("Coverage util actual", f"{coverage_util_actual:.3f}")
    g4.metric("Human like actual", f"{human_like_actual:.3f}")
    goal_rows = [
        {
            "indicador": "accepted",
            "actual": round(accepted_actual, 3),
            "meta": round(_to_float(targets.get("accepted_min", 0.0)), 3),
            "gap": round(accepted_actual - _to_float(targets.get("accepted_min", 0.0)), 3),
        },
        {
            "indicador": "coverage_util",
            "actual": round(coverage_util_actual, 3),
            "meta": round(_to_float(targets.get("coverage_util_min", 0.0)), 3),
            "gap": round(coverage_util_actual - _to_float(targets.get("coverage_util_min", 0.0)), 3),
        },
        {
            "indicador": "human_like",
            "actual": round(human_like_actual, 3),
            "meta": round(_to_float(targets.get("human_like_min", 0.0)), 3),
            "gap": round(human_like_actual - _to_float(targets.get("human_like_min", 0.0)), 3),
        },
    ]
    st.dataframe(_df(goal_rows), use_container_width=True, hide_index=True)

    if run_clicked:
        try:
            st.session_state["calib_execution_log"] = []
            global_progress_ph = st.empty()
            iteration_progress_ph = st.empty()
            run_progress_ph = st.empty()
            status_ph = st.empty()
            now_ph = st.empty()

            global_bar = global_progress_ph.progress(0.0, text="Barra global de calibracion: 0.0%")
            iter_bar = iteration_progress_ph.progress(0.0, text="Barra de iteracion actual: 0.0%")
            run_bar = run_progress_ph.progress(0.0, text="Barra de batch/run actual: 0.0%")
            status_ph.caption("Iteracion 0 de 0 | fase: esperando ejecucion")
            now_ph.info("Que esta haciendo ahora: preparando calibracion.")

            def _on_calibration_progress(evt: Dict[str, Any]) -> None:
                if not isinstance(evt, dict):
                    return

                ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
                log_evt = {
                    "timestamp": ts,
                    "phase": str(evt.get("phase", "running")),
                    "iteration": int(evt.get("iteration", 0) or 0),
                    "case_name": str(evt.get("case_name", "") or ""),
                    "seed": evt.get("seed"),
                    "run_index": int(evt.get("run_index", 0) or 0),
                    "total_runs": int(evt.get("total_runs", 0) or 0),
                    "progress_ratio": round(_to_float(evt.get("progress_ratio", 0.0), 0.0), 4),
                }
                execution_log = st.session_state.get("calib_execution_log", [])
                if not isinstance(execution_log, list):
                    execution_log = []
                execution_log.append(log_evt)
                st.session_state["calib_execution_log"] = execution_log[-500:]

                ratio_global = min(max(_to_float(evt.get("progress_ratio", 0.0), 0.0), 0.0), 1.0)
                phase = str(evt.get("phase", "running"))
                it = int(evt.get("iteration", 0) or 0)
                it_max = int(evt.get("max_iterations", 0) or 0)
                done = int(evt.get("completed_batches", 0) or 0)
                total = int(evt.get("total_batches", 0) or 0)
                case_name = str(evt.get("case_name", "") or "")
                seed = evt.get("seed")
                run_idx = int(evt.get("run_index", 0) or 0)
                run_total = int(evt.get("total_runs", 0) or 0)
                local_done = int(evt.get("local_completed_batches", 0) or 0)
                local_total = int(evt.get("local_total_batches", 0) or 0)

                if local_total > 0:
                    ratio_iter = min(max(local_done / max(local_total, 1), 0.0), 1.0)
                elif phase == "iteration_done":
                    ratio_iter = 1.0
                elif phase == "iteration_start":
                    ratio_iter = 0.0
                else:
                    ratio_iter = min(max(run_idx / max(run_total, 1), 0.0), 1.0) if run_total > 0 else 0.0
                ratio_run = min(max(run_idx / max(run_total, 1), 0.0), 1.0) if run_total > 0 else 0.0

                global_bar.progress(
                    ratio_global,
                    text=f"Barra global de calibracion: {ratio_global*100:.1f}% | iteracion {it} de {it_max}",
                )
                iter_bar.progress(
                    ratio_iter,
                    text=f"Barra de iteracion actual: {ratio_iter*100:.1f}% | batches completados {done} de {total}",
                )
                run_bar.progress(
                    ratio_run,
                    text=f"Barra de batch/run actual: {ratio_run*100:.1f}% | run actual {run_idx} de {run_total}",
                )
                status_ph.caption(
                    f"iteracion {it} de {it_max} | fase actual: {phase} | caso actual: {case_name or '-'} | "
                    f"seed actual: {seed if seed is not None else '-'} | run actual {run_idx} de {run_total} | "
                    f"batches completados {done} de {total}"
                )
                now_ph.info(
                    f"Que esta haciendo ahora: evaluando caso {case_name or '-'} con seed {seed if seed is not None else '-'}, "
                    f"run {run_idx}/{run_total if run_total > 0 else 0}."
                )

            result = run_category_calibration(
                categoria=str(categoria),
                max_iterations=int(max_iterations),
                seeds=_parse_seeds(seeds_text),
                n_runs=int(n_runs),
                progress_callback=_on_calibration_progress,
            )
            global_bar.progress(1.0, text="Barra global de calibracion: 100%")
            iter_bar.progress(1.0, text="Barra de iteracion actual: 100%")
            run_bar.progress(1.0, text="Barra de batch/run actual: 100%")

            st.session_state["calib_result"] = result
            saved = save_calibration_iteration(categoria, result)
            st.session_state["calib_history_rows"] = load_calibration_history(categoria)
            rows_latest = st.session_state.get("calib_history_rows", [])
            curr_latest = _calib_latest(rows_latest)
            prev_latest = _calib_prev(rows_latest)
            targets_latest = _get_calib_targets(guides, categoria)
            st.session_state["calib_last_summary"] = _build_calib_summary(curr_latest, prev_latest, targets_latest)
            st.success(f"Calibracion completada. Iteracion guardada: {saved.get('iteracion')}.")

            st.session_state["calib_result"] = st.session_state.get("calib_result", {})
            st.session_state["calib_history_rows"] = st.session_state.get("calib_history_rows", [])
            st.session_state["calib_execution_log"] = st.session_state.get("calib_execution_log", [])
            st.session_state["calib_last_summary"] = st.session_state.get("calib_last_summary", {})
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible ejecutar calibracion: {exc}")

    rows = st.session_state.get("calib_history_rows", [])
    current = _calib_latest(rows)
    previous = _calib_prev(rows)
    result = st.session_state.get("calib_result", {})
    raw_result = current.get("raw_result", {}) if isinstance(current, dict) else {}
    result_payload = result if isinstance(result, dict) and result else (raw_result if isinstance(raw_result, dict) else {})
    archivo_objetivo = str(_pick_calib_field(current, result_payload, "archivo_objetivo", "interview_engine.py") or "interview_engine.py")
    cambio_sugerido = str(_pick_calib_field(current, result_payload, "cambio_sugerido", "") or "")
    codex_prompt_sugerido = str(_pick_calib_field(current, result_payload, "codex_prompt_sugerido", "") or "")

    tab_dash, tab_hist, tab_diag, tab_cases, tab_process, tab_guides = st.tabs(
        ["Dashboard", "Historico", "Diagnostico", "Casos", "Proceso", "Guias"]
    )

    with tab_dash:
        c1, c2, c3 = st.columns(3)
        c1.metric("Categoria", categoria or "-")
        c2.metric("Ultima iteracion", int(current.get("iteracion", 0) or 0))
        c3.metric("Cumple metas", "Si" if bool(current.get("cumple_metas", False)) else "No")
        st.caption("Metricas en escala 0-1. En historico se guardan como promedio por iteracion.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Accepted rate", f"{_to_float(current.get('accepted', 0.0)):.3f}", _delta(current, previous, "accepted"))
        m1.caption("Accepted: corridas que ya cumplen criterio objetivo.")
        m2.metric("Review rate", f"{_to_float(current.get('review', 0.0)):.3f}", _delta(current, previous, "review"))
        m2.caption("Review: corridas que aun requieren revision.")
        m3.metric("Rejected rate", f"{_to_float(current.get('rejected', 0.0)):.3f}", _delta(current, previous, "rejected"))
        m3.caption("Rejected: corridas que fallan criterio.")

        m4, m5, m6 = st.columns(3)
        m4.metric("Coverage", f"{_to_float(current.get('coverage', 0.0)):.3f}", _delta(current, previous, "coverage"))
        m4.caption("Coverage: cobertura total de atributos.")
        m5.metric("Coverage util", f"{_to_float(current.get('coverage_util', 0.0)):.3f}", _delta(current, previous, "coverage_util"))
        m5.caption("Coverage util: cobertura con profundidad util.")
        m6.metric("Human like", f"{_to_float(current.get('human_like', 0.0)):.3f}", _delta(current, previous, "human_like"))
        m6.caption("Human like: que tan humano se siente el moderador.")

    with tab_hist:
        if not rows:
            st.info("No hay historico para la categoria seleccionada.")
        else:
            useful_cols = [
                "categoria",
                "iteracion",
                "accepted",
                "review",
                "rejected",
                "coverage",
                "coverage_util",
                "human_like",
                "cumple_metas",
                "fecha",
            ]
            clean_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                clean_rows.append({k: row.get(k) for k in useful_cols})
            st.dataframe(_df(clean_rows), use_container_width=True, hide_index=True)

            latest_hist = _calib_latest(rows)
            if isinstance(latest_hist, dict) and latest_hist:
                with st.expander("Detalle tecnico Codex (ultima iteracion)", expanded=False):
                    st.write(f"principal_cuello_botella: {latest_hist.get('principal_cuello_botella', '')}")
                    st.write(f"archivo_objetivo: {latest_hist.get('archivo_objetivo', '')}")
                    st.write(f"cambio_sugerido: {latest_hist.get('cambio_sugerido', '')}")
                    st.code(str(latest_hist.get("codex_prompt_sugerido", "") or ""), language="text")

            with st.expander("Raw result tecnico", expanded=False):
                for row in sorted([r for r in rows if isinstance(r, dict)], key=lambda r: int(r.get("iteracion", 0) or 0)):
                    it = int(row.get("iteracion", 0) or 0)
                    with st.expander(f"Iteracion {it}", expanded=False):
                        st.json(row.get("raw_result", {}), expanded=False)

            if pd is not None and clean_rows:
                dfr = pd.DataFrame(clean_rows).sort_values("iteracion")
                if "iteracion" in dfr.columns:
                    dfr = dfr.set_index("iteracion")
                if "accepted" in dfr.columns:
                    st.markdown("### evolucion de accepted")
                    st.line_chart(dfr[["accepted"]])
                if "coverage_util" in dfr.columns:
                    st.markdown("### evolucion de coverage_util")
                    st.line_chart(dfr[["coverage_util"]])
                if "human_like" in dfr.columns:
                    st.markdown("### evolucion de human_like")
                    st.line_chart(dfr[["human_like"]])

    with tab_diag:
        st.markdown("### Diagnostico principal")
        diagnostico = current.get("diagnostico", []) if isinstance(current, dict) else []
        if not isinstance(diagnostico, list):
            diagnostico = []
        principal_diag = str(_pick_calib_field(current, result_payload, "principal_cuello_botella", "sin_diagnostico") or "sin_diagnostico")
        st.write(principal_diag)

        st.markdown("### Que significa")
        st.write(_diagnostico_meaning(principal_diag))

        st.markdown("### Recomendaciones")
        recomendaciones = current.get("recomendaciones", []) if isinstance(current, dict) else []
        if not isinstance(recomendaciones, list):
            recomendaciones = []
        if recomendaciones:
            for rec in recomendaciones:
                st.markdown(f"- {rec}")
        else:
            st.markdown("- Sin recomendaciones criticas.")

        st.markdown("### Archivo objetivo")
        st.write(current.get("archivo_objetivo", "-") or archivo_objetivo or "-")

        st.markdown("### Cambio sugerido")
        st.write(current.get("cambio_sugerido", "-") or cambio_sugerido or "-")

        st.markdown("### Prompt sugerido para Codex VS Code")
        codex_prompt = str(current.get("codex_prompt_sugerido", "") or codex_prompt_sugerido or "")
        if codex_prompt:
            st.code(codex_prompt, language="text")
            st.download_button(
                "Descargar prompt Codex",
                data=codex_prompt,
                file_name="codex_prompt.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_codex_prompt_{current.get('categoria', 'categoria')}_{int(current.get('iteracion', 0) or 0)}",
            )
            if st.button("Copiar prompt sugerido", key=f"calib_copy_prompt_{int(current.get('iteracion', 0) or 0)}"):
                st.session_state["calib_prompt_buffer"] = codex_prompt
                st.success("Prompt listo. Copialo desde el bloque de codigo.")
        else:
            st.info("Aún no hay prompt sugerido.")

        accepted_delta = _delta_num(current, previous, "accepted")
        cov_delta = _delta_num(current, previous, "coverage_util")
        human_delta = _delta_num(current, previous, "human_like")
        st.markdown("### Cambio vs iteracion previa")
        st.markdown(f"Accepted: {':green' if accepted_delta >= 0 else ':red'}[{accepted_delta:+.3f}]")
        st.markdown(f"Coverage util: {':green' if cov_delta >= 0 else ':red'}[{cov_delta:+.3f}]")
        st.markdown(f"Human like: {':green' if human_delta >= 0 else ':red'}[{human_delta:+.3f}]")

    with tab_cases:
        accepted_runs = result_payload.get("accepted_runs", []) if isinstance(result_payload, dict) else []
        review_runs = result_payload.get("review_runs", []) if isinstance(result_payload, dict) else []
        rejected_runs = result_payload.get("rejected_runs", []) if isinstance(result_payload, dict) else []
        if not isinstance(accepted_runs, list):
            accepted_runs = []
        if not isinstance(review_runs, list):
            review_runs = []
        if not isinstance(rejected_runs, list):
            rejected_runs = []

        all_runs = [r for r in (accepted_runs + review_runs + rejected_runs) if isinstance(r, dict)]
        if not all_runs:
            st.info("Ejecuta calibracion para ver casos.")
        else:
            st.caption("Tabla resumida de corridas. Los detalles se muestran por caso en expanders.")
            case_rows = [_case_row(run) for run in all_runs]
            st.dataframe(_df(case_rows), use_container_width=True, hide_index=True)

            max_expanders = min(len(all_runs), 25)
            st.caption(f"Mostrando detalles de {max_expanders} casos para mantener velocidad de render.")
            for run in all_runs[:max_expanders]:
                run_id = str(run.get("run_id", "-"))
                case_name = str(run.get("case_name", "-"))
                with st.expander(f"Caso {case_name} | run_id {run_id}", expanded=False):
                    transcript_lines = _run_transcript_lines(run)
                    trace_rows = _run_trace_rows(run)
                    with st.expander("Ver transcript resumido", expanded=False):
                        if transcript_lines:
                            for line in transcript_lines:
                                st.write(line)
                        else:
                            st.write("Sin transcript disponible.")
                    with st.expander("Ver trazas resumidas", expanded=False):
                        if trace_rows:
                            st.dataframe(_df(trace_rows), use_container_width=True, hide_index=True)
                        else:
                            st.write("Sin trazas disponibles.")

    with tab_process:
        st.markdown("### Proceso de ejecucion")
        log_rows = st.session_state.get("calib_execution_log", [])
        if not isinstance(log_rows, list):
            log_rows = []
        if not log_rows:
            st.info("Aun no hay eventos de progreso.")
        else:
            cols = ["timestamp", "phase", "iteration", "case_name", "seed", "run_index", "total_runs", "progress_ratio"]
            latest_events = []
            for event in log_rows[-20:]:
                if not isinstance(event, dict):
                    continue
                latest_events.append({k: event.get(k) for k in cols})
            st.dataframe(_df(latest_events), use_container_width=True, hide_index=True)

    with tab_guides:
        st.caption("Estas son las metas que la categoria debe cumplir para considerarse calibrada.")
        cats = guides.get("categories", {}) if isinstance(guides, dict) else {}
        rows_guides = []
        if isinstance(cats, dict):
            for cat_name, payload in cats.items():
                p = payload if isinstance(payload, dict) else {}
                rows_guides.append(
                    {
                        "categoria": cat_name,
                        "accepted_min": _to_float(p.get("accepted_min", 0.0)),
                        "coverage_util_min": _to_float(p.get("coverage_util_min", 0.0)),
                        "human_like_min": _to_float(p.get("human_like_min", 0.0)),
                    }
                )
        st.dataframe(_df(rows_guides), use_container_width=True, hide_index=True)

    last_summary = st.session_state.get("calib_last_summary", {})
    if isinstance(last_summary, dict) and last_summary:
        st.markdown("### Resumen final post-ejecucion")
        with st.container(border=True):
            summary_card = {
                "estado_final": last_summary.get("estado_final", "-"),
                "cumple_metas": bool(last_summary.get("cumple_metas", False)),
                "principal_cuello_botella": last_summary.get("principal_cuello_botella", "-"),
                "archivo_objetivo": last_summary.get("archivo_objetivo", "-"),
                "cambio_sugerido": last_summary.get("cambio_sugerido", "-"),
                "codex_prompt_disponible": bool(last_summary.get("codex_prompt_disponible", False)),
            }
            st.json(summary_card, expanded=True)
            st.caption(f"Mejoria vs iteracion previa: {last_summary.get('mejoria_vs_previa', '-')}")


def render_plan(plan: Dict[str, Any]) -> None:
    st.subheader(plan.get("titulo_plan", "Plan"))
    st.caption(f"Version: {plan.get('version', 'N/A')}")

    client_plan = plan.get("client_plan", {})
    engine_blueprint = plan.get("engine_blueprint", {})

    st.markdown("### Resumen ejecutivo")
    for bullet in client_plan.get("resumen_ejecutivo", []):
        st.markdown(f"- {bullet}")

    target_res = client_plan.get("target_participante_resumen", {})
    if isinstance(target_res, dict) and target_res:
        st.markdown("### Target participante simulado")
        rangos = target_res.get("rangos_edad", [])
        perfiles = target_res.get("perfiles", [])
        rangos_txt = ", ".join([str(x) for x in rangos]) if isinstance(rangos, list) and rangos else target_res.get("rango_edad", "-")
        perfiles_txt = ", ".join([str(x) for x in perfiles]) if isinstance(perfiles, list) and perfiles else target_res.get("perfil", "-")
        st.write(
            f"Edad objetivo: {rangos_txt} | Genero: {target_res.get('genero','-')} | "
            f"Perfiles objetivo: {perfiles_txt}"
        )
        st.write(
            f"Perfil autoplay: {target_res.get('perfil','-')} | "
            f"Conocimiento: {target_res.get('nivel_conocimiento','-')} | "
            f"Profundizacion: {target_res.get('nivel_profundizacion','-')} | "
            f"Prob. confusion: {target_res.get('probabilidad_confusion', 0.15)}"
        )

    estilo = client_plan.get("estilo_moderacion_aplicado", {})
    if isinstance(estilo, dict) and estilo:
        st.markdown("### Estilo de moderacion aplicado")
        st.write(
            f"Lenguaje: {estilo.get('lenguaje','-')} | Max palabras/pregunta: {estilo.get('max_palabras_pregunta','-')} | "
            f"Tono: {estilo.get('tono','-')} | Evita abstracciones: {estilo.get('evitar_abstracciones', False)}"
        )
        ejemplos = client_plan.get("preguntas_ejemplo_estilo", {})
        st.write(f"**Primera pregunta final:** {client_plan.get('primera_pregunta_inicio', '-')}")
        st.write(f"**Ejemplo profundizacion:** {ejemplos.get('profundizar', '-')}")
        st.write(f"**Ejemplo reformulacion:** {ejemplos.get('reformular', '-')}")

    st.markdown("### Etapas visuales (cliente)")
    for etapa in client_plan.get("etapas_visuales", []):
        with st.container(border=True):
            st.write(f"**Paso {etapa.get('paso')} - {etapa.get('nombre', '')}**")
            st.write(f"**Proposito:** {etapa.get('proposito', '')}")
            for c in etapa.get("criterios_exito", []):
                st.markdown(f"- {c}")
            st.info(etapa.get("pregunta_guia", ""))

    st.markdown("### Medicion resumida")
    medicion = client_plan.get("medicion_resumida", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target declarado", medicion.get("target_declarado", "-"))
    m2.metric("Target operativo", medicion.get("target_operativo", "-"))
    umbral = medicion.get("umbral_cierre")
    m3.metric("Umbral cierre", f"{umbral:.2f}" if isinstance(umbral, (int, float)) else "-")
    m4.metric("Max preguntas", medicion.get("max_preguntas", "-"))
    st.write(medicion.get("como_se_calcula", ""))
    if isinstance(medicion.get("target_declarado"), int) and isinstance(medicion.get("target_operativo"), int):
        if medicion["target_declarado"] != medicion["target_operativo"]:
            st.warning("Hay inconsistencia: ajustar target o lista.")

    primera_pregunta = client_plan.get("primera_pregunta_inicio", "")
    if primera_pregunta:
        st.markdown("### Primera pregunta sugerida")
        st.info(primera_pregunta)

    st.markdown("### Engine blueprint")
    st.markdown("#### Ciclo por turno")
    ciclo = engine_blueprint.get("ciclo_por_turno", [])
    if ciclo:
        st.code(" -> ".join(ciclo), language="text")

    with st.expander("Reglas de decision"):
        for r in engine_blueprint.get("reglas_de_decision", []):
            st.write(
                f"[{r.get('prioridad','-')}] {r.get('id','-')} | {r.get('si','')} -> {r.get('accion','')} ({r.get('entonces','')})"
            )

    with st.expander("Templates por accion"):
        st.json(engine_blueprint.get("templates_por_accion", {}), expanded=False)

    with st.expander("Parametros"):
        st.json(engine_blueprint.get("parametros", {}), expanded=False)

    with st.expander("Guardrails"):
        for g in engine_blueprint.get("guardrails", []):
            st.markdown(f"- {g}")

    st.markdown("### Validaciones del blueprint")
    for val in engine_blueprint.get("validaciones", []):
        estado = val.get("estado", "")
        icon = "OK" if estado == "ok" else "WARN"
        criticidad = val.get("criticidad", "-")
        st.write(f"`{icon}` [{criticidad}] **{val.get('regla', '')}**: {val.get('detalle', '')}")

    st.markdown("### Pendientes")
    pendientes = plan.get("pendientes", [])
    if not pendientes:
        st.success("No hay pendientes. El brief esta completo.")
    else:
        for pendiente in pendientes:
            with st.container(border=True):
                st.write(f"**Campo faltante:** {pendiente.get('campo_faltante', '')}")
                st.write(f"**Por que importa:** {pendiente.get('por_que_importa', '')}")
                st.write(f"**Pregunta para cliente:** {pendiente.get('pregunta_para_cliente', '')}")


def render_autoplay(state: Dict[str, Any]) -> None:
    st.markdown("### Cobertura actual")
    ratio = float(state.get("coverage_ratio", 0.0) or 0.0)
    foco_actual = state.get("foco_actual", "")
    traces = state.get("traces", [])
    naturalidad_vals = [float(t.get("naturalidad_turno", 0.0)) for t in traces if isinstance(t, dict)]
    human_vals = [float(t.get("human_like_turno", 0.0)) for t in traces if isinstance(t, dict)]
    naturalidad_prom = round(sum(naturalidad_vals) / len(naturalidad_vals), 3) if naturalidad_vals else 0.0
    human_like_index = round(sum(human_vals) / len(human_vals), 3) if human_vals else 0.0
    profundidad_map = state.get("_profundidad_por_atributo", {})
    attrs_depth_ok = [a for a, d in (profundidad_map.items() if isinstance(profundidad_map, dict) else []) if int(d) >= 2]
    cierres_evitados = int(state.get("premature_closure_prevented", 0))

    st.progress(min(max(ratio, 0.0), 1.0), text=f"Cobertura: {ratio:.0%}")
    st.write(f"**Foco actual:** {foco_actual or '-'}")
    st.caption(
        f"Naturalidad promedio: {naturalidad_prom} | Human-like index: {human_like_index} | "
        f"Atributos con profundidad >=2: {len(attrs_depth_ok)} | Cierres prematuros evitados: {cierres_evitados}"
    )

    st.markdown("### Indicadores de calidad")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Naturalidad promedio", f"{naturalidad_prom:.3f}")
    q2.metric("Human-like index", f"{human_like_index:.3f}")
    q3.metric("Atributos profundidad>=2", len(attrs_depth_ok))
    q4.metric("Cierres prematuros evitados", cierres_evitados)
    q5, q6 = st.columns(2)
    q5.metric("Turn index", int(state.get("turn_index", 0) or 0))
    q6.metric("Coverage ratio", f"{ratio:.3f}")

    if traces:
        last_trace = traces[-1] if isinstance(traces[-1], dict) else {}
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Naturalidad ultimo turno", f"{float(last_trace.get('naturalidad_turno', 0.0)):.2f}")
        l2.metric("Human-like ultimo turno", f"{float(last_trace.get('human_like_turno', 0.0)):.2f}")
        l3.metric("Profundidad foco actual", int(last_trace.get("profundidad_foco_actual", 0) or 0))
        l4.metric("Salto forzado atributo", "Si" if bool(last_trace.get("salto_forzado_atributo", False)) else "No")

        with st.expander("Detalle de calidad por turno"):
            for tr in traces:
                if not isinstance(tr, dict):
                    continue
                st.write(
                    f"Turno {tr.get('turn_index','-')}: "
                    f"naturalidad={tr.get('naturalidad_turno','-')} | "
                    f"human_like={tr.get('human_like_turno','-')} | "
                    f"no_repeticion={tr.get('no_repeticion','-')} | "
                    f"profundidad_turno={tr.get('profundidad_turno','-')} | "
                    f"continuidad={tr.get('continuidad_contextual','-')}"
                )
    cva, cvb = st.columns(2)
    with cva:
        st.write("**Atributos detectados**")
        detected = state.get("atributos_detectados_unicos", [])
        if detected:
            for a in detected:
                st.markdown(f"- {a}")
        else:
            st.write("-")
    with cvb:
        st.write("**Atributos pendientes**")
        pending = state.get("atributos_pendientes", [])
        if pending:
            for a in pending:
                st.markdown(f"- {a}")
        else:
            st.write("-")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Transcript")
        transcript = state.get("transcript", [])
        if not transcript:
            st.info("Aun no hay turnos ejecutados.")
        for item in transcript:
            with st.container(border=True):
                st.write(f"**Turno {item.get('turn_index')} - Q:** {item.get('pregunta', '')}")
                st.write(f"**A:** {item.get('respuesta', '')}")

    with col_right:
        st.markdown("### Decision del moderador IA")
        if not traces:
            st.info("Aun no hay trazas.")
        for trace in traces:
            with st.expander(f"Turno {trace.get('turn_index')} | {trace.get('accion')}"):
                st.markdown(f"**Regla disparada:** `{trace.get('regla_disparada')}`")
                st.markdown(f"**Etapa actual:** `{trace.get('etapa_actual')}`")
                st.write(f"**Etapa:** {trace.get('etapa_actual')}")
                st.write(f"**Accion:** {trace.get('accion')}")
                st.write(f"**Regla:** {trace.get('regla_disparada')}")
                if "queja_repeticion" in str(trace.get("regla_disparada", "")):
                    st.warning("Se detecto queja por repeticion y se cambio la estrategia de pregunta.")
                st.write(f"**Foco:** {trace.get('foco')}")
                st.write(f"**Atributos:** {', '.join(trace.get('atributos_detectados', [])) or '-'}")
                items = trace.get("atributos_items", [])
                if items:
                    st.write("**Atributos items (cuenta/confianza):**")
                    for it in items:
                        st.write(
                            f"- {it.get('atributo','-')} | cuenta={it.get('cuenta', False)} | "
                            f"confianza={it.get('confianza','-')} | evidencia={it.get('evidencia','')}"
                        )
                cov = trace.get("cobertura", {})
                st.write(
                    f"**Cobertura:** detectados={cov.get('detectados', 0)} | "
                    f"target_decl={cov.get('target_declarado', 0)} | "
                    f"target_op={cov.get('target_operativo', 0)} | ratio={cov.get('ratio', 0)}"
                )
                st.write(
                    f"**Naturalidad turno:** {trace.get('naturalidad_turno', '-')}"
                    f" | **Salto forzado atributo:** {trace.get('salto_forzado_atributo', False)}"
                )
                st.write(
                    f"**Human-like turno:** {trace.get('human_like_turno', '-')}"
                    f" | **Profundidad foco actual:** {trace.get('profundidad_foco_actual', '-')}"
                )
                st.write(f"**Razon corta:** {trace.get('razon_corta')}")
                st.write(f"**Evidencia:** {trace.get('evidencia')}")


BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_CONFIG_FILE = BASE_DIR / "deployment_config.json"
CLIENT_SESSIONS_DIR = BASE_DIR / "client_sessions"
TRAINED_CATEGORIES_DIR = BASE_DIR / "trained_categories"


def load_deployment_config() -> Dict[str, Any]:
    default_cfg = {
        "app_mode": "client",
        "default_category": "universidad",
        "show_admin_tools": False,
        "allow_category_selection": True,
        "save_client_sessions": True,
    }
    if not DEPLOYMENT_CONFIG_FILE.exists():
        return default_cfg
    try:
        payload = json.loads(DEPLOYMENT_CONFIG_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_cfg
        merged = dict(default_cfg)
        merged.update(payload)
        return merged
    except Exception:
        return default_cfg


def _all_known_categories() -> List[str]:
    categories: List[str] = []
    if TRAINED_CATEGORIES_DIR.exists():
        for child in TRAINED_CATEGORIES_DIR.iterdir():
            if child.is_dir():
                categories.append(child.name.strip().lower())
    guides = load_calibration_guides()
    guide_cats = guides.get("categories", {}) if isinstance(guides, dict) else {}
    if isinstance(guide_cats, dict):
        categories.extend([str(k).strip().lower() for k in guide_cats.keys() if str(k).strip()])
    published = load_published_categories()
    categories.extend([str(c).strip().lower() for c in published.get("categorias_publicadas", []) if str(c).strip()])
    metadata = published.get("metadata", {}) if isinstance(published, dict) else {}
    if isinstance(metadata, dict):
        categories.extend([str(k).strip().lower() for k in metadata.keys() if str(k).strip()])
    return sorted(list(dict.fromkeys([c for c in categories if c])))


def _latest_metrics_for_category(categoria: str) -> Dict[str, Any]:
    history = load_calibration_history(categoria)
    if history:
        return history[-1] if isinstance(history[-1], dict) else {}
    try:
        trained = load_trained_category(categoria)
        metrics = trained.get("latest_metrics", {})
        return metrics if isinstance(metrics, dict) else {}
    except Exception:
        return {}


def _save_client_session(categoria: str, state: Dict[str, Any], trained_payload: Dict[str, Any]) -> Path:
    cat = str(categoria or "categoria").strip().lower() or "categoria"
    target_dir = CLIENT_SESSIONS_DIR / cat
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{cat}_{ts}_{int(state.get('turn_index', 0) or 0)}"
    payload = {
        "session_id": session_id,
        "categoria": cat,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "brief": trained_payload.get("brief", {}),
        "plan_v3_2": trained_payload.get("plan", {}),
        "transcript": state.get("transcript", []),
        "traces": state.get("traces", []),
        "resumen_final": {
            "done": bool(state.get("done", False)),
            "turnos": int(state.get("turn_index", 0) or 0),
            "coverage_ratio": float(state.get("coverage_ratio", 0.0) or 0.0),
            "motivo_cierre": str(state.get("final_reason", "")),
        },
    }
    out_path = target_dir / f"{ts}_{session_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def render_client_mode(deployment_cfg: Dict[str, Any]) -> None:
    st.title("Encuesta de prueba")
    st.caption("Experiencia de entrevista para cliente")

    published = load_published_categories()
    available = [c for c in published.get("categorias_publicadas", []) if is_category_available_for_client(c, published)]
    metadata = published.get("metadata", {}) if isinstance(published, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    default_category = str(deployment_cfg.get("default_category", "universidad")).strip().lower()
    if default_category and default_category in available:
        selected_default = default_category
    elif available:
        selected_default = available[0]
    else:
        selected_default = default_category

    if "client_selected_category" not in st.session_state:
        st.session_state["client_selected_category"] = selected_default
    if "client_state" not in st.session_state:
        st.session_state["client_state"] = None
    if "client_session_saved_path" not in st.session_state:
        st.session_state["client_session_saved_path"] = ""
    if "client_go_to_categories" not in st.session_state:
        st.session_state["client_go_to_categories"] = False

    if st.session_state.pop("client_go_to_categories", False):
        st.session_state["client_nav_view"] = "Categorias disponibles"

    view = st.sidebar.radio(
        "Secciones",
        options=["Inicio", "Categorias disponibles", "Probar encuesta", "Acerca del estudio"],
        index=0,
        key="client_nav_view",
    )

    if view == "Inicio":
        st.subheader("Inicio")
        st.write("Este sitio permite probar una encuesta cualitativa moderada por IA.")
        if st.button("Comenzar prueba", type="primary", use_container_width=True, key="btn_client_start"):
            st.session_state["client_go_to_categories"] = True
            st.rerun()
        return

    if view == "Categorias disponibles":
        st.subheader("Categorias disponibles")
        if not available:
            st.warning("No hay categorias publicadas para cliente en este momento.")
            return
        for categoria in available:
            md = metadata.get(categoria, {}) if isinstance(metadata.get(categoria, {}), dict) else {}
            display_name = str(md.get("display_name", categoria)).strip() or categoria
            st.markdown(f"- **{display_name}** (`{categoria}`)")
            if md.get("description"):
                st.caption(str(md.get("description")))
        if bool(deployment_cfg.get("allow_category_selection", True)):
            chosen = st.selectbox(
                "Selecciona categoria",
                options=available,
                index=max(available.index(st.session_state["client_selected_category"]), 0) if st.session_state["client_selected_category"] in available else 0,
                key="client_category_selector",
            )
            st.session_state["client_selected_category"] = str(chosen)
        else:
            st.info(f"Categoria fija para cliente: {st.session_state['client_selected_category']}")
        return

    if view == "Acerca del estudio":
        st.subheader("Acerca del estudio")
        st.write("La encuesta busca entender percepciones y atributos relevantes mediante una entrevista guiada por turnos.")
        st.write("Tus respuestas se usan solo para pruebas de experiencia de entrevista.")
        return

    st.subheader("Probar encuesta")
    if not available:
        st.warning("No hay categorias publicadas para ejecutar encuesta.")
        return

    categoria = str(st.session_state.get("client_selected_category", selected_default)).strip().lower()
    if categoria not in available:
        categoria = available[0]
        st.session_state["client_selected_category"] = categoria

    if bool(deployment_cfg.get("allow_category_selection", True)):
        categoria = st.selectbox(
            "Categoria",
            options=available,
            index=available.index(categoria),
            key="client_probe_category",
        )
        st.session_state["client_selected_category"] = categoria

    try:
        trained_payload = load_trained_category(categoria)
    except Exception as exc:
        st.error(f"No se pudo cargar la categoria entrenada '{categoria}': {exc}")
        return

    if st.button("Iniciar encuesta", use_container_width=True, type="primary", key="client_btn_init"):
        st.session_state["client_state"] = start_session(trained_payload["brief"], trained_payload["plan"])
        st.session_state["client_session_saved_path"] = ""
        st.session_state["client_answer_pending_clear"] = True
        st.rerun()

    client_state = st.session_state.get("client_state")
    if not isinstance(client_state, dict) or not client_state:
        st.info("Presiona 'Iniciar encuesta' para comenzar.")
        return

    max_q = int(client_state.get("max_preguntas", 12) or 12)
    current_turn = int(client_state.get("turn_index", 0) or 0)
    ratio = min(max(current_turn / max(max_q, 1), 0.0), 1.0)
    st.progress(ratio, text=f"Progreso: turno {current_turn}/{max_q}")
    st.markdown("### Pregunta actual")
    st.info(str(client_state.get("next_question", "")))

    answer_key = "client_answer_input"
    clear_key = "client_answer_pending_clear"
    if answer_key not in st.session_state:
        st.session_state[answer_key] = ""
    if clear_key not in st.session_state:
        st.session_state[clear_key] = False
    if st.session_state.get(clear_key):
        st.session_state[answer_key] = ""
        st.session_state[clear_key] = False
    st.text_area("Tu respuesta", key=answer_key, height=120, placeholder="Escribe tu respuesta aqui...")

    if st.button("Enviar", use_container_width=True, key="client_btn_send"):
        answer = str(st.session_state.get(answer_key, "")).strip()
        if not answer:
            st.warning("Escribe una respuesta antes de enviar.")
        else:
            result = step_with_human_answer(client_state, answer)
            st.session_state["client_state"] = result.get("state", client_state)
            st.session_state[clear_key] = True
            st.rerun()

    latest_state = st.session_state.get("client_state", {})
    if bool(latest_state.get("done", False)):
        st.success("Gracias. La encuesta ha finalizado.")
        if bool(deployment_cfg.get("save_client_sessions", True)) and not st.session_state.get("client_session_saved_path"):
            try:
                out = _save_client_session(categoria, latest_state, trained_payload)
                st.session_state["client_session_saved_path"] = str(out)
            except Exception as exc:
                st.error(f"No se pudo guardar la sesion: {exc}")
        if st.session_state.get("client_session_saved_path"):
            st.caption(f"Sesion guardada en: {st.session_state['client_session_saved_path']}")
        if st.button("Nueva encuesta", use_container_width=True, key="client_btn_new"):
            st.session_state["client_state"] = None
            st.session_state["client_session_saved_path"] = ""
            st.session_state[clear_key] = True
            st.rerun()


def render_publicacion_cliente_admin() -> None:
    st.subheader("Publicacion cliente")
    guides = load_calibration_guides()
    guide_cats = guides.get("categories", {}) if isinstance(guides, dict) else {}
    if not isinstance(guide_cats, dict):
        guide_cats = {}

    published = load_published_categories()
    metadata = published.get("metadata", {}) if isinstance(published, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    rows: List[Dict[str, Any]] = []
    for categoria in _all_known_categories():
        latest = _latest_metrics_for_category(categoria)
        guide_metrics = guide_cats.get(categoria, guide_cats.get("producto_generico", {}))
        eval_pub = evaluate_category_publishability(categoria, latest, guide_metrics if isinstance(guide_metrics, dict) else {})
        md = metadata.get(categoria, {}) if isinstance(metadata.get(categoria, {}), dict) else {}
        published_cliente = is_category_available_for_client(categoria, published)
        rows.append(
            {
                "categoria": categoria,
                "accepted": round(float(eval_pub.get("metricas", {}).get("accepted", 0.0)), 3),
                "coverage_util": round(float(eval_pub.get("metricas", {}).get("coverage_util", 0.0)), 3),
                "human_like": round(float(eval_pub.get("metricas", {}).get("human_like", 0.0)), 3),
                "cumple_guias": bool(md.get("cumple_guias", eval_pub.get("cumple_guias", False))),
                "aprobada_manual": bool(md.get("aprobada_manual", False)),
                "publicada_cliente": bool(published_cliente),
                "recomendada_para_publicar": bool(eval_pub.get("recomendada_para_publicar", False)),
                "motivos": "; ".join([str(x) for x in eval_pub.get("motivos", [])]),
            }
        )
        metadata.setdefault(categoria, {})
        metadata[categoria].update(
            {
                "cumple_guias": bool(eval_pub.get("cumple_guias", False)),
                "accepted": float(eval_pub.get("metricas", {}).get("accepted", 0.0)),
                "coverage_util": float(eval_pub.get("metricas", {}).get("coverage_util", 0.0)),
                "human_like": float(eval_pub.get("metricas", {}).get("human_like", 0.0)),
            }
        )

    published["metadata"] = metadata
    save_published_categories(published)

    if rows:
        st.dataframe(_df(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No hay categorias detectadas aun.")

    categorias = [row["categoria"] for row in rows]
    if not categorias:
        return

    categoria_sel = st.selectbox("Categoria a administrar", options=categorias, key="admin_publicacion_categoria")
    md_sel = metadata.get(categoria_sel, {}) if isinstance(metadata.get(categoria_sel, {}), dict) else {}
    cumple_guias_sel = bool(md_sel.get("cumple_guias", False))
    if not cumple_guias_sel:
        st.warning("La categoria no cumple guias minimas. Puedes hacer override manual si decides publicarla.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Publicar categoria", use_container_width=True, key="btn_publicar_cat"):
            categorias_publicadas = [str(x).strip().lower() for x in published.get("categorias_publicadas", []) if str(x).strip()]
            if categoria_sel not in categorias_publicadas:
                categorias_publicadas.append(categoria_sel)
            published["categorias_publicadas"] = sorted(list(dict.fromkeys(categorias_publicadas)))
            metadata.setdefault(categoria_sel, {})
            metadata[categoria_sel]["estado"] = "publicada"
            published["metadata"] = metadata
            save_published_categories(published)
            st.success(f"Categoria '{categoria_sel}' publicada.")
            st.rerun()
    with c2:
        if st.button("Retirar categoria", use_container_width=True, key="btn_retirar_cat"):
            categorias_publicadas = [str(x).strip().lower() for x in published.get("categorias_publicadas", []) if str(x).strip()]
            categorias_publicadas = [c for c in categorias_publicadas if c != categoria_sel]
            published["categorias_publicadas"] = sorted(list(dict.fromkeys(categorias_publicadas)))
            metadata.setdefault(categoria_sel, {})
            metadata[categoria_sel]["estado"] = "retirada"
            published["metadata"] = metadata
            save_published_categories(published)
            st.success(f"Categoria '{categoria_sel}' retirada de cliente.")
            st.rerun()
    with c3:
        if st.button("Aprobar manualmente", use_container_width=True, key="btn_aprobar_manual"):
            metadata.setdefault(categoria_sel, {})
            metadata[categoria_sel]["aprobada_manual"] = True
            published["metadata"] = metadata
            save_published_categories(published)
            st.success(f"Categoria '{categoria_sel}' aprobada manualmente.")
            st.rerun()
    with c4:
        if st.button("Rechazar manualmente", use_container_width=True, key="btn_rechazar_manual"):
            metadata.setdefault(categoria_sel, {})
            metadata[categoria_sel]["aprobada_manual"] = False
            published["metadata"] = metadata
            save_published_categories(published)
            st.success(f"Categoria '{categoria_sel}' marcada como no aprobada.")
            st.rerun()

st.set_page_config(page_title="Plan Visual de Trabajo", page_icon=":bar_chart:", layout="wide")

deployment_cfg = load_deployment_config()
try:
    ensure_trained_categories_structure()
except Exception as exc:
    st.warning(f"No se pudo inicializar trained_categories: {exc}")

configured_mode = str(deployment_cfg.get("app_mode", "client")).strip().lower()
if configured_mode not in {"client", "admin"}:
    configured_mode = "client"

admin_password = ""
try:
    admin_password = str(st.secrets.get("ADMIN_PASSWORD", deployment_cfg.get("admin_password", ""))).strip()
except Exception:
    admin_password = str(deployment_cfg.get("admin_password", "")).strip()

allow_admin = bool(deployment_cfg.get("show_admin_tools", False)) or bool(admin_password)
if allow_admin:
    mode_ui = st.sidebar.radio(
        "Modo app",
        options=["cliente", "admin"],
        index=0 if configured_mode == "client" else 1,
        key="app_mode_selector",
    )
else:
    mode_ui = "cliente"

if mode_ui == "admin" and admin_password:
    pwd = st.sidebar.text_input("Password admin", type="password", key="admin_pwd_input")
    if pwd != admin_password:
        st.title("Acceso admin protegido")
        st.info("Ingresa password de administrador para habilitar herramientas internas.")
        st.stop()

if mode_ui == "cliente":
    render_client_mode(deployment_cfg)
    st.stop()

st.title("Generador de Plan Visual de Moderacion IA")
st.caption("Blueprint ejecutable por turnos para moderacion cualitativa")

try:
    api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        api_key = load_api_key_from_file("tokenkey.txt")
    os.environ["OPENAI_API_KEY"] = api_key
except Exception as exc:
    st.error(f"Error al cargar API key: {exc}")
    st.stop()

DEFAULTS = {
    "antecedente": "",
    "objetivo_principal": "",
    "tipo_sesion": "Exploracion por atributos",
    "target_atributos": 10,
    "max_preguntas": 12,
    "numero_entrevistas_planeadas": 12,
    "atributos_raw": "",
    "definiciones_raw": "",
    "definiciones_operativas_fuente": "no_provistas",
    "profundidad": "Alta",
    "permitir_atributos_emergentes": False,
    "max_atributos_emergentes": 3,
    "evitar_sugestivas": True,
    "no_preguntas_temporales": True,
    "rango_edad": "16-18",
    "rangos_edad_objetivo": ["16-18"],
    "genero": "mixto",
    "perfil_participante": "estudiante",
    "perfiles_objetivo": ["estudiante"],
    "nivel_conocimiento": "bajo",
    "nivel_profundizacion": "baja",
    "probabilidad_confusion": 0.15,
    "primera_pregunta_modo": "ia",
    "primera_pregunta_manual": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

for key, value in {
    "brief_result": None,
    "plan_result": None,
    "definiciones_operativas_ai": {},
    "definiciones_raw_pending_update": None,
    "autoplay_state": None,
    "autoplay_result": None,
    "autoplay_lab_result": None,
    "live_state": None,
    "live_result": None,
    "live_answer_input": "",
    "live_answer_pending_clear": False,
    "profile_pending_update": None,
    "cat_attrs_preview": "",
    "cat_semantic_seed_preview": None,
    "cat_suite_preview": None,
    "cat_result": None,
    "calib_result": {},
    "calib_history_rows": [],
    "calib_execution_log": [],
    "calib_last_summary": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

pending_profile = st.session_state.get("profile_pending_update")
if isinstance(pending_profile, dict) and pending_profile:
    for k in [
        "rango_edad",
        "rangos_edad_objetivo",
        "genero",
        "perfil_participante",
        "perfiles_objetivo",
        "nivel_conocimiento",
        "nivel_profundizacion",
        "probabilidad_confusion",
    ]:
        if k in pending_profile:
            st.session_state[k] = pending_profile[k]
    st.session_state["profile_pending_update"] = None

pending_raw = st.session_state.get("definiciones_raw_pending_update")
if isinstance(pending_raw, str):
    st.session_state["definiciones_raw"] = pending_raw
    st.session_state["definiciones_raw_pending_update"] = None

if st.session_state.get("live_answer_pending_clear"):
    st.session_state["live_answer_input"] = ""
    st.session_state["live_answer_pending_clear"] = False

modulo_ui = st.sidebar.radio(
    "Modulo",
    options=["Entrevista", "Autoplay Lab", "Stress Validation", "Crear Categoria", "Calibration Lab", "Publicacion cliente"],
    index=0,
    key="modulo_ui_selector",
)

if modulo_ui == "Stress Validation":
    render_stress_validation()
    st.stop()

if modulo_ui == "Calibration Lab":
    st.sidebar.markdown("### Calibration Lab")
    guides_payload = load_calibration_guides()
    guide_cats = []
    if isinstance(guides_payload, dict):
        categories_obj = guides_payload.get("categories", {})
        if isinstance(categories_obj, dict):
            guide_cats = [str(k) for k in categories_obj.keys()]
    if not guide_cats:
        guide_cats = ["universidad", "soda", "auto", "banco", "seguro", "retail", "tecnologia"]

    calib_categoria = st.sidebar.selectbox("categoria", options=guide_cats, index=0, key="calib_categoria")
    calib_iter = st.sidebar.number_input("numero maximo de iteraciones", min_value=1, max_value=100, value=5, step=1, key="calib_max_iterations")
    calib_seeds = st.sidebar.text_input("seeds", value="123,456", key="calib_seeds")
    calib_n_runs = st.sidebar.number_input("n_runs", min_value=1, max_value=100, value=20, step=1, key="calib_n_runs")
    calib_run = st.sidebar.button("Ejecutar calibracion", use_container_width=True, key="calib_btn_run")
    calib_load = st.sidebar.button("Cargar historico", use_container_width=True, key="calib_btn_load")
    if st.sidebar.button("Limpiar vista", use_container_width=True, key="calib_btn_clear"):
        st.session_state["calib_result"] = {}
        st.session_state["calib_history_rows"] = []
        st.session_state["calib_execution_log"] = []
        st.session_state["calib_last_summary"] = {}
        st.rerun()

    render_calibration_lab(
        categoria=str(calib_categoria),
        max_iterations=int(calib_iter),
        seeds_text=str(calib_seeds),
        n_runs=int(calib_n_runs),
        run_clicked=bool(calib_run),
        load_clicked=bool(calib_load),
    )
    st.stop()

if modulo_ui == "Publicacion cliente":
    render_publicacion_cliente_admin()
    st.stop()

if modulo_ui == "Crear Categoria":
    st.subheader("CREAR CATEGORIA")
    st.caption("Crea nuevas categorias de entrenamiento para el moderador.")

    c1, c2, c3 = st.columns(3)
    with c1:
        nombre_categoria = st.text_input("nombre_categoria", key="cat_nombre_categoria", placeholder="ej: telecom")
    with c2:
        target_atributos_cat = st.number_input("target_atributos", min_value=3, max_value=20, value=8, step=1, key="cat_target_atributos")
    with c3:
        n_suite_ejemplos = st.slider("ejemplos_suite_lab", min_value=10, max_value=20, value=12, step=1, key="cat_n_suite")

    objetivo_categoria = st.text_area(
        "objetivo",
        key="cat_objetivo",
        height=90,
        placeholder="Describe el objetivo de la categoria para el laboratorio.",
    )
    atributos_usuario_raw = st.text_area(
        "atributos opcionales (uno por linea)",
        key="cat_atributos_usuario_raw",
        height=110,
        placeholder="Si dejas vacio, se sugieren con IA.",
    )

    st.markdown("### Paso 2 - Atributos iniciales")
    if st.button("Generar atributos iniciales", use_container_width=True, key="cat_btn_generar_attrs"):
        attrs_user = [limpiar_texto_input(x) for x in atributos_usuario_raw.splitlines() if limpiar_texto_input(x)]
        try:
            attrs_gen = generar_atributos_iniciales(
                categoria=limpiar_texto_input(nombre_categoria),
                objetivo=limpiar_texto_input(objetivo_categoria),
                target_atributos=int(target_atributos_cat),
                atributos_usuario=attrs_user,
            )
            st.session_state["cat_attrs_preview"] = "\n".join(attrs_gen)
            st.success(f"Atributos generados: {len(attrs_gen)}")
        except Exception as exc:
            st.error(f"No fue posible generar atributos iniciales: {exc}")

    attrs_edit_raw = st.text_area(
        "lista editable de atributos",
        key="cat_attrs_preview",
        height=150,
        placeholder="Aqui veras/editaras los atributos iniciales.",
    )
    attrs_edit = [limpiar_texto_input(x) for x in attrs_edit_raw.splitlines() if limpiar_texto_input(x)]

    st.markdown("### Paso 3 - Equivalencias semanticas")
    if st.button("Generar equivalencias semanticas", use_container_width=True, key="cat_btn_semantic"):
        if not attrs_edit:
            st.warning("Primero agrega o genera atributos.")
        else:
            st.session_state["cat_semantic_seed_preview"] = _build_semantic_seed_preview(
                categoria=limpiar_texto_input(nombre_categoria),
                atributos=attrs_edit,
            )
    if isinstance(st.session_state.get("cat_semantic_seed_preview"), dict):
        st.json(st.session_state["cat_semantic_seed_preview"], expanded=False)

    st.markdown("### Paso 4 - Suite de laboratorio")
    if st.button("Generar suite de laboratorio", use_container_width=True, key="cat_btn_suite"):
        if not attrs_edit:
            st.warning("Primero agrega o genera atributos.")
        else:
            try:
                suite_preview = generar_suite_laboratorio(
                    categoria=limpiar_texto_input(nombre_categoria),
                    objetivo=limpiar_texto_input(objetivo_categoria),
                    atributos=attrs_edit,
                    n_ejemplos=int(n_suite_ejemplos),
                )
                st.session_state["cat_suite_preview"] = suite_preview
                st.success(f"Suite creada con {suite_preview.get('n_casos', 0)} casos.")
            except Exception as exc:
                st.error(f"No fue posible generar suite: {exc}")

    if isinstance(st.session_state.get("cat_suite_preview"), dict):
        suite_cases = st.session_state["cat_suite_preview"].get("cases", [])
        st.write(f"Casos de suite: {len(suite_cases)}")
        st.dataframe(suite_cases[:10], use_container_width=True)

    st.markdown("### Paso 5 - Crear categoria")
    if st.button("Crear categoria", use_container_width=True, type="primary", key="cat_btn_crear"):
        if not limpiar_texto_input(nombre_categoria):
            st.error("Define nombre_categoria.")
        elif not limpiar_texto_input(objetivo_categoria):
            st.error("Define objetivo.")
        elif not attrs_edit:
            st.error("Genera o captura atributos primero.")
        else:
            try:
                cat_bar = st.progress(0.0, text="Preparando creacion de categoria...")
                cat_status = st.empty()
                cat_status.caption("Generando estructura base...")
                cat_bar.progress(0.25, text="25% - Generando estructura")
                result = crear_categoria_proyecto(
                    categoria=limpiar_texto_input(nombre_categoria),
                    objetivo=limpiar_texto_input(objetivo_categoria),
                    target_atributos=int(target_atributos_cat),
                    atributos_usuario=attrs_edit,
                    n_ejemplos_suite=int(n_suite_ejemplos),
                )
                cat_status.caption("Guardando artefactos y registro...")
                cat_bar.progress(0.9, text="90% - Guardando archivos")
                cat_bar.progress(1.0, text="100% - Categoria creada")
                st.session_state["cat_result"] = result
                st.success("Categoria creada correctamente.")
            except Exception as exc:
                st.error(f"No fue posible crear categoria: {exc}")

    if isinstance(st.session_state.get("cat_result"), dict):
        result = st.session_state["cat_result"]
        st.markdown("### Resumen")
        st.write(f"Categoria creada: **{result.get('categoria','-')}**")
        st.write(f"Atributos generados: **{len(result.get('atributos_generados', []))}**")
        st.write(f"Suite creada: **{result.get('suite_generada', 0)}**")
        st.write("Lista para laboratorio.")
        with st.expander("Archivos creados"):
            st.json(result.get("archivos", {}), expanded=False)
    st.stop()

if modulo_ui == "Autoplay Lab":
    st.info("Vista enfocada en Autoplay Lab: usa la tab 'Laboratorio autoplay'.")


tab_plan, tab_auto, tab_lab, tab_live = st.tabs(
    ["Plan visual", "Ejecutar entrevista (autoplay)", "Laboratorio autoplay", "Entrevista en vivo (humano)"]
)

with tab_plan:
    st.markdown("### Precargar proyecto")
    uploaded_project = st.file_uploader("Cargar JSON exportado", type=["json"], key="project_uploader")
    if st.button("Precargar datos", use_container_width=True):
        if not uploaded_project:
            st.warning("Selecciona un archivo JSON primero.")
        else:
            try:
                payload = json.loads(uploaded_project.read().decode("utf-8"))
                if isinstance(payload, dict) and "plan" in payload:
                    brief_payload = payload.get("brief", {})
                    st.session_state["plan_result"] = payload.get("plan")
                elif isinstance(payload, dict) and "plan_v3_2" in payload:
                    brief_payload = payload.get("brief", {})
                    st.session_state["plan_result"] = payload.get("plan_v3_2")
                else:
                    brief_payload = payload
                    st.session_state["plan_result"] = None

                if not isinstance(brief_payload, dict):
                    raise ValueError("El JSON no contiene un objeto de brief valido.")

                base_brief = brief_payload.get("brief", {})
                atributos_objetivo = brief_payload.get("atributos_objetivo", {})
                definiciones = brief_payload.get("definiciones_operativas", {})
                config = brief_payload.get("config", {})
                guardrails = brief_payload.get("guardrails", {})
                target_participante = brief_payload.get("target_participante", {})

                st.session_state["antecedente"] = limpiar_texto_input(base_brief.get("antecedente", ""))
                st.session_state["objetivo_principal"] = limpiar_texto_input(base_brief.get("objetivo_principal", ""))
                st.session_state["tipo_sesion"] = limpiar_texto_input(base_brief.get("tipo_sesion", "Exploracion por atributos"))
                st.session_state["target_atributos"] = int(atributos_objetivo.get("target", 10))
                st.session_state["max_preguntas"] = int(config.get("max_preguntas", 12))
                st.session_state["numero_entrevistas_planeadas"] = int(config.get("numero_entrevistas_planeadas", 12))
                st.session_state["atributos_raw"] = "\n".join(atributos_objetivo.get("lista", []))
                st.session_state["definiciones_raw"] = definiciones_to_raw_lines(definiciones if isinstance(definiciones, dict) else {})
                st.session_state["profundidad"] = config.get("profundidad", "Alta")
                st.session_state["permitir_atributos_emergentes"] = bool(config.get("permitir_atributos_emergentes", False))
                st.session_state["max_atributos_emergentes"] = int(config.get("max_atributos_emergentes", 3))
                st.session_state["definiciones_operativas_fuente"] = str(
                    config.get("definiciones_operativas_fuente", "manual" if definiciones else "no_provistas")
                )
                st.session_state["evitar_sugestivas"] = bool(guardrails.get("evitar_sugestivas", True))
                st.session_state["no_preguntas_temporales"] = bool(guardrails.get("no_preguntas_temporales", True))
                st.session_state["rango_edad"] = str(target_participante.get("rango_edad", "16-18"))
                rangos_pre = target_participante.get("rangos_edad", [st.session_state["rango_edad"]])
                if isinstance(rangos_pre, list) and rangos_pre:
                    st.session_state["rangos_edad_objetivo"] = [str(x) for x in rangos_pre]
                else:
                    st.session_state["rangos_edad_objetivo"] = [st.session_state["rango_edad"]]
                st.session_state["genero"] = str(target_participante.get("genero", "mixto"))
                st.session_state["perfil_participante"] = str(target_participante.get("perfil", "estudiante"))
                perfiles_pre = target_participante.get("perfiles", [st.session_state["perfil_participante"]])
                if isinstance(perfiles_pre, list) and perfiles_pre:
                    st.session_state["perfiles_objetivo"] = [str(x) for x in perfiles_pre]
                else:
                    st.session_state["perfiles_objetivo"] = [st.session_state["perfil_participante"]]
                st.session_state["nivel_conocimiento"] = str(target_participante.get("nivel_conocimiento", "bajo"))
                st.session_state["nivel_profundizacion"] = str(target_participante.get("nivel_profundizacion", "baja"))
                st.session_state["probabilidad_confusion"] = float(config.get("probabilidad_confusion", 0.15))
                st.session_state["primera_pregunta_modo"] = str(config.get("primera_pregunta_modo", "ia"))
                st.session_state["primera_pregunta_manual"] = str(config.get("primera_pregunta_manual", ""))
                st.session_state["definiciones_operativas_ai"] = definiciones if isinstance(definiciones, dict) else {}
                st.session_state["brief_result"] = brief_payload

                if st.session_state["plan_result"] is None:
                    st.session_state["plan_result"] = generar_plan_visual(brief_payload)

                st.success("Proyecto precargado.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo precargar el archivo: {exc}")

    with st.form("form_brief"):
        st.markdown("### Brief del cliente")

        antecedente = st.text_area("Antecedente", placeholder="Contexto del estudio", height=90, key="antecedente")
        objetivo_principal = st.text_area(
            "Objetivo principal",
            placeholder="Que decision de negocio debe habilitar este estudio",
            height=90,
            key="objetivo_principal",
        )

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            tipo_sesion = st.selectbox(
                "Tipo de sesion",
                options=["Exploracion por atributos", "Evaluacion de concepto", "Uso y experiencia", "Percepcion de marca"],
                key="tipo_sesion",
            )
        with col_b:
            target_atributos = st.number_input("Target de atributos", min_value=1, max_value=50, step=1, key="target_atributos")
        with col_c:
            max_preguntas = st.number_input("Maximo de preguntas", min_value=1, max_value=60, step=1, key="max_preguntas")
        with col_d:
            numero_entrevistas_planeadas = st.number_input(
                "Numero de entrevistas", min_value=1, max_value=200, step=1, key="numero_entrevistas_planeadas"
            )

        atributos_raw = st.text_area(
            "Atributos objetivo (uno por linea)",
            placeholder="calidad academica\nempleabilidad\nprecio percibido",
            height=100,
            key="atributos_raw",
        )
        lista_atributos_form = [limpiar_texto_input(x) for x in atributos_raw.splitlines() if limpiar_texto_input(x)]
        if lista_atributos_form and int(target_atributos) != len(lista_atributos_form):
            st.warning(
                f"target_atributos ({int(target_atributos)}) no coincide con len(lista_atributos) ({len(lista_atributos_form)})."
            )

        definiciones_raw = st.text_area(
            "Definiciones operativas (formato: atributo | que_cuenta | que_no_cuenta)",
            placeholder="calidad academica | nivel docente y exigencia | reputacion sin evidencia",
            height=120,
            key="definiciones_raw",
        )

        col_e, col_f, col_g, col_h, col_i = st.columns(5)
        with col_e:
            profundidad = st.selectbox("Profundidad", options=["Alta", "Media", "Baja"], key="profundidad")
        with col_f:
            permitir_atributos_emergentes = st.checkbox("Permitir atributos emergentes", key="permitir_atributos_emergentes")
        with col_g:
            max_atributos_emergentes = st.number_input(
                "Max emergentes", min_value=1, max_value=10, step=1, key="max_atributos_emergentes"
            )
        with col_h:
            evitar_sugestivas = st.checkbox("Evitar preguntas sugestivas", key="evitar_sugestivas")
        with col_i:
            no_preguntas_temporales = st.checkbox("No preguntas temporales", key="no_preguntas_temporales")

        st.markdown("### Participante simulado (autoplay)")
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            rangos_edad_objetivo = st.multiselect(
                "Rangos edad objetivo",
                options=["13-15", "16-18", "19-24", "25-34", "35-49", "50+"],
                key="rangos_edad_objetivo",
                help="Selecciona uno o varios rangos para dirigir el plan visual al publico objetivo.",
            )
            rango_edad = st.selectbox(
                "Rango edad para autoplay",
                options=["13-15", "16-18", "19-24", "25-34", "35-49", "50+"],
                key="rango_edad",
            )
            genero = st.selectbox(
                "Genero",
                options=["mujer", "hombre", "mixto", "no_especificado"],
                key="genero",
            )
        with cp2:
            perfiles_objetivo = st.multiselect(
                "Perfiles objetivo",
                options=["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"],
                key="perfiles_objetivo",
                help="Selecciona uno o varios perfiles para orientar el plan visual.",
            )
            perfil_participante = st.selectbox(
                "Perfil para autoplay",
                options=["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"],
                key="perfil_participante",
            )
            nivel_conocimiento = st.selectbox(
                "Nivel conocimiento",
                options=["bajo", "medio", "alto"],
                key="nivel_conocimiento",
            )
        with cp3:
            nivel_profundizacion = st.selectbox(
                "Nivel profundizacion",
                options=["baja", "media", "alta"],
                key="nivel_profundizacion",
            )
            probabilidad_confusion = st.slider(
                "Probabilidad confusion",
                min_value=0.0,
                max_value=0.5,
                value=float(st.session_state.get("probabilidad_confusion", 0.15)),
                step=0.01,
                key="probabilidad_confusion",
            )

        st.markdown("### Primera pregunta de inicio")
        pq1, pq2 = st.columns([1, 2])
        with pq1:
            primera_pregunta_modo = st.selectbox(
                "Origen",
                options=["ia", "manual"],
                key="primera_pregunta_modo",
                help="ia: la IA propone la primera pregunta con base en el brief; manual: la defines tu.",
            )
        with pq2:
            primera_pregunta_manual = st.text_input(
                "Primera pregunta manual (opcional)",
                key="primera_pregunta_manual",
                placeholder="Ejemplo: Cuando piensas en elegir, que es lo mas importante para ti?",
                help="Se usa solo cuando Origen = manual.",
            )

        # Preview visible antes de generar plan
        try:
            preview_attrs = [x.strip() for x in st.session_state.get("atributos_raw", "").splitlines() if x.strip()]
            preview_brief = {
                "brief": {
                    "antecedente": st.session_state.get("antecedente", ""),
                    "objetivo_principal": st.session_state.get("objetivo_principal", ""),
                    "tipo_sesion": st.session_state.get("tipo_sesion", "Exploracion por atributos"),
                },
                "atributos_objetivo": {
                    "target": int(st.session_state.get("target_atributos", 1) or 1),
                    "lista": preview_attrs,
                },
                "config": {
                    "primera_pregunta_modo": str(st.session_state.get("primera_pregunta_modo", "ia")),
                    "primera_pregunta_manual": str(st.session_state.get("primera_pregunta_manual", "")),
                },
            }
            primera_preview = generar_primera_pregunta_inicio(
                brief_cliente=preview_brief,
                modo=str(st.session_state.get("primera_pregunta_modo", "ia")),
                manual=str(st.session_state.get("primera_pregunta_manual", "")),
            )
            st.caption("Preview primera pregunta")
            st.info(primera_preview)
        except Exception:
            st.caption("Preview primera pregunta")
            st.info("Cuando llenes objetivo/atributos, se mostrara aqui la primera pregunta.")

        col_btn_a, col_btn_b = st.columns(2)
        with col_btn_a:
            generar_defs_ai = st.form_submit_button("Generar definiciones con IA", use_container_width=True)
        with col_btn_b:
            submitted = st.form_submit_button("Generar plan visual", use_container_width=True)

    if generar_defs_ai:
        lista_atributos = [limpiar_texto_input(x) for x in st.session_state["atributos_raw"].splitlines() if limpiar_texto_input(x)]
        if not lista_atributos:
            st.warning("Agrega al menos un atributo para generar definiciones con IA.")
            st.stop()
        try:
            definiciones_ai = generate_definiciones_operativas_ai(
                antecedente=limpiar_texto_input(st.session_state["antecedente"]),
                objetivo_principal=limpiar_texto_input(st.session_state["objetivo_principal"]),
                tipo_sesion=limpiar_texto_input(st.session_state["tipo_sesion"]),
                lista_atributos=lista_atributos,
            )
            st.session_state["definiciones_operativas_ai"] = definiciones_ai
            st.session_state["definiciones_raw_pending_update"] = definiciones_to_raw_lines(definiciones_ai)
            st.session_state["definiciones_operativas_fuente"] = "generadas_por_ia"
            st.rerun()
        except Exception as exc:
            st.error(f"No fue posible generar definiciones con IA: {exc}")

    if submitted:
        lista_atributos = [limpiar_texto_input(x) for x in atributos_raw.splitlines() if limpiar_texto_input(x)]

        definiciones_fuente = st.session_state.get("definiciones_operativas_fuente", "no_provistas")
        if definiciones_raw.strip():
            try:
                definiciones = parse_definiciones_operativas(definiciones_raw)
                if definiciones_fuente == "no_provistas":
                    definiciones_fuente = "manual"
            except Exception as exc:
                st.error(f"Error en definiciones operativas: {exc}")
                st.stop()
        else:
            defs_ai = st.session_state.get("definiciones_operativas_ai", {})
            if isinstance(defs_ai, dict) and defs_ai:
                definiciones = defs_ai
                definiciones_fuente = "generadas_por_ia"
            else:
                definiciones = {}
                definiciones_fuente = "no_provistas"

        brief = build_brief(
            antecedente=limpiar_texto_input(antecedente),
            objetivo_principal=limpiar_texto_input(objetivo_principal),
            tipo_sesion=tipo_sesion,
            target_atributos=int(target_atributos),
            lista_atributos=lista_atributos,
            definiciones_operativas=definiciones,
            definiciones_operativas_fuente=definiciones_fuente,
            max_preguntas=int(max_preguntas),
            numero_entrevistas_planeadas=int(numero_entrevistas_planeadas),
            profundidad=profundidad,
            permitir_atributos_emergentes=permitir_atributos_emergentes,
            max_atributos_emergentes=int(max_atributos_emergentes),
            evitar_sugestivas=evitar_sugestivas,
            no_preguntas_temporales=no_preguntas_temporales,
            rango_edad=rango_edad,
            rangos_edad_objetivo=[str(x) for x in (rangos_edad_objetivo or [rango_edad])],
            genero=genero,
            perfil_participante=perfil_participante,
            perfiles_objetivo=[str(x) for x in (perfiles_objetivo or [perfil_participante])],
            nivel_conocimiento=nivel_conocimiento,
            nivel_profundizacion=nivel_profundizacion,
            probabilidad_confusion=float(probabilidad_confusion),
            primera_pregunta_modo=str(primera_pregunta_modo),
            primera_pregunta_manual=limpiar_texto_input(primera_pregunta_manual),
        )
        st.session_state["brief_result"] = brief

        try:
            with st.spinner("Generando plan visual de moderacion IA..."):
                st.session_state["plan_result"] = generar_plan_visual(brief)
                st.session_state["autoplay_state"] = None
                st.session_state["autoplay_result"] = None
                st.session_state["live_state"] = None
                st.session_state["live_result"] = None
        except Exception as exc:
            st.error(f"No fue posible generar el plan: {exc}")

    defs_preview: Dict[str, Dict[str, str]] = {}
    try:
        if st.session_state.get("definiciones_raw", "").strip():
            defs_preview = parse_definiciones_operativas(st.session_state.get("definiciones_raw", ""))
    except Exception:
        defs_preview = st.session_state.get("definiciones_operativas_ai", {}) or {}

    if defs_preview:
        st.markdown("### Preview de definiciones operativas")
        for atributo, info in defs_preview.items():
            with st.container(border=True):
                st.write(f"**Atributo:** {atributo}")
                st.write(f"**Que cuenta:** {info.get('que_cuenta', '')}")
                st.write(f"**Que no cuenta:** {info.get('que_no_cuenta', '')}")

    if st.session_state.get("brief_result"):
        st.markdown("### Brief enviado")
        st.json(st.session_state["brief_result"], expanded=False)

    if st.session_state.get("plan_result"):
        plan = st.session_state["plan_result"]
        render_plan(plan)

        with st.expander("Ver JSON del plan"):
            st.json(plan, expanded=False)

        project_payload = {
            "version": "1.2",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "brief": st.session_state["brief_result"],
            "plan": st.session_state["plan_result"],
        }
        project_slug = _project_slug(st.session_state["brief_result"])
        st.download_button(
            "Descargar JSON del proyecto",
            data=json.dumps(project_payload, ensure_ascii=False, indent=2),
            file_name=f"{project_slug}_plan_visual.json",
            mime="application/json",
            use_container_width=True,
        )

with tab_auto:
    st.markdown("### Ejecucion automatica (autoplay)")
    if not st.session_state.get("brief_result") or not st.session_state.get("plan_result"):
        st.info("Primero genera un plan visual para habilitar autoplay.")
    else:
        if not isinstance(st.session_state["brief_result"].get("target_participante", {}), dict):
            st.session_state["brief_result"]["target_participante"] = {}
        if not isinstance(st.session_state["brief_result"].get("config", {}), dict):
            st.session_state["brief_result"]["config"] = {}

        target_p = st.session_state["brief_result"].get("target_participante", {})
        cfg = st.session_state["brief_result"].get("config", {})
        if not target_p:
            target_p = {
                "rango_edad": st.session_state.get("rango_edad", "16-18"),
                "genero": st.session_state.get("genero", "mixto"),
                "perfil": st.session_state.get("perfil_participante", "estudiante"),
                "perfiles": st.session_state.get("perfiles_objetivo", ["estudiante"]),
                "nivel_conocimiento": st.session_state.get("nivel_conocimiento", "bajo"),
                "nivel_profundizacion": st.session_state.get("nivel_profundizacion", "baja"),
            }
            st.session_state["brief_result"]["target_participante"] = target_p
        if "probabilidad_confusion" not in cfg:
            cfg["probabilidad_confusion"] = float(st.session_state.get("probabilidad_confusion", 0.15))
            st.session_state["brief_result"]["config"] = cfg

        with st.expander("Configurar perfil participante para autoplay", expanded=True):
            ap1, ap2, ap3 = st.columns(3)
            with ap1:
                auto_rangos_obj = st.multiselect(
                    "Rangos edad objetivo (plan)",
                    options=["13-15", "16-18", "19-24", "25-34", "35-49", "50+"],
                    default=(
                        [str(x) for x in target_p.get("rangos_edad", [])]
                        if isinstance(target_p.get("rangos_edad", []), list) and target_p.get("rangos_edad", [])
                        else [str(target_p.get("rango_edad", "16-18"))]
                    ),
                    key="auto_rangos_edad_objetivo",
                )
                auto_rango_edad = st.selectbox(
                    "Rango edad (autoplay)",
                    options=["13-15", "16-18", "19-24", "25-34", "35-49", "50+"],
                    index=["13-15", "16-18", "19-24", "25-34", "35-49", "50+"].index(
                        str(target_p.get("rango_edad", "16-18"))
                        if str(target_p.get("rango_edad", "16-18")) in ["13-15", "16-18", "19-24", "25-34", "35-49", "50+"]
                        else "16-18"
                    ),
                    key="auto_rango_edad",
                )
                auto_genero = st.selectbox(
                    "Genero (autoplay)",
                    options=["mujer", "hombre", "mixto", "no_especificado"],
                    index=["mujer", "hombre", "mixto", "no_especificado"].index(
                        str(target_p.get("genero", "mixto"))
                        if str(target_p.get("genero", "mixto")) in ["mujer", "hombre", "mixto", "no_especificado"]
                        else "mixto"
                    ),
                    key="auto_genero",
                )
            with ap2:
                auto_perfil = st.selectbox(
                    "Perfil (autoplay)",
                    options=["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"],
                    index=["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"].index(
                        str(target_p.get("perfil", "estudiante"))
                        if str(target_p.get("perfil", "estudiante")) in ["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"]
                        else "estudiante"
                    ),
                    key="auto_perfil_participante",
                )
                auto_perfiles_obj = st.multiselect(
                    "Perfiles objetivo (plan)",
                    options=["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"],
                    default=(
                        [str(x) for x in target_p.get("perfiles", [])]
                        if isinstance(target_p.get("perfiles", []), list) and target_p.get("perfiles", [])
                        else [str(target_p.get("perfil", "estudiante"))]
                    ),
                    key="auto_perfiles_objetivo",
                )
                auto_nivel_conocimiento = st.selectbox(
                    "Nivel conocimiento (autoplay)",
                    options=["bajo", "medio", "alto"],
                    index=["bajo", "medio", "alto"].index(
                        str(target_p.get("nivel_conocimiento", "bajo"))
                        if str(target_p.get("nivel_conocimiento", "bajo")) in ["bajo", "medio", "alto"]
                        else "bajo"
                    ),
                    key="auto_nivel_conocimiento",
                )
            with ap3:
                auto_nivel_profundizacion = st.selectbox(
                    "Nivel profundizacion (autoplay)",
                    options=["baja", "media", "alta"],
                    index=["baja", "media", "alta"].index(
                        str(target_p.get("nivel_profundizacion", "baja"))
                        if str(target_p.get("nivel_profundizacion", "baja")) in ["baja", "media", "alta"]
                        else "baja"
                    ),
                    key="auto_nivel_profundizacion",
                )
                auto_prob_confusion = st.slider(
                    "Probabilidad confusion (autoplay)",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(cfg.get("probabilidad_confusion", 0.15)),
                    step=0.01,
                    key="auto_probabilidad_confusion",
                )

            if st.button("Aplicar perfil al autoplay", use_container_width=True, key="btn_apply_auto_profile"):
                st.session_state["profile_pending_update"] = {
                    "rango_edad": auto_rango_edad,
                    "rangos_edad_objetivo": [str(x) for x in (auto_rangos_obj or [auto_rango_edad])],
                    "genero": auto_genero,
                    "perfil_participante": auto_perfil,
                    "perfiles_objetivo": [str(x) for x in (auto_perfiles_obj or [auto_perfil])],
                    "nivel_conocimiento": auto_nivel_conocimiento,
                    "nivel_profundizacion": auto_nivel_profundizacion,
                    "probabilidad_confusion": float(auto_prob_confusion),
                }

                st.session_state["brief_result"]["target_participante"] = {
                    "rango_edad": auto_rango_edad,
                    "rangos_edad": [str(x) for x in (auto_rangos_obj or [auto_rango_edad])],
                    "genero": auto_genero,
                    "perfil": auto_perfil,
                    "perfiles": [str(x) for x in (auto_perfiles_obj or [auto_perfil])],
                    "nivel_conocimiento": auto_nivel_conocimiento,
                    "nivel_profundizacion": auto_nivel_profundizacion,
                }
                st.session_state["brief_result"]["config"]["probabilidad_confusion"] = float(auto_prob_confusion)

                if st.session_state.get("autoplay_state"):
                    st.session_state["autoplay_state"]["brief"] = st.session_state["brief_result"]
                st.success("Perfil actualizado para autoplay.")
                st.rerun()

        target_p = st.session_state["brief_result"].get("target_participante", {})
        cfg = st.session_state["brief_result"].get("config", {})
        rangos_txt = ", ".join(target_p.get("rangos_edad", [])) if isinstance(target_p.get("rangos_edad", []), list) and target_p.get("rangos_edad", []) else target_p.get("rango_edad", "-")
        perfiles_txt = ", ".join(target_p.get("perfiles", [])) if isinstance(target_p.get("perfiles", []), list) and target_p.get("perfiles", []) else target_p.get("perfil", "-")
        with st.container(border=True):
            st.markdown("**Perfil participante simulado**")
            st.write(
                f"Edad objetivo: {rangos_txt} | Edad autoplay: {target_p.get('rango_edad','-')} | Genero: {target_p.get('genero','-')} | "
                f"Perfiles objetivo: {perfiles_txt} | Perfil autoplay: {target_p.get('perfil','-')}"
            )
            st.write(
                f"Conocimiento: {target_p.get('nivel_conocimiento','-')} | "
                f"Profundizacion: {target_p.get('nivel_profundizacion','-')} | "
                f"Prob. confusion: {cfg.get('probabilidad_confusion', 0.15)}"
            )
        estilo_auto = st.session_state["plan_result"].get("client_plan", {}).get("estilo_moderacion_aplicado", {})
        if isinstance(estilo_auto, dict) and estilo_auto:
            st.caption(
                "Estilo moderacion: "
                f"{estilo_auto.get('lenguaje','-')} | max palabras={estilo_auto.get('max_palabras_pregunta','-')} | "
                f"tono={estilo_auto.get('tono','-')} | evita abstracciones={estilo_auto.get('evitar_abstracciones', False)}"
            )
        with st.expander("Ver client_plan / engine_blueprint"):
            p = st.session_state["plan_result"]
            st.json(
                {
                    "client_plan": p.get("client_plan", {}),
                    "engine_blueprint": p.get("engine_blueprint", {}),
                },
                expanded=False,
            )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Iniciar autoplay", use_container_width=True, key="btn_auto_start"):
                st.session_state["autoplay_state"] = start_session(
                    st.session_state["brief_result"], st.session_state["plan_result"]
                )
                st.session_state["autoplay_result"] = None
                st.rerun()
        with c2:
            if st.button("Siguiente turno", use_container_width=True, key="btn_auto_step"):
                if not st.session_state.get("autoplay_state"):
                    st.warning("Inicia autoplay primero.")
                else:
                    result = step_turn(st.session_state["autoplay_state"])
                    st.session_state["autoplay_state"] = result["state"]
                    if result.get("done"):
                        st.success("La entrevista ha terminado.")
                    st.rerun()
        with c3:
            if st.button("Correr completo", use_container_width=True, key="btn_auto_full"):
                if not st.session_state.get("autoplay_state"):
                    st.session_state["autoplay_state"] = start_session(
                        st.session_state["brief_result"], st.session_state["plan_result"]
                    )
                state = st.session_state["autoplay_state"]
                max_q = int((state.get("brief", {}).get("config", {}) or {}).get("max_preguntas", 12) or 12)
                pbar = st.progress(0.0, text="Iniciando ejecucion completa...")
                status = st.empty()
                done = bool(state.get("done", False))
                while not done:
                    step_out = step_turn(state)
                    state = step_out.get("state", state)
                    turn_idx = int(state.get("turn_index", 0) or 0)
                    ratio = min(max(turn_idx / max(max_q, 1), 0.0), 1.0)
                    traces = state.get("traces", [])
                    last_action = "-"
                    if isinstance(traces, list) and traces and isinstance(traces[-1], dict):
                        last_action = str(traces[-1].get("accion", "-"))
                    pbar.progress(ratio, text=f"{int(ratio*100)}% - Turno {turn_idx}/{max_q}")
                    status.caption(f"Procesando turno {turn_idx} | accion: {last_action}")
                    done = bool(step_out.get("done", False) or state.get("done", False))
                    if turn_idx >= max_q:
                        break
                st.session_state["autoplay_result"] = {"state": state}
                st.session_state["autoplay_state"] = state
                pbar.progress(1.0, text="100% - Ejecucion completa finalizada")
                st.success("Ejecucion completa finalizada.")
                st.rerun()

        state = st.session_state.get("autoplay_state")
        if state:
            render_autoplay(state)

            export_payload = {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "brief": st.session_state["brief_result"],
                "plan_v3_2": st.session_state["plan_result"],
                "transcript": state.get("transcript", []),
                "traces": state.get("traces", []),
                "autoplay": {
                    "transcript": state.get("transcript", []),
                    "traces": state.get("traces", []),
                    "resumen_final": {
                        "done": bool(state.get("done")),
                        "motivo_cierre": state.get("final_reason", ""),
                        "turnos": int(state.get("turn_index", 0)),
                        "coverage_ratio": state.get("coverage_ratio", 0),
                        "foco_actual": state.get("foco_actual", ""),
                        "atributos_detectados_unicos": state.get("atributos_detectados_unicos", []),
                        "atributos_pendientes": state.get("atributos_pendientes", []),
                        "human_like_index": (
                            round(
                                sum(float(t.get("human_like_turno", 0.0)) for t in state.get("traces", []) if isinstance(t, dict))
                                / max(len([t for t in state.get("traces", []) if isinstance(t, dict)]), 1),
                                3,
                            )
                            if state.get("traces")
                            else 0.0
                        ),
                        "naturalidad_promedio": (
                            round(
                                sum(float(t.get("naturalidad_turno", 0.0)) for t in state.get("traces", []) if isinstance(t, dict))
                                / max(len([t for t in state.get("traces", []) if isinstance(t, dict)]), 1),
                                3,
                            )
                            if state.get("traces")
                            else 0.0
                        ),
                        "atributos_con_profundidad_suficiente": [
                            a
                            for a, d in (
                                state.get("_profundidad_por_atributo", {}).items()
                                if isinstance(state.get("_profundidad_por_atributo", {}), dict)
                                else []
                            )
                            if int(d) >= 2
                        ],
                        "cierres_prematuros_evitados": int(state.get("premature_closure_prevented", 0)),
                    },
                },
                "atributos_detectados_unicos": state.get("atributos_detectados_unicos", []),
                "atributos_pendientes": state.get("atributos_pendientes", []),
                "coverage_ratio": state.get("coverage_ratio", 0),
                "foco_actual": state.get("foco_actual", ""),
                "motivo_cierre": state.get("final_reason", ""),
                "human_like_index": (
                    round(
                        sum(float(t.get("human_like_turno", 0.0)) for t in state.get("traces", []) if isinstance(t, dict))
                        / max(len([t for t in state.get("traces", []) if isinstance(t, dict)]), 1),
                        3,
                    )
                    if state.get("traces")
                    else 0.0
                ),
            }
            project_slug = _project_slug(st.session_state["brief_result"])
            st.download_button(
                "Descargar resultados",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"{project_slug}_autoplay_result.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Presiona 'Iniciar autoplay' para crear la sesion.")

with tab_lab:
    st.markdown("### Laboratorio autoplay")
    if not st.session_state.get("brief_result") or not st.session_state.get("plan_result"):
        st.info("Primero genera un plan visual para habilitar laboratorio autoplay.")
    else:
        lc1, lc2, lc3 = st.columns([1, 1, 2])
        with lc1:
            n_runs_lab = st.number_input("Numero de corridas", min_value=1, max_value=200, value=20, step=1, key="lab_n_runs")
        with lc2:
            use_seed = st.checkbox("Usar seed fija", value=False, key="lab_use_seed")
            seed_val = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="lab_seed") if use_seed else None
        with lc3:
            st.caption("Ejecuta multiples corridas autoplay, calificalas y separalas en accepted/review/rejected.")

        if st.button("Ejecutar batch autoplay", use_container_width=True, key="btn_lab_run"):
            with st.spinner("Corriendo laboratorio autoplay..."):
                st.session_state["autoplay_lab_result"] = run_batch_autoplay(
                    brief=st.session_state["brief_result"],
                    plan=st.session_state["plan_result"],
                    n_runs=int(n_runs_lab),
                    seed=int(seed_val) if use_seed and seed_val is not None else None,
                )
            st.success("Batch autoplay finalizado.")
            st.rerun()

        lab_result = st.session_state.get("autoplay_lab_result")
        if not lab_result:
            st.info("Configura corridas y presiona 'Ejecutar batch autoplay'.")
        else:
            summary = lab_result.get("summary", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Corridas", int(summary.get("n_runs", 0)))
            m2.metric("Accepted", int(summary.get("accepted_count", 0)))
            m3.metric("Review", int(summary.get("review_count", 0)))
            m4.metric("Rejected", int(summary.get("rejected_count", 0)))
            m5, m6, m7 = st.columns(3)
            m5.metric("Accepted rate", f"{float(summary.get('accepted_rate', 0.0)):.2f}")
            m6.metric("Avg coverage", f"{float(summary.get('avg_coverage', 0.0)):.2f}")
            m7.metric("Avg human-like", f"{float(summary.get('avg_human_like_index', 0.0)):.2f}")
            m8, m9 = st.columns(2)
            m8.metric("Avg coverage util", f"{float(summary.get('avg_coverage_util', 0.0)):.2f}")
            m9.metric("Avg resp. genericas", f"{float(summary.get('avg_porcentaje_respuestas_genericas', 0.0)):.2f}")
            st.caption(f"Avg naturalidad: {float(summary.get('avg_naturalidad', 0.0)):.2f}")
            with st.expander("Failure counts"):
                st.json(summary.get("failure_counts", {}), expanded=False)

            accepted_runs = lab_result.get("accepted_runs", [])
            review_runs = lab_result.get("review_runs", [])
            rejected_runs = lab_result.get("rejected_runs", [])

            la, lr, lj = st.columns(3)
            with la:
                st.markdown("**Accepted runs**")
                st.write([r.get("run_id", "-") for r in accepted_runs])
            with lr:
                st.markdown("**Review runs**")
                st.write([r.get("run_id", "-") for r in review_runs])
            with lj:
                st.markdown("**Rejected runs**")
                st.write([r.get("run_id", "-") for r in rejected_runs])

            st.markdown("### Top 3 accepted")
            top_accepted = sorted(
                accepted_runs,
                key=lambda r: float((r.get("quality_scores", {}) or {}).get("human_like_index", 0.0)),
                reverse=True,
            )[:3]
            if not top_accepted:
                st.info("No hay corridas accepted en este batch.")
            for run in top_accepted:
                qs = run.get("quality_scores", {})
                with st.container(border=True):
                    st.markdown(f"**{run.get('run_id', '-')}**")
                    t1, t2, t3, t4 = st.columns(4)
                    t1.metric("Coverage", f"{float(qs.get('coverage_ratio_final', 0.0)):.2f}")
                    t2.metric("Naturalidad", f"{float(qs.get('naturalidad_promedio', 0.0)):.2f}")
                    t3.metric("Human-like", f"{float(qs.get('human_like_index', 0.0)):.2f}")
                    t4.metric("Profundidad>=2", int(qs.get("atributos_con_profundidad_suficiente", 0)))
                    fails = run.get("quality_failures", [])
                    if fails:
                        st.caption("Fallas: " + ", ".join(str(x) for x in fails))
                    else:
                        st.caption("Sin fallas.")

            project_slug = _project_slug(st.session_state["brief_result"])
            st.download_button(
                "Descargar JSON laboratorio autoplay",
                data=json.dumps(lab_result, ensure_ascii=False, indent=2),
                file_name=f"{project_slug}_autoplay_lab_v1.json",
                mime="application/json",
                use_container_width=True,
            )

with tab_live:
    st.markdown("### Entrevista en vivo (participante humano)")
    if not st.session_state.get("brief_result") or not st.session_state.get("plan_result"):
        st.info("Primero genera un plan visual para habilitar entrevista en vivo.")
    else:
        with st.expander("Ver client_plan / engine_blueprint"):
            p = st.session_state["plan_result"]
            st.json(
                {
                    "client_plan": p.get("client_plan", {}),
                    "engine_blueprint": p.get("engine_blueprint", {}),
                },
                expanded=False,
            )
        h1, h2 = st.columns(2)
        with h1:
            if st.button("Iniciar entrevista en vivo", use_container_width=True, key="btn_live_start"):
                st.session_state["live_state"] = start_session(
                    st.session_state["brief_result"], st.session_state["plan_result"]
                )
                st.session_state["live_result"] = None
                st.session_state["live_answer_pending_clear"] = True
                st.rerun()
        with h2:
            if st.button("Reiniciar entrevista", use_container_width=True, key="btn_live_reset"):
                st.session_state["live_state"] = start_session(
                    st.session_state["brief_result"], st.session_state["plan_result"]
                )
                st.session_state["live_result"] = None
                st.session_state["live_answer_pending_clear"] = True
                st.rerun()

        live_state = st.session_state.get("live_state")
        if not live_state:
            st.info("Presiona 'Iniciar entrevista en vivo' para comenzar.")
        else:
            st.markdown("### Pregunta actual")
            st.write(f"**Etapa:** {live_state.get('etapa_actual', '-')}")
            st.info(live_state.get("next_question", ""))

            st.text_area(
                "Respuesta del participante",
                key="live_answer_input",
                height=120,
                placeholder="Escribe aqui la respuesta del participante...",
            )
            if st.button("Enviar respuesta", use_container_width=True, key="btn_live_send"):
                answer = st.session_state.get("live_answer_input", "")
                if not str(answer).strip():
                    st.warning("Escribe una respuesta antes de enviar.")
                else:
                    result = step_with_human_answer(live_state, answer)
                    st.session_state["live_state"] = result["state"]
                    st.session_state["live_answer_pending_clear"] = True
                    if result.get("done"):
                        st.success("La entrevista en vivo ha terminado.")
                    st.rerun()

            render_autoplay(st.session_state["live_state"])

            export_payload_live = {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "brief": st.session_state["brief_result"],
                "plan_v3_2": st.session_state["plan_result"],
                "transcript": st.session_state["live_state"].get("transcript", []),
                "traces": st.session_state["live_state"].get("traces", []),
                "entrevista_vivo": {
                    "transcript": st.session_state["live_state"].get("transcript", []),
                    "traces": st.session_state["live_state"].get("traces", []),
                    "resumen_final": {
                        "done": bool(st.session_state["live_state"].get("done")),
                        "motivo_cierre": st.session_state["live_state"].get("final_reason", ""),
                        "turnos": int(st.session_state["live_state"].get("turn_index", 0)),
                        "coverage_ratio": st.session_state["live_state"].get("coverage_ratio", 0),
                        "foco_actual": st.session_state["live_state"].get("foco_actual", ""),
                        "atributos_detectados_unicos": st.session_state["live_state"].get("atributos_detectados_unicos", []),
                        "atributos_pendientes": st.session_state["live_state"].get("atributos_pendientes", []),
                        "human_like_index": (
                            round(
                                sum(float(t.get("human_like_turno", 0.0)) for t in st.session_state["live_state"].get("traces", []) if isinstance(t, dict))
                                / max(len([t for t in st.session_state["live_state"].get("traces", []) if isinstance(t, dict)]), 1),
                                3,
                            )
                            if st.session_state["live_state"].get("traces")
                            else 0.0
                        ),
                    },
                },
                "atributos_detectados_unicos": st.session_state["live_state"].get("atributos_detectados_unicos", []),
                "atributos_pendientes": st.session_state["live_state"].get("atributos_pendientes", []),
                "coverage_ratio": st.session_state["live_state"].get("coverage_ratio", 0),
                "foco_actual": st.session_state["live_state"].get("foco_actual", ""),
                "motivo_cierre": st.session_state["live_state"].get("final_reason", ""),
                "human_like_index": (
                    round(
                        sum(float(t.get("human_like_turno", 0.0)) for t in st.session_state["live_state"].get("traces", []) if isinstance(t, dict))
                        / max(len([t for t in st.session_state["live_state"].get("traces", []) if isinstance(t, dict)]), 1),
                        3,
                    )
                    if st.session_state["live_state"].get("traces")
                    else 0.0
                ),
            }
            project_slug = _project_slug(st.session_state["brief_result"])
            st.download_button(
                "Descargar resultados entrevista en vivo",
                data=json.dumps(export_payload_live, ensure_ascii=False, indent=2),
                file_name=f"{project_slug}_live_result.json",
                mime="application/json",
                use_container_width=True,
            )
