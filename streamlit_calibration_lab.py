from __future__ import annotations

import inspect
from typing import Any, Dict, List

import streamlit as st

from calibration_storage import load_guides, load_history, save_iteration_result

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from auto_calibrator import run_category_calibration  # type: ignore
except Exception:
    run_category_calibration = None


DEFAULT_CATEGORIES = ["universidad", "soda", "auto", "banco", "seguro", "retail", "tecnologia"]


def _to_table(rows: List[Dict[str, Any]]):
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _extract_categories(guides: Dict[str, Any]) -> List[str]:
    keys = []
    if isinstance(guides.get("categories"), dict):
        keys = list(guides["categories"].keys())
    if not keys and isinstance(guides.get("categorias"), dict):
        keys = list(guides["categorias"].keys())
    if not keys and isinstance(guides, dict):
        keys = [k for k, v in guides.items() if isinstance(v, dict)]
    merged = list(dict.fromkeys([str(x).strip().lower() for x in keys + DEFAULT_CATEGORIES if str(x).strip()]))
    return merged or DEFAULT_CATEGORIES


def _safe_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out or [123, 456]


def _call_run_category_calibration(categoria: str, max_iterations: int, seeds: List[int], n_runs: int) -> Dict[str, Any]:
    if run_category_calibration is None:
        raise RuntimeError("No se encontro auto_calibrator.py o run_category_calibration().")

    fn = run_category_calibration
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        low = name.lower()
        if low in {"categoria", "category"}:
            kwargs[name] = categoria
        elif low in {"max_iterations", "max_iter", "iteraciones", "iterations"}:
            kwargs[name] = max_iterations
        elif low in {"seeds", "seed_list"}:
            kwargs[name] = seeds
        elif low in {"n_runs", "runs", "runs_per_seed"}:
            kwargs[name] = n_runs
    return fn(**kwargs)  # type: ignore[arg-type]


def _latest_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return sorted(rows, key=lambda x: int(x.get("iteracion", 0)))[-1]


def _previous_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda x: int(x.get("iteracion", 0)))
    if len(ordered) < 2:
        return {}
    return ordered[-2]


def _metric_delta(curr: Dict[str, Any], prev: Dict[str, Any], key: str) -> str:
    if not curr or not prev:
        return "n/a"
    try:
        d = float(curr.get(key, 0.0)) - float(prev.get(key, 0.0))
        return f"{d:+.3f}"
    except Exception:
        return "n/a"


def _render_cases(payload: Dict[str, Any]) -> None:
    accepted = payload.get("accepted_runs", []) if isinstance(payload, dict) else []
    review = payload.get("review_runs", []) if isinstance(payload, dict) else []
    rejected = payload.get("rejected_runs", []) if isinstance(payload, dict) else []

    st.subheader("Mejores casos")
    st.dataframe(_to_table(accepted[:10]))
    st.subheader("Casos en revision")
    st.dataframe(_to_table(review[:10]))
    st.subheader("Peores casos")
    st.dataframe(_to_table(rejected[:10]))


def main() -> None:
    st.set_page_config(page_title="Calibration Lab", layout="wide")
    st.title("Calibration Lab del Moderador IA")

    guides = load_guides()
    categories = _extract_categories(guides)

    if "calib_result" not in st.session_state:
        st.session_state["calib_result"] = {}
    if "calib_history_rows" not in st.session_state:
        st.session_state["calib_history_rows"] = []

    with st.sidebar:
        st.header("Control")
        categoria = st.selectbox("Categoria", options=categories, index=0)
        max_iterations = st.number_input("Numero maximo de iteraciones", min_value=1, max_value=200, value=10, step=1)
        seeds_txt = st.text_input("Seeds (coma separadas)", value="123,456")
        n_runs = st.number_input("n_runs por iteracion", min_value=1, max_value=200, value=20, step=1)

        run_btn = st.button("Ejecutar calibracion", use_container_width=True, type="primary")
        load_btn = st.button("Cargar historico", use_container_width=True)
        clear_btn = st.button("Limpiar vista", use_container_width=True)

    if load_btn:
        st.session_state["calib_history_rows"] = load_history(categoria)
        st.success("Historico cargado.")

    if clear_btn:
        st.session_state["calib_result"] = {}
        st.session_state["calib_history_rows"] = []
        st.info("Vista limpiada.")

    if run_btn:
        try:
            seeds = _safe_list(seeds_txt)
            result = _call_run_category_calibration(
                categoria=categoria,
                max_iterations=int(max_iterations),
                seeds=seeds,
                n_runs=int(n_runs),
            )
            st.session_state["calib_result"] = result
            row = save_iteration_result(categoria, result if isinstance(result, dict) else {"summary": {}})
            st.session_state["calib_history_rows"] = load_history(categoria)
            st.success(f"Calibracion completada. Iteracion guardada: {row.get('iteracion')}.")
        except Exception as exc:
            st.error(f"No fue posible ejecutar calibracion: {exc}")

    history_rows = st.session_state.get("calib_history_rows", [])
    result = st.session_state.get("calib_result", {})
    current = _latest_row(history_rows)
    previous = _previous_row(history_rows)

    tabs = st.tabs(["Dashboard", "Historico", "Diagnostico", "Casos", "Configuracion"])

    with tabs[0]:
        st.subheader("Estado actual")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Categoria", str(categoria))
        c2.metric("Ultima iteracion", int(current.get("iteracion", 0)) if current else 0)
        c3.metric("Estado", "cumple metas" if current.get("cumple_metas") else "pendiente")
        c4.metric("Fecha", str(current.get("fecha", "-"))[:19] if current else "-")

        m1, m2, m3 = st.columns(3)
        m1.metric("Accepted", f"{float(current.get('accepted', 0.0)):.3f}", _metric_delta(current, previous, "accepted"))
        m2.metric("Review", f"{float(current.get('review', 0.0)):.3f}", _metric_delta(current, previous, "review"))
        m3.metric("Rejected", f"{float(current.get('rejected', 0.0)):.3f}", _metric_delta(current, previous, "rejected"))

        m4, m5, m6 = st.columns(3)
        m4.metric("Coverage", f"{float(current.get('coverage', 0.0)):.3f}", _metric_delta(current, previous, "coverage"))
        m5.metric("Coverage util", f"{float(current.get('coverage_util', 0.0)):.3f}", _metric_delta(current, previous, "coverage_util"))
        m6.metric("Human like", f"{float(current.get('human_like', 0.0)):.3f}", _metric_delta(current, previous, "human_like"))

    with tabs[1]:
        st.subheader("Historico acumulado")
        if not history_rows:
            st.info("No hay historico para esta categoria.")
        else:
            st.dataframe(_to_table(history_rows), use_container_width=True)
            table = _to_table(history_rows)
            if pd is not None and not table.empty:
                line_base = table.sort_values("iteracion").set_index("iteracion")
                st.line_chart(line_base[["accepted"]])
                st.line_chart(line_base[["coverage_util"]])
                st.line_chart(line_base[["human_like"]])

    with tabs[2]:
        st.subheader("Diagnostico")
        if not current:
            st.info("Sin datos de calibracion.")
        else:
            st.write("Comparacion contra iteracion previa")
            st.write(
                {
                    "accepted_delta": _metric_delta(current, previous, "accepted"),
                    "coverage_util_delta": _metric_delta(current, previous, "coverage_util"),
                    "human_like_delta": _metric_delta(current, previous, "human_like"),
                }
            )
            with st.expander("Fallos detectados", expanded=True):
                st.write(current.get("diagnostico", []))
            with st.expander("Recomendaciones", expanded=True):
                st.write(current.get("recomendaciones", []))

    with tabs[3]:
        st.subheader("Casos")
        if not result:
            st.info("Ejecuta calibracion para ver casos.")
        else:
            _render_cases(result if isinstance(result, dict) else {})

    with tabs[4]:
        st.subheader("Configuracion")
        st.write(
            {
                "categoria": categoria,
                "max_iterations": int(max_iterations),
                "seeds": _safe_list(seeds_txt),
                "n_runs": int(n_runs),
            }
        )
        with st.expander("Metas y guias (calibration_guides.json)", expanded=True):
            st.json(guides)


if __name__ == "__main__":
    main()

