from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "calibration_history_master.json"
GUIDES_FILE = BASE_DIR / "calibration_guides.json"
PUBLISHED_CATEGORIES_FILE = BASE_DIR / "published_categories.json"
TRAINED_CATEGORIES_DIR = BASE_DIR / "trained_categories"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history(categoria: Optional[str] = None) -> List[Dict[str, Any]]:
    history = _read_json(HISTORY_FILE, [])
    if not isinstance(history, list):
        return []
    if not categoria:
        return history
    cat = str(categoria).strip().lower()
    return [row for row in history if str(row.get("categoria", "")).strip().lower() == cat]


def load_guides() -> Dict[str, Any]:
    guides = _read_json(GUIDES_FILE, {})
    if isinstance(guides, dict):
        return guides
    return {}


def save_guides(guides: Dict[str, Any]) -> None:
    _write_json(GUIDES_FILE, guides if isinstance(guides, dict) else {})


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick(source: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return default


def _next_iteration(history: List[Dict[str, Any]], categoria: str) -> int:
    filtered = [int(row.get("iteracion", 0)) for row in history if str(row.get("categoria", "")).lower() == categoria]
    return (max(filtered) + 1) if filtered else 1


def save_iteration_result(categoria: str, resultado: Dict[str, Any]) -> Dict[str, Any]:
    history = load_history()
    cat = str(categoria or "proyecto").strip().lower()
    summary = resultado.get("summary", {}) if isinstance(resultado, dict) else {}

    record = {
        "categoria": cat,
        "iteracion": _next_iteration(history, cat),
        "accepted": _to_float(
            _pick(summary, ["accepted_rate", "accepted", "accepted_count"], _pick(resultado, ["accepted_rate", "accepted", "accepted_count"], 0.0)),
            0.0,
        ),
        "review": _to_float(
            _pick(summary, ["review_rate", "review", "review_count"], _pick(resultado, ["review_rate", "review", "review_count"], 0.0)),
            0.0,
        ),
        "rejected": _to_float(
            _pick(summary, ["rejected_rate", "rejected", "rejected_count"], _pick(resultado, ["rejected_rate", "rejected", "rejected_count"], 0.0)),
            0.0,
        ),
        "coverage": _to_float(_pick(summary, ["avg_coverage", "coverage_ratio_final", "coverage"], _pick(resultado, ["avg_coverage", "coverage_ratio_final", "coverage"], 0.0))),
        "coverage_util": _to_float(
            _pick(summary, ["avg_coverage_util", "coverage_util", "coverage_util_final"], _pick(resultado, ["avg_coverage_util", "coverage_util", "coverage_util_final"], 0.0))
        ),
        "human_like": _to_float(
            _pick(summary, ["avg_human_like_index", "human_like_index", "human_like"], _pick(resultado, ["avg_human_like_index", "human_like_index", "human_like"], 0.0))
        ),
        "cumple_metas": bool(_pick(resultado, ["cumple_metas"], _pick(summary, ["cumple_metas"], False))),
        "diagnostico": _pick(resultado, ["diagnostico", "diagnostico_automatico"], []),
        "recomendaciones": _pick(resultado, ["recomendaciones"], []),
        "principal_cuello_botella": _pick(resultado, ["principal_cuello_botella"], "sin_diagnostico"),
        "archivo_objetivo": _pick(resultado, ["archivo_objetivo"], "interview_engine.py"),
        "cambio_sugerido": _pick(resultado, ["cambio_sugerido"], ""),
        "codex_prompt_sugerido": _pick(resultado, ["codex_prompt_sugerido"], ""),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "raw_result": resultado if isinstance(resultado, dict) else {},
    }

    history.append(record)
    _write_json(HISTORY_FILE, history)
    return record


def load_published_categories() -> Dict[str, Any]:
    payload = _read_json(PUBLISHED_CATEGORIES_FILE, {})
    if not isinstance(payload, dict):
        payload = {}
    categorias = payload.get("categorias_publicadas", [])
    metadata = payload.get("metadata", {})
    if not isinstance(categorias, list):
        categorias = []
    if not isinstance(metadata, dict):
        metadata = {}
    categorias_norm = [str(c).strip().lower() for c in categorias if str(c).strip()]
    metadata_norm: Dict[str, Any] = {}
    for key, value in metadata.items():
        slug = str(key).strip().lower()
        if not slug:
            continue
        metadata_norm[slug] = value if isinstance(value, dict) else {}
    return {
        "categorias_publicadas": sorted(list(dict.fromkeys(categorias_norm))),
        "metadata": metadata_norm,
    }


def save_published_categories(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe = {
        "categorias_publicadas": [],
        "metadata": {},
    }
    if isinstance(payload, dict):
        safe = {
            "categorias_publicadas": payload.get("categorias_publicadas", []),
            "metadata": payload.get("metadata", {}),
        }
    normalized = load_published_categories() if not safe else {
        "categorias_publicadas": [str(c).strip().lower() for c in safe.get("categorias_publicadas", []) if str(c).strip()],
        "metadata": {
            str(k).strip().lower(): (v if isinstance(v, dict) else {})
            for k, v in (safe.get("metadata", {}) if isinstance(safe.get("metadata", {}), dict) else {}).items()
            if str(k).strip()
        },
    }
    normalized["categorias_publicadas"] = sorted(list(dict.fromkeys(normalized["categorias_publicadas"])))
    _write_json(PUBLISHED_CATEGORIES_FILE, normalized)
    return normalized


def is_category_available_for_client(categoria: str, published_payload: Optional[Dict[str, Any]] = None) -> bool:
    slug = str(categoria or "").strip().lower()
    if not slug:
        return False
    payload = published_payload if isinstance(published_payload, dict) else load_published_categories()
    categorias = payload.get("categorias_publicadas", []) if isinstance(payload, dict) else []
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if slug not in categorias:
        return False
    item = metadata.get(slug, {}) if isinstance(metadata, dict) else {}
    if not isinstance(item, dict):
        item = {}
    return bool(item.get("cumple_guias", False)) and bool(item.get("aprobada_manual", False))


def _metric_from_payload(latest_metrics: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    if not isinstance(latest_metrics, dict):
        return default
    for key in keys:
        if key in latest_metrics and latest_metrics.get(key) is not None:
            return _to_float(latest_metrics.get(key), default)
    summary = latest_metrics.get("summary", {})
    if isinstance(summary, dict):
        for key in keys:
            if key in summary and summary.get(key) is not None:
                return _to_float(summary.get(key), default)
    return default


def evaluate_category_publishability(
    categoria: str,
    latest_metrics: Dict[str, Any],
    guide_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    cat = str(categoria or "").strip().lower() or "producto_generico"
    accepted = _metric_from_payload(latest_metrics, ["accepted", "accepted_rate", "accepted_count"], 0.0)
    coverage_util = _metric_from_payload(latest_metrics, ["coverage_util", "avg_coverage_util", "coverage_util_final"], 0.0)
    human_like = _metric_from_payload(latest_metrics, ["human_like", "human_like_index", "avg_human_like_index"], 0.0)
    accepted_min = _to_float(guide_metrics.get("accepted_min", 0.25), 0.25) if isinstance(guide_metrics, dict) else 0.25
    coverage_util_min = _to_float(guide_metrics.get("coverage_util_min", 0.35), 0.35) if isinstance(guide_metrics, dict) else 0.35
    human_like_min = _to_float(guide_metrics.get("human_like_min", 0.70), 0.70) if isinstance(guide_metrics, dict) else 0.70

    motivos: List[str] = []
    if accepted < accepted_min:
        motivos.append(f"accepted {accepted:.3f} < {accepted_min:.3f}")
    if coverage_util < coverage_util_min:
        motivos.append(f"coverage_util {coverage_util:.3f} < {coverage_util_min:.3f}")
    if human_like < human_like_min:
        motivos.append(f"human_like {human_like:.3f} < {human_like_min:.3f}")

    return {
        "categoria": cat,
        "cumple_guias": len(motivos) == 0,
        "motivos": motivos if motivos else ["Cumple accepted, coverage_util y human_like."],
        "recomendada_para_publicar": len(motivos) == 0,
        "metricas": {
            "accepted": accepted,
            "coverage_util": coverage_util,
            "human_like": human_like,
        },
        "guias": {
            "accepted_min": accepted_min,
            "coverage_util_min": coverage_util_min,
            "human_like_min": human_like_min,
        },
    }


def _load_file_if_exists(path: Path, default: Any) -> Any:
    return _read_json(path, default) if path.exists() else default


def load_trained_category(categoria: str) -> Dict[str, Any]:
    slug = str(categoria or "").strip().lower()
    if not slug:
        raise ValueError("Categoria vacia. Proporciona una categoria valida.")
    category_dir = TRAINED_CATEGORIES_DIR / slug
    if not category_dir.exists():
        raise FileNotFoundError(
            f"No existe la categoria entrenada '{slug}' en {TRAINED_CATEGORIES_DIR}. "
            "Crea la carpeta trained_categories/<categoria>/ con brief.json y latest_plan.json."
        )

    brief = _load_file_if_exists(category_dir / "brief.json", {})
    plan = _load_file_if_exists(category_dir / "latest_plan.json", {})
    model_config = _load_file_if_exists(category_dir / "model_config.json", {})
    latest_metrics = _load_file_if_exists(category_dir / "latest_metrics.json", {})
    semantic_seed = _load_file_if_exists(category_dir / "semantic_seed.json", {})
    metadata = _load_file_if_exists(category_dir / "metadata.json", {})

    if not isinstance(brief, dict) or not brief:
        raise RuntimeError(f"Falta brief.json valido en {category_dir}.")
    if not isinstance(plan, dict) or not plan:
        raise RuntimeError(f"Falta latest_plan.json valido en {category_dir}.")

    return {
        "categoria": slug,
        "path": str(category_dir),
        "brief": brief,
        "plan": plan,
        "configuracion": model_config if isinstance(model_config, dict) else {},
        "latest_metrics": latest_metrics if isinstance(latest_metrics, dict) else {},
        "semantic_seed": semantic_seed if isinstance(semantic_seed, dict) else {},
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def ensure_trained_categories_structure() -> Dict[str, Any]:
    TRAINED_CATEGORIES_DIR.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    updated: List[str] = []

    legacy_export = BASE_DIR / "Universidades Correcto.json"
    target_slug = "universidad"
    target_dir = TRAINED_CATEGORIES_DIR / target_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    if legacy_export.exists():
        payload = _read_json(legacy_export, {})
        if isinstance(payload, dict):
            brief = payload.get("brief", {})
            plan = payload.get("plan", payload.get("plan_v3_2", {}))
            if isinstance(brief, dict) and brief:
                (target_dir / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
                created.append(str(target_dir / "brief.json"))
            if isinstance(plan, dict) and plan:
                (target_dir / "latest_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                created.append(str(target_dir / "latest_plan.json"))

    model_cfg_path = target_dir / "model_config.json"
    if not model_cfg_path.exists():
        brief_payload = _load_file_if_exists(target_dir / "brief.json", {})
        model_cfg = brief_payload.get("config", {}) if isinstance(brief_payload, dict) else {}
        model_cfg_path.write_text(json.dumps(model_cfg if isinstance(model_cfg, dict) else {}, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(str(model_cfg_path))

    metrics_path = target_dir / "latest_metrics.json"
    if not metrics_path.exists():
        history = load_history(target_slug)
        latest = history[-1] if history else {}
        metrics_path.write_text(json.dumps(latest if isinstance(latest, dict) else {}, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(str(metrics_path))

    metadata_path = target_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_payload = {
            "display_name": "Universidades",
            "version_modelo": "v1.0",
            "estado": "entrenada",
        }
        metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(str(metadata_path))

    seed_src = BASE_DIR / "categorias" / "demo_producto" / "semantic_seed.json"
    seed_dst = target_dir / "semantic_seed.json"
    if not seed_dst.exists():
        if seed_src.exists():
            seed_dst.write_text(seed_src.read_text(encoding="utf-8"), encoding="utf-8")
            updated.append(str(seed_dst))
        else:
            seed_dst.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
            created.append(str(seed_dst))

    return {
        "trained_categories_dir": str(TRAINED_CATEGORIES_DIR),
        "created_or_seeded_files": created,
        "updated_files": updated,
    }
