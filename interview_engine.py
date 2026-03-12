import json
import os
import random
import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher
from math import ceil
from typing import Any, Dict, List, Optional

from openai import OpenAI

ETAPAS_FIJAS = [
    "apertura_y_encuadre",
    "exploracion_espontanea",
    "probing_por_atributos_objetivo",
    "evidencia_y_ejemplos",
    "priorizacion_y_contraste",
    "confirmacion",
    "cierre",
]

ACCIONES_VALIDAS = [
    "profundizar",
    "pedir_ejemplo",
    "aclarar",
    "explorar_nuevo_tema",
    "reformular",
    "reconducir",
    "confirmacion",
    "cerrar",
]

STOPWORDS_ES = {
    "y",
    "o",
    "de",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "en",
    "con",
    "por",
    "para",
    "que",
    "del",
    "al",
    "se",
    "es",
    "muy",
    "mas",
    "maso",
    "como",
    "sobre",
    "entre",
    "sin",
    "a",
    "mi",
    "tu",
    "su",
    "sus",
    "tan",
    "este",
    "esta",
}

VAGUE_MARKERS = [
    "esta bien",
    "esta padre",
    "me gusta",
    "normal",
    "mas o menos",
    "depende",
]

EMPTY_MARKERS = [
    "no se",
    "no estoy seguro",
    "no tengo informacion",
    "no sabria",
    "ni idea",
]

CONFUSION_MARKERS_HARD = [
    "no entendi",
    "reformula",
    "puedes explicarlo mejor",
    "a que te refieres",
]

CONFUSION_MARKERS_SOFT = [
    "no estoy seguro",
    "no se",
    "no sabria",
    "ni idea",
]

WEAK_EVIDENCE_MARKERS = [
    "no me late",
    "si eso falla",
    "esta bien",
    "me gusta",
    "normal",
    "depende",
    "este tan",
]

SEMANTIC_MAP: Dict[str, str] = {
    "profesor": "calidad academica",
    "profesores": "calidad academica",
    "maestro": "calidad academica",
    "maestros": "calidad academica",
    "docente": "calidad academica",
    "docentes": "calidad academica",
    "buenos docentes": "calidad academica",
    "instalacion": "infraestructura",
    "instalaciones": "infraestructura",
    "campus": "infraestructura",
    "laboratorio": "infraestructura",
    "laboratorios": "infraestructura",
    "prestigio": "reputacion",
    "reconocimiento": "reputacion",
    "reconocida": "reputacion",
    "tiene nombre": "reputacion",
    "buen prestigio": "reputacion",
    "tiene prestigio": "reputacion",
    "prestigiosa": "reputacion",
    "nivel educativo": "calidad academica",
    "buen nivel": "calidad academica",
    "buena ensenanza": "calidad academica",
    "buenos maestros": "calidad academica",
    "buenos profesores": "calidad academica",
    "buenas clases": "calidad academica",
    "buen campus": "infraestructura",
    "buenas instalaciones": "infraestructura",
    "buena bolsa de trabajo": "empleabilidad",
    "bolsa de trabajo": "empleabilidad",
    "bolsa de trabajo": "empleabilidad",
    "salida laboral": "empleabilidad",
    "costo justo": "costo",
    "no tan cara": "costo",
    "vale lo que cuesta": "costo",
    "buen ambiente": "ambiente universitario",
    "ambiente estudiantil": "ambiente universitario",
    "sabor": "sabor",
    "fresca": "sabor refrescante",
    "refrescante": "sabor refrescante",
    "rico": "sabor",
    "rica": "sabor",
    "feo": "sabor",
    "dulce": "nivel_azucar",
    "empalagosa": "nivel_azucar",
    "azucar": "nivel_azucar",
    "muy dulce": "nivel_azucar",
    "gas": "carbonatacion burbujeo",
    "burbujas": "carbonatacion burbujeo",
    "no tiene gas": "carbonatacion burbujeo",
    "precio": "precio",
    "cara": "precio",
    "barata": "precio",
    "economica": "precio",
    "burbujeo": "carbonatacion burbujeo",
    "carbonatacion": "carbonatacion burbujeo",
    "rapido": "desempeno",
    "lento": "desempeno",
    "seguro": "seguridad",
    "estable": "seguridad",
    "comodo": "comodidad",
    "incomodo": "comodidad",
    "espacioso": "espacio",
    "apretado": "espacio",
    "consumo": "rendimiento",
    "gasta mucho": "consumo",
    "ahorrador": "consumo",
    "economico": "consumo",
    "carrera": "oferta academica",
    "carreras": "oferta academica",
    "programa": "oferta academica",
    "programas": "oferta academica",
}
# compatibilidad interna
SEMANTIC_EQUIVALENCES = SEMANTIC_MAP

UNIVERSIDAD_NATURAL_SIGNAL_MAP: Dict[str, str] = {
    "buenos maestros": "calidad academica",
    "buenos profesores": "calidad academica",
    "nivel educativo": "calidad academica",
    "buen nivel": "calidad academica",
    "buena ensenanza": "calidad academica",
    "buenas clases": "calidad academica",
    "buenos docentes": "calidad academica",
    "reconocida": "reputacion",
    "tiene nombre": "reputacion",
    "buen prestigio": "reputacion",
    "tiene prestigio": "reputacion",
    "buenas instalaciones": "infraestructura",
    "buen campus": "infraestructura",
    "laboratorios": "infraestructura",
    "buena bolsa de trabajo": "empleabilidad",
    "bolsa de trabajo": "empleabilidad",
    "salida laboral": "empleabilidad",
    "costo justo": "costo",
    "no tan cara": "costo",
    "vale lo que cuesta": "costo",
    "buen ambiente": "ambiente universitario",
    "ambiente estudiantil": "ambiente universitario",
}

EMERGENT_BAD_TOKENS = {
    "estoy",
    "seguro",
    "entiendo",
    "entiendas",
    "entienda",
    "refieres",
    "gusta",
    "padre",
    "normal",
    "cosa",
    "tema",
    "eso",
    "aqui",
    "ahi",
}

ATTR_ITEMS_SCHEMA: Dict[str, Any] = {
    "name": "attribute_items_eval",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["atributo", "cuenta", "evidencia", "confianza", "razon_corta"],
                    "properties": {
                        "atributo": {"type": "string"},
                        "cuenta": {"type": "boolean"},
                        "evidencia": {"type": "string", "maxLength": 120},
                        "confianza": {"type": "string", "enum": ["baja", "media", "alta"]},
                        "razon_corta": {"type": "string", "maxLength": 160},
                    },
                },
            }
        },
    },
}


def _get_client() -> Any:
    key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None


def _normalize_text(text: str) -> str:
    raw = (text or "").lower()
    raw = unicodedata.normalize("NFKD", raw)
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", raw)).strip()


def detectar_objeto_estudio(brief: Dict[str, Any]) -> Dict[str, str]:
    base = brief.get("brief", {}) if isinstance(brief, dict) else {}
    text = _normalize_text(
        f"{base.get('antecedente','')} {base.get('objetivo_principal','')} {brief.get('contexto','') if isinstance(brief, dict) else ''}"
    )
    if "universidad" in text or "universidades" in text:
        return {
            "categoria": "universidad",
            "sustantivo_singular": "universidad",
            "sustantivo_plural": "universidades",
            "verbo_experiencia": "elegir",
            "verbo_eleccion": "elegir",
        }
    if any(x in text for x in ["soda", "refresco", "refrescos"]):
        return {
            "categoria": "soda",
            "sustantivo_singular": "soda",
            "sustantivo_plural": "sodas",
            "verbo_experiencia": "tomar",
            "verbo_eleccion": "elegir",
        }
    if any(x in text for x in ["auto", "autos", "vehiculo", "vehiculos"]):
        return {
            "categoria": "auto",
            "sustantivo_singular": "auto",
            "sustantivo_plural": "autos",
            "verbo_experiencia": "usar",
            "verbo_eleccion": "comprar",
        }
    return {
        "categoria": "producto_generico",
        "sustantivo_singular": "opcion",
        "sustantivo_plural": "opciones",
        "verbo_experiencia": "usar",
        "verbo_eleccion": "elegir",
    }


def _slugify(value: str) -> str:
    base = _normalize_text(value).replace(" ", "_").strip("_")
    return base[:80] if base else "proyecto"


def identidad_caso(brief: Dict[str, Any]) -> Dict[str, str]:
    obj = detectar_objeto_estudio(brief or {})
    categoria = str(obj.get("categoria", "producto_generico")).strip() or "producto_generico"
    base = brief.get("brief", {}) if isinstance(brief, dict) else {}
    objetivo = str(base.get("objetivo_principal", "")).strip() or str(base.get("antecedente", "")).strip()
    slug_objetivo = _slugify(objetivo)
    slug_categoria = _slugify(categoria)
    return {
        "categoria": categoria,
        "slug_categoria": slug_categoria,
        "slug_objetivo": slug_objetivo,
        "slug_caso": f"{slug_categoria}_{slug_objetivo}".strip("_")[:120],
        "sustantivo_singular": str(obj.get("sustantivo_singular", "opcion")),
        "sustantivo_plural": str(obj.get("sustantivo_plural", "opciones")),
    }


def unir_prep_articulo(frase: str) -> str:
    out = " ".join(str(frase or "").split())
    out = re.sub(r"\bde el\b", "del", out, flags=re.IGNORECASE)
    out = re.sub(r"\ba el\b", "al", out, flags=re.IGNORECASE)
    return out.strip()


def contexto_categoria(brief: Dict[str, Any]) -> Dict[str, str]:
    return detectar_objeto_estudio(brief)


def frase_experiencia_categoria(brief: Dict[str, Any]) -> str:
    cat = detectar_objeto_estudio(brief).get("categoria", "producto_generico")
    if cat == "soda":
        return "cuando la tomas"
    if cat == "universidad":
        return "cuando la eliges"
    if cat == "auto":
        return "cuando lo manejas"
    return "cuando la usas"


def frase_comparacion_categoria(brief: Dict[str, Any]) -> str:
    obj = detectar_objeto_estudio(brief)
    return f"cuando comparas {obj.get('sustantivo_plural', 'opciones')}"


def _frase_importa(concepto: str) -> str:
    c = str(concepto or "").strip().lower()
    if c.startswith("las ") or c.startswith("los "):
        return f"Por que te importan {concepto}?"
    return f"Por que te importa {concepto}?"


def alias_natural_atributo(attr: str, brief: Optional[Dict[str, Any]] = None) -> str:
    low = _normalize_text(attr)
    categoria = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
    if categoria == "universidad":
        if any(x in low for x in ["calidad academica", "academica", "clases"]):
            return "la calidad de clases"
        if any(x in low for x in ["prestigio", "reputacion", "nombre"]):
            return "el prestigio"
        if any(x in low for x in ["infraestructura", "instalaciones", "recursos"]):
            return "las instalaciones"
        if any(x in low for x in ["costo", "accesibilidad", "colegiatura", "precio"]):
            return "el costo"
        if any(x in low for x in ["empleabilidad", "oportunidades laborales", "trabajo"]):
            return "las oportunidades de trabajo"
        if any(x in low for x in ["ambiente universitario", "experiencia estudiantil", "ambiente"]):
            return "el ambiente universitario"
    if categoria == "soda":
        if "sabor refrescante" in low:
            return "que se sienta fresca"
        if "nivel de azucar" in low or "azucar" in low or "dulce" in low:
            return "que no este muy dulce"
        if "burbujeo" in low or "burbuja" in low or "carbonat" in low or "gas" in low:
            return "que tenga buen gas"
        if "precio" in low or "costo" in low:
            return "que no este tan cara"
        if "sabor" in low:
            return "que sepa rico"
    if categoria == "auto":
        if "rendimiento" in low or "desempeno" in low or "consumo" in low:
            return "que rinda bien"
        if "seguridad" in low or "seguro" in low:
            return "que se sienta seguro"
        if "comodidad" in low or "comodo" in low:
            return "que sea comodo"
        if "espacio" in low or "espacioso" in low:
            return "que sea espacioso"
        if "precio" in low or "costo" in low:
            return "que no este tan caro"
    if "nivel de azucar" in low or "azucar" in low or "dulce" in low:
        return "que tan dulce es"
    if "burbujeo" in low or "burbuja" in low or "carbonat" in low or "gas" in low:
        return "el gas"
    if "precio" in low or "costo" in low:
        return "el precio"
    if "sabor" in low:
        return "el sabor"
    if "instal" in low:
        return "las instalaciones"
    if "calidad academica" in low:
        return "la calidad de clases"
    clean_low = " ".join(str(attr or "").strip().split()).lower()
    if not clean_low:
        return "este tema"
    if clean_low.startswith(("el ", "la ", "los ", "las ", "que ")):
        return unir_prep_articulo(clean_low)
    return unir_prep_articulo(clean_low)


def _natural_phrases_for_attr(attr: str, brief: Optional[Dict[str, Any]] = None) -> List[str]:
    low = _normalize_text(attr)
    categoria = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
    if categoria == "universidad":
        if any(x in low for x in ["calidad academica", "academica", "clases"]):
            return ["que las clases sean buenas", "que los profes expliquen bien", "que se aprenda bien"]
        if any(x in low for x in ["prestigio", "reputacion", "nombre"]):
            return ["que tenga buen nombre", "que se vea reconocida", "que me de confianza"]
        if any(x in low for x in ["infraestructura", "instalaciones", "recursos"]):
            return ["que tenga buenas instalaciones", "que los espacios esten bien", "que tenga buenos recursos"]
        if any(x in low for x in ["costo", "accesibilidad", "colegiatura", "precio"]):
            return ["que no sea tan cara", "que el costo sea justo", "que pueda pagarla"]
        if any(x in low for x in ["empleabilidad", "oportunidades laborales", "trabajo"]):
            return ["que ayude a conseguir trabajo", "que tenga buenas practicas", "que abra oportunidades"]
        if any(x in low for x in ["ambiente universitario", "experiencia estudiantil", "ambiente"]):
            return ["que el ambiente sea bueno", "que me sienta comodo", "que haya buena experiencia"]
    if "sabor" in low:
        return ["que sepa rico", "que sepa bien", "que no sepa feo"]
    if "burbujeo" in low or "burbuja" in low or "gas" in low:
        return ["que tenga buen gas", "que se sienta fuerte", "que no se le vaya rapido el gas"]
    if "azucar" in low or "dulce" in low:
        return ["que no este muy dulce", "que no empalague", "que se sienta ligera"]
    if "precio" in low or "costo" in low:
        return ["que no este tan cara", "que cueste bien", "que valga lo que cuesta"]
    alias = alias_natural_atributo(attr, brief)
    return [f"que {alias} me guste", f"que {alias} se sienta bien", "que no me falle"]


def _humanize_simulated_response(text: str, brief: Dict[str, Any]) -> str:
    out = " ".join(str(text or "").split()).strip()
    if not out:
        return out
    for attr in _allowed_attributes(brief):
        attr_low = _normalize_text(attr)
        if not attr_low:
            continue
        if attr_low in _normalize_text(out):
            safe_phrases = [p for p in _natural_phrases_for_attr(attr, brief) if attr_low not in _normalize_text(p)]
            phrase = random.choice(safe_phrases or _natural_phrases_for_attr(attr, brief))
            out = re.sub(rf"\b{re.escape(attr)}\b", phrase, out, count=1, flags=re.IGNORECASE)
    return out


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize_text(text))
    return [t for t in toks if len(t) >= 3 and t not in STOPWORDS_ES]


def _canonical_semantic_token(token: str) -> str:
    t = str(token or "").strip().lower()
    semantic_map = {
        # universidad
        "educativo": "academica",
        "educacion": "academica",
        "academico": "academica",
        "aprendizaje": "academica",
        "docentes": "profesores",
        "docente": "profesores",
        "maestros": "profesores",
        "maestro": "profesores",
        "ensenanza": "academica",
        "nivel": "calidad",
        "reconocida": "prestigio",
        "reconocido": "prestigio",
        "fama": "prestigio",
        "nombre": "prestigio",
        "prestigiosa": "prestigio",
        "prestigioso": "prestigio",
        "infraestructura": "instalaciones",
        "laboratorio": "instalaciones",
        "laboratorios": "instalaciones",
        "campus": "instalaciones",
        "biblioteca": "instalaciones",
        "colegiatura": "costo",
        "cuota": "costo",
        "cuotas": "costo",
        "precio": "costo",
        "cara": "costo",
        "barata": "costo",
        "economica": "costo",
        "empleo": "trabajo",
        "empleabilidad": "trabajo",
        "practicas": "trabajo",
        "practica": "trabajo",
        "bolsa": "trabajo",
        "laboral": "trabajo",
        "salida": "trabajo",
        "ambiente": "ambiente",
        "estudiantil": "ambiente",
        "campus": "ambiente",
        # soda/producto
        "gas": "burbujeo",
        "burbujas": "burbujeo",
        "dulce": "azucar",
    }
    return semantic_map.get(t, t)


def _max_semantic_similarity(resp_tokens: List[str], attr_tokens: List[str]) -> float:
    if not resp_tokens or not attr_tokens:
        return 0.0
    resp_text = " ".join(resp_tokens)
    attr_text = " ".join(attr_tokens)
    best = SequenceMatcher(None, resp_text, attr_text).ratio()

    max_n = min(4, len(resp_tokens))
    for n in range(1, max_n + 1):
        for i in range(0, len(resp_tokens) - n + 1):
            chunk = " ".join(resp_tokens[i : i + n])
            best = max(best, SequenceMatcher(None, chunk, attr_text).ratio())
    return best


def _semantic_target_matches_attribute(mapped_target: str, attribute: str) -> bool:
    mapped_norm = str(mapped_target or "").replace("_", " ")
    attr_norm = str(attribute or "").replace("_", " ")
    mapped_tokens = set(_tokenize(mapped_norm))
    attr_tokens = set(_tokenize(attr_norm)).union(set(_tokenize(alias_natural_atributo(attr_norm, {}))))
    alias_map = {
        "calidad academica": {"calidad_academica"},
        "ambiente universitario": {"ambiente_universitario"},
    }
    mapped_low = _normalize_text(mapped_norm)
    attr_low = _normalize_text(attr_norm)
    for canonical, variants in alias_map.items():
        canonical_low = _normalize_text(canonical)
        variant_lows = {_normalize_text(v.replace("_", " ")) for v in variants}
        if mapped_low in {canonical_low, *variant_lows}:
            mapped_tokens.update(_tokenize(canonical_low))
            for v in variant_lows:
                mapped_tokens.update(_tokenize(v))
        if attr_low in {canonical_low, *variant_lows}:
            attr_tokens.update(_tokenize(canonical_low))
            for v in variant_lows:
                attr_tokens.update(_tokenize(v))
    if not mapped_tokens or not attr_tokens:
        return False
    return len(mapped_tokens.intersection(attr_tokens)) >= 1


def semantic_match(response: str, attribute: str) -> Dict[str, Any]:
    response_tokens = _tokenize(response)
    response_text = " ".join(response_tokens)
    attribute_text = " ".join(_tokenize(attribute))
    if not response_text or not attribute_text:
        return {"similarity": 0.0, "semantic_class": "rejected", "boosted": False}

    boosted = False
    semantic_word_detected = ""
    phrase_signal_detected = ""
    expanded_response_tokens = list(response_tokens)
    for token in response_tokens:
        mapped = str(SEMANTIC_MAP.get(token, "")).strip()
        if mapped and _semantic_target_matches_attribute(mapped, attribute):
            expanded_response_tokens.extend(_tokenize(mapped))
            boosted = True
            if not semantic_word_detected:
                semantic_word_detected = token
    # soporte para frases multi-token en SEMANTIC_MAP
    if response_text:
        for key, mapped_val in SEMANTIC_MAP.items():
            key_norm = " ".join(_tokenize(str(key)))
            if not key_norm or " " not in key_norm:
                continue
            if key_norm in response_text and _semantic_target_matches_attribute(str(mapped_val), attribute):
                expanded_response_tokens.extend(_tokenize(str(mapped_val)))
                boosted = True
                if not semantic_word_detected:
                    semantic_word_detected = key_norm

    # senales naturales de universidad: permiten accepted cuando el concepto esta bien alineado.
    has_university_phrase_signal = False
    if response_text:
        for phrase, mapped_target in UNIVERSIDAD_NATURAL_SIGNAL_MAP.items():
            phrase_norm = " ".join(_tokenize(phrase))
            if not phrase_norm:
                continue
            if phrase_norm in response_text and _semantic_target_matches_attribute(mapped_target, attribute):
                has_university_phrase_signal = True
                expanded_response_tokens.extend(_tokenize(mapped_target))
                if not phrase_signal_detected:
                    phrase_signal_detected = phrase_norm
    expanded_response_text = " ".join(expanded_response_tokens)

    # regla solicitada: similarity = SequenceMatcher(None, response, attribute).ratio()
    similarity = SequenceMatcher(None, response_text, attribute_text).ratio()
    similarity = max(
        similarity,
        SequenceMatcher(None, expanded_response_text, attribute_text).ratio(),
        SequenceMatcher(None, response_text, " ".join(_tokenize(alias_natural_atributo(attribute, {})))).ratio(),
    )
    if boosted:
        similarity += 0.15

    # Si respuesta corta (<=5) tiene concepto del atributo o alias, suma bonus.
    attr_tokens = set(_tokenize(attribute)).union(set(_tokenize(alias_natural_atributo(attribute, {}))))
    short_has_concept = len(response_tokens) <= 5 and len(set(response_tokens).intersection(attr_tokens)) >= 1
    if not short_has_concept and semantic_word_detected and len(response_tokens) <= 5:
        short_has_concept = True
    if not short_has_concept and has_university_phrase_signal and len(response_tokens) <= 5:
        short_has_concept = True
    if short_has_concept:
        similarity += 0.10

    if has_university_phrase_signal:
        similarity += 0.10

    # Penalizacion por longitud reducida y acotada.
    length_penalty = min(0.05, max(0.0, (3 - len(response_tokens)) * 0.02))
    similarity -= length_penalty

    if has_university_phrase_signal:
        similarity = max(similarity, 0.46)
    similarity = max(0.0, min(1.0, similarity))
    if similarity >= 0.45:
        semantic_class = "accepted"
    elif similarity >= 0.25:
        semantic_class = "review"
    else:
        semantic_class = "rejected"
    return {
        "similarity": round(similarity, 3),
        "semantic_class": semantic_class,
        "boosted": boosted,
        "debug_semantic_match": bool(boosted),
        "semantic_word_detected": semantic_word_detected,
        "phrase_signal_detected": phrase_signal_detected,
        "has_university_phrase_signal": bool(has_university_phrase_signal),
        "length_penalty": round(length_penalty, 3),
    }


def semantic_attribute_match(response: str, attributes: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for attr in attributes if isinstance(attributes, list) else []:
        attr_name = str(attr or "").strip()
        if not attr_name:
            continue
        sem = semantic_match(response, attr_name)
        out.append(
            {
                "atributo": attr_name,
                "similaridad": float(sem.get("similarity", 0.0)),
                "semantic_class": str(sem.get("semantic_class", "rejected")),
                "boosted": bool(sem.get("boosted", False)),
                "debug_semantic_match": bool(sem.get("debug_semantic_match", False)),
                "semantic_word_detected": str(sem.get("semantic_word_detected", "")),
                "phrase_signal_detected": str(sem.get("phrase_signal_detected", "")),
                "has_university_phrase_signal": bool(sem.get("has_university_phrase_signal", False)),
            }
        )
    out.sort(key=lambda x: float(x.get("similaridad", 0.0)), reverse=True)
    return out


def match_attribute_semantic(respuesta: str, atributos: List[str]) -> List[Dict[str, Any]]:
    # compatibilidad con llamadas existentes
    return semantic_attribute_match(respuesta, atributos)


def _short_evidence(text: str, max_words: int = 8) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    return " ".join(words[:max_words])


def is_weak_evidence(evidencia: str) -> bool:
    low = _normalize_text(evidencia)
    if not low:
        return True
    short_valid_signals = {
        "profesores",
        "nivel educativo",
        "prestigio",
        "campus",
        "costo",
        "ambiente",
        "laboratorios",
        "bolsa de trabajo",
    }
    if low in short_valid_signals:
        return False
    min_valid_patterns = [
        "buenos profesores",
        "buenos docentes",
        "buena reputacion",
        "instalaciones modernas",
        "se aprenda bien",
        "se vea reconocida",
        "tenga buenas instalaciones",
        "el costo sea justo",
        "que tenga buen ambiente",
        "buen ambiente",
        "vale lo que cuesta",
        "que ayude a conseguir trabajo",
        "buena bolsa de trabajo",
        "salida laboral",
    ]
    if any(p in low for p in min_valid_patterns):
        return False
    concept_tokens = {"profesores", "reputacion", "instalaciones", "costo", "ambiente", "trabajo", "clases", "practicas", "precio"}
    tok_set = set(_tokenize(low))
    if len(tok_set) >= 2 and len(tok_set.intersection(concept_tokens)) >= 1:
        return False
    if len(_tokenize(low)) >= 2 and re.search(r"\b(se|tenga|tienen|tiene|ayude|aprenda|vea|sea)\b", low):
        if len(tok_set.intersection(concept_tokens)) >= 1:
            return False
    if any(low == m or low.startswith(f"{m} ") for m in WEAK_EVIDENCE_MARKERS):
        return True
    toks = _tokenize(low)
    if len(toks) <= 1:
        return True
    weak_tokens = {"cosa", "tema", "eso", "algo", "bien", "padre"}
    if len([t for t in toks if t not in weak_tokens]) <= 1:
        return True
    return False


def is_attribute_aligned_evidence(evidencia: str, attr: str, brief: Optional[Dict[str, Any]] = None) -> bool:
    ev_toks = set(_tokenize(evidencia))
    if not ev_toks:
        return False
    attr_toks = set(_tokenize(attr))
    alias_toks = set(_tokenize(alias_natural_atributo(attr, brief)))
    nat_toks: set[str] = set()
    for p in _natural_phrases_for_attr(attr, brief):
        nat_toks.update(_tokenize(p))
    if ev_toks.intersection(attr_toks):
        return True
    if ev_toks.intersection(alias_toks):
        return True
    if len(ev_toks.intersection(nat_toks)) >= 1:
        return True
    ev_low = _normalize_text(evidencia)
    attr_low = _normalize_text(attr)
    cat = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
    if cat == "universidad":
        soft_patterns = [
            ("calidad academica", ["aprenda bien", "clases buenas", "profes expliquen"]),
            ("prestigio", ["se vea reconocida", "tiene nombre", "me da confianza"]),
            ("reputacion", ["se vea reconocida", "tiene nombre", "me da confianza"]),
            ("infraestructura", ["buenas instalaciones", "espacios esten bien"]),
            ("recursos", ["buenas instalaciones", "buenos recursos"]),
            ("costo", ["costo sea justo", "no sea tan cara", "parece cara"]),
            ("accesibilidad", ["costo sea justo", "pueda pagarla"]),
            ("empleabilidad", ["buenas practicas", "consiga trabajo", "abra oportunidades"]),
            ("ambiente", ["me sienta comodo", "ambiente sea bueno"]),
            ("experiencia estudiantil", ["me sienta comodo", "ambiente sea bueno"]),
        ]
        for key, pats in soft_patterns:
            if key in attr_low and any(p in ev_low for p in pats):
                return True
        short_signal_by_attr = [
            ("calidad academica", {"profesores", "nivel educativo"}),
            ("prestigio", {"prestigio"}),
            ("reputacion", {"prestigio"}),
            ("infraestructura", {"campus", "laboratorios"}),
            ("costo", {"costo"}),
            ("empleabilidad", {"bolsa de trabajo", "salida laboral"}),
            ("ambiente", {"ambiente"}),
            ("experiencia estudiantil", {"ambiente"}),
        ]
        for key, signals in short_signal_by_attr:
            if key in attr_low and ev_low in signals:
                return True
    return False


def _infer_focus_from_question(question: str, brief: Dict[str, Any]) -> str:
    q_tokens = set(_tokenize(question))
    best_attr = ""
    best_score = 0
    for attr in _allowed_attributes(brief):
        tokens = set(_tokenize(attr)).union(set(_tokenize(alias_natural_atributo(attr, brief))))
        score = len(q_tokens.intersection(tokens))
        if score > best_score:
            best_score = score
            best_attr = attr
    return best_attr


def _is_generic_response(text: str) -> bool:
    low = _normalize_text(text)
    if not low:
        return True
    generic_markers = [
        "si eso falla",
        "no me late",
        "me gusta cuando",
        "pues sepa bien",
        "pues sepa rico",
        "esta bien",
        "normal",
    ]
    return any(m in low for m in generic_markers) or is_weak_evidence(text)


def _confusion_level(text: str) -> str:
    low = _normalize_text(text)
    if any(m in low for m in CONFUSION_MARKERS_HARD):
        return "hard"
    if any(m in low for m in CONFUSION_MARKERS_SOFT):
        return "soft"
    return "none"


def _extract_concept(answer: str, foco: str, lexicon: Dict[str, Dict[str, List[str]]]) -> str:
    if foco:
        return foco
    tokens = _tokenize(answer)
    if not tokens:
        return "este punto"
    known = set()
    for item in lexicon.values():
        known.update(item.get("keywords_pos", []))
    picked = [t for t in tokens if t in known][:3]
    if not picked:
        picked = tokens[:3]
    return " ".join(picked)[:40]


def obtener_estilo_moderacion(target_participante: Dict[str, Any]) -> Dict[str, Any]:
    target = target_participante if isinstance(target_participante, dict) else {}
    rango = str(target.get("rango_edad", "")).strip()
    conocimiento = str(target.get("nivel_conocimiento", "medio")).strip().lower()
    profundizacion = str(target.get("nivel_profundizacion", "media")).strip().lower()
    perfil = str(target.get("perfil", "consumidor_general")).strip().lower()

    lenguaje = "medio"
    max_palabras = 16
    usar_comparaciones_simples = True
    evitar_abstracciones = False
    tono = "claro y conversacional"

    if rango in {"13-15", "16-18"} or conocimiento == "bajo" or profundizacion == "baja":
        lenguaje = "simple"
        max_palabras = 14
        usar_comparaciones_simples = True
        evitar_abstracciones = True
        tono = "cotidiano"
    elif perfil == "usuario_experto" and conocimiento == "alto" and profundizacion == "alta":
        lenguaje = "avanzado"
        max_palabras = 20
        usar_comparaciones_simples = False
        evitar_abstracciones = False
        tono = "detallado pero claro"

    return {
        "lenguaje": lenguaje,
        "max_palabras_pregunta": max_palabras,
        "usar_comparaciones_simples": usar_comparaciones_simples,
        "evitar_abstracciones": evitar_abstracciones,
        "tono": tono,
    }


def _estilo_from_brief(brief: Dict[str, Any]) -> Dict[str, Any]:
    return obtener_estilo_moderacion(brief.get("target_participante", {}) if isinstance(brief, dict) else {})


def simplificar_pregunta(texto: str, estilo: Optional[Dict[str, Any]] = None) -> str:
    t = " ".join((texto or "").strip().split())
    if not t:
        return t
    estilo_obj = estilo if isinstance(estilo, dict) else {}
    max_words = int(estilo_obj.get("max_palabras_pregunta", 16) or 16)
    replacements = {
        "percepcion": "opinion",
        "determinante": "importante",
        "decision": "eleccion",
        "atributo": "cosa",
        "diferenciador": "diferencia",
        "evaluacion": "opinion",
        "posicionamiento": "tema",
        "variable": "tema",
        "factor": "tema",
        "elemento": "cosa",
        "dimension": "tema",
        "consideras": "piensas",
        "de lo conversado": "",
        "para confirmar": "para cerrar",
        "cual aspecto": "que cosa",
        "mas importante": "mas importante",
        "influye": "importa",
        "detallar": "explicar",
    }
    low = t.lower()
    for k, v in replacements.items():
        low = low.replace(k, v)
    # Mantener una sola idea: recorta a primera oracion si hay multiples.
    low = re.split(r"[!?\.]", low)[0].strip()
    # Quitar conectores formales que hacen sonar tecnico.
    low = re.sub(r"\b(en este punto|de forma general|en terminos generales)\b", "", low).strip()
    low = re.sub(r"\b(podrias|quisiera|me gustaria)\b", "", low).strip()
    low = " ".join(low.split())
    low = low.replace(" de el ", " del ").replace(" de la la ", " de la ")
    words = low.split()
    if "que tan " in low and max_words < 15:
        max_words = 15
    if len(words) > max_words:
        low = " ".join(words[:max_words])
    if not low.endswith("?"):
        low = low.rstrip(",;:") + "?"
    out = low[:140]
    return out[:1].upper() + out[1:] if out else out


def sanitizar_pregunta_por_categoria(pregunta: str, brief: Dict[str, Any]) -> str:
    q = " ".join(str(pregunta or "").split()).strip()
    if not q:
        return q
    cat = detectar_objeto_estudio(brief).get("categoria", "producto_generico")
    if cat == "universidad":
        q = re.sub(r"\bsoda(s)?\b|\brefresco(s)?\b|\bbebida(s)?\b", "universidad", q, flags=re.IGNORECASE)
        q = re.sub(r"cuando la tomas", "cuando la eliges", q, flags=re.IGNORECASE)
    elif cat == "soda":
        q = re.sub(r"\buniversidad(es)?\b", "soda", q, flags=re.IGNORECASE)
        q = re.sub(r"cuando la eliges", "cuando la tomas", q, flags=re.IGNORECASE)
    elif cat == "auto":
        q = re.sub(r"\bsoda(s)?\b|\brefresco(s)?\b|\bbebida(s)?\b", "auto", q, flags=re.IGNORECASE)
        q = re.sub(r"\buniversidad(es)?\b", "auto", q, flags=re.IGNORECASE)
        q = re.sub(r"cuando la tomas|cuando la eliges", "cuando lo manejas", q, flags=re.IGNORECASE)
    q = q.replace("de el", "del").replace("a el", "al")
    q = re.sub(r"\bque tan dulce es\s+que tan dulce es\b", "que tan dulce es", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def postprocesar_pregunta_final(pregunta: str, brief: Dict[str, Any]) -> str:
    q = sanitizar_pregunta_por_categoria(pregunta, brief)
    fixes = {
        "laliges": "la eliges",
        "notal": "notas",
        "importal": "importa el",
        "importanl": "importan las",
        "que hace que las instalaciones pese": "que hace que las instalaciones pesen",
        "de el ": "del ",
        "a el ": "al ",
    }
    low = q.lower()
    for bad, good in fixes.items():
        low = low.replace(bad, good)
    low = re.sub(r"\bimportal\s+las\b", "importan las", low)
    low = re.sub(r"\bimportal\s+el\b", "importa el", low)
    low = re.sub(r"\bimportal\b", "importa", low)
    low = re.sub(r"\bpese para ti\b", "pesen mas para ti", low)
    low = re.sub(r"\bpesen para ti\b", "pesen mas para ti", low)
    low = re.sub(r"\bque hace que\s+que\b", "que hace que", low)
    low = re.sub(r"\bte viene a\?\s*$", "se te viene a la mente?", low)
    low = re.sub(r"\bla\?\s*$", "la?", low)
    low = re.sub(r"\?{2,}", "?", low)
    low = re.sub(r"\s+", " ", low).strip()
    if low and not low.endswith("?"):
        low = low.rstrip(",;:") + "?"
    return low[:1].upper() + low[1:] if low else low


def postprocesar_pregunta(pregunta: str, brief: Dict[str, Any]) -> str:
    return postprocesar_pregunta_final(pregunta, brief)


def normalizar_pregunta(pregunta: str, brief: Dict[str, Any], estilo: Optional[Dict[str, Any]] = None) -> str:
    estilo_obj = estilo if isinstance(estilo, dict) else _estilo_from_brief(brief or {})
    q = simplificar_pregunta(str(pregunta or ""), estilo=estilo_obj)
    q = postprocesar_pregunta_final(q, brief or {})
    low = _normalize_text(q)
    if "que atributo es mas importante" in low:
        cat = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
        if cat == "universidad":
            q = "Que cosas hacen que una universidad te parezca buena?"
    return q


def generar_pregunta_profundizacion(atributo: str, brief: Optional[Dict[str, Any]] = None) -> str:
    attr = str(atributo or "").strip()
    if not attr:
        return "Que significa eso para ti?"
    cat = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
    low = _normalize_text(attr)
    if cat == "universidad":
        if any(x in low for x in ["calidad academica", "academica", "clases"]):
            return "Que hace que sientas que una universidad ensena bien?"
        if any(x in low for x in ["prestigio", "reputacion", "nombre"]):
            return "Como notas que una universidad tiene buena reputacion?"
        if any(x in low for x in ["infraestructura", "instalaciones", "recursos"]):
            return "Que cosas de las instalaciones te hacen pensar que es buena?"
        if any(x in low for x in ["costo", "accesibilidad", "colegiatura", "precio"]):
            return "Que te hace sentir que una universidad vale lo que cuesta?"
        if any(x in low for x in ["empleabilidad", "oportunidades laborales", "trabajo"]):
            return "Que te hace pensar que te ayudara a conseguir trabajo?"
        if any(x in low for x in ["ambiente", "experiencia estudiantil"]):
            return "Que te hace sentir que el ambiente universitario es bueno?"
    alias = alias_natural_atributo(attr, brief)
    return f"Como notas {alias} cuando comparas opciones?"


def generate_question_by_step_type(step_type: str, attribute: str, context: Dict[str, Any]) -> List[str]:
    brief = context.get("brief", {}) if isinstance(context, dict) else {}
    foco = str(attribute or "").strip()
    foco_alias = alias_natural_atributo(foco, brief) if foco else "esto"
    obj = detectar_objeto_estudio(brief)
    sing = obj.get("sustantivo_singular", "opcion")
    plur = obj.get("sustantivo_plural", "opciones")
    step = str(step_type or "").strip().lower()
    if step == "explorar":
        return [
            f"Que cosas te hacen pensar que una {sing} es buena?",
            f"Cuando comparas {plur}, que notas primero?",
            "Que te llama mas la atencion al inicio?",
        ]
    if step == "profundizar":
        if obj.get("categoria") == "universidad":
            return [
                "Como te das cuenta de que una universidad es buena?",
                "Que ves o escuchas para pensar que tiene buen nivel?",
                "Que te haria confiar mas en una universidad?",
                "Que te hace sentir que vale lo que cuesta?",
                generar_pregunta_profundizacion(foco_alias, brief),
            ]
        return [
            f"Como se nota {foco_alias} en una {sing}?",
            f"Que significa para ti {foco_alias}?",
            f"Puedes dar un ejemplo de {foco_alias}?",
            f"Por que es importante {foco_alias} para ti?",
            f"A que te refieres con eso de {foco_alias}?",
            generar_pregunta_profundizacion(foco_alias, brief),
        ]
    if step == "comparar":
        return [
            f"Eso pesa mas para ti que {foco_alias}?",
            f"Si dos {plur} fueran iguales, eso marcaria diferencia?",
            "Que pesa mas para ti al comparar?",
        ]
    if step == "reconducir":
        return [
            f"Eso que dices es interesante. Pensando en elegir {sing}, que pesa mas para ti?",
            "Y de eso, que te ayuda a decidir?",
            f"Si lo llevamos a la {sing}, que seria lo importante?",
        ]
    if step == "cerrar":
        return ["Si tuvieras que resumirlo en tres cosas, cuales serian?"]
    return []


def _select_pending_focus(state: Dict[str, Any], pending_attrs: List[str]) -> str:
    if not pending_attrs:
        return ""
    profundidad = state.get("_profundidad_por_atributo", {})
    if not isinstance(profundidad, dict):
        profundidad = {}
    detectados = state.get("atributos_detectados_unicos", [])
    if not isinstance(detectados, list):
        detectados = []
    for attr in detectados:
        if attr in pending_attrs and atributo_necesita_profundizacion(state, attr):
            return attr
    sorted_pending = sorted(pending_attrs, key=lambda a: int(profundidad.get(a, 0)))
    last_focus = str(state.get("foco_actual", "")).strip()
    for attr in sorted_pending:
        if attr != last_focus:
            return attr
    return sorted_pending[0]


def atributo_necesita_profundizacion(state: Dict[str, Any], attr: str) -> bool:
    profundidad = state.get("_profundidad_por_atributo", {})
    if not isinstance(profundidad, dict):
        return True
    nivel = int(profundidad.get(attr, 0))
    return nivel < 2


def _item_is_confirmed(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    cuenta = bool(item.get("cuenta", False))
    conf = str(item.get("confianza", "")).strip().lower()
    ev = str(item.get("evidencia", "")).strip()
    if not ev:
        return False
    return cuenta and conf in {"media", "alta"}


def _is_confused_response(text: str) -> bool:
    return _confusion_level(text) == "hard"


def _has_explanation(answer: str) -> bool:
    low = _normalize_text(answer)
    reason_markers = ["porque", "ya que", "debido", "influye", "impacta", "por eso"]
    return len(_tokenize(answer)) >= 8 or any(m in low for m in reason_markers)


def _has_example(answer: str) -> bool:
    low = _normalize_text(answer)
    example_markers = ["por ejemplo", "ejemplo", "una vez", "cuando", "caso", "me paso", "experiencia"]
    return any(m in low for m in example_markers)


def generar_respuesta_simulada(pregunta: str, target: Dict[str, Any], brief: Dict[str, Any]) -> str:
    target = target if isinstance(target, dict) else {}
    nivel = str(target.get("nivel_profundizacion", "baja")).strip().lower()
    conocimiento = str(target.get("nivel_conocimiento", "bajo")).strip().lower()
    rango_edad = str(target.get("rango_edad", "")).strip().lower()
    attrs = _allowed_attributes(brief)
    foco = attrs[0] if attrs else "este tema"
    qlow = _normalize_text(pregunta)
    for attr in attrs:
        attr_norm = _normalize_text(attr)
        alias_norm = _normalize_text(alias_natural_atributo(attr, brief))
        if attr_norm in qlow or (alias_norm and alias_norm in qlow):
            foco = attr
            break

    frases = _natural_phrases_for_attr(foco, brief)
    frase = random.choice(frases)

    frase_sin_prefijo = frase[4:] if frase.startswith("que ") else frase
    if nivel == "baja":
        base = [
            f"porque {frase_sin_prefijo}",
            f"{frase_sin_prefijo}",
            f"porque {frase_sin_prefijo}",
            "se siente mejor",
            "me cae mejor",
        ]
    elif nivel == "media":
        base = [
            f"porque {frase_sin_prefijo}",
            f"a mi me importa que {frase_sin_prefijo}",
            random.choice(
                [
                    "si falla eso, ya no me convence",
                    "si falla eso, siento que pierde algo",
                    "si no cumple eso, ya no se me antoja igual",
                ]
            ),
        ]
    else:
        base = [
            f"Para mi pesa mucho {frase}, porque cambia la experiencia al tomarla.",
            f"Yo noto mucho {frase}; si falla en eso, la dejo de comprar.",
            f"Cuando comparo opciones, me quedo con la que cumple mejor en eso.",
        ]

    patterns = brief.setdefault("_sim_response_pattern_counts", {}) if isinstance(brief, dict) else {}
    if not isinstance(patterns, dict):
        patterns = {}
    scored = sorted(base, key=lambda x: int(patterns.get(_normalize_text(x), 0)))
    pick_pool = scored[: max(1, min(3, len(scored)))]
    resp = random.choice(pick_pool)
    sig = _normalize_text(resp)
    patterns[sig] = int(patterns.get(sig, 0)) + 1
    if isinstance(brief, dict):
        brief["_sim_response_pattern_counts"] = patterns
    if conocimiento == "alto" and nivel != "baja":
        resp += " Tambien la comparo con otras marcas."
    if any(x in rango_edad for x in ["13-15", "16-18"]):
        youth = [
            f"porque {frase_sin_prefijo}",
            f"{frase_sin_prefijo}",
            f"porque {frase_sin_prefijo}",
            "no me convence",
            "siento que pierde algo",
        ]
        youth_scored = sorted(youth, key=lambda x: int(patterns.get(_normalize_text(x), 0)))
        ypool = youth_scored[: max(1, min(3, len(youth_scored)))]
        resp = random.choice(ypool if nivel == "baja" else ypool + [resp])
        sig2 = _normalize_text(resp)
        patterns[sig2] = int(patterns.get(sig2, 0)) + 1
        if isinstance(brief, dict):
            brief["_sim_response_pattern_counts"] = patterns
    return resp


def _allowed_attributes(brief: Dict[str, Any]) -> List[str]:
    attrs = brief.get("atributos_objetivo", {}).get("lista", [])
    return [str(x).strip() for x in attrs if str(x).strip()]


def build_project_lexicon(brief: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    attrs = _allowed_attributes(brief)
    defs = brief.get("definiciones_operativas", {})
    if not isinstance(defs, dict):
        defs = {}

    out: Dict[str, Dict[str, List[str]]] = {}
    for attr in attrs:
        item = defs.get(attr, {}) if isinstance(defs.get(attr, {}), dict) else {}
        que_cuenta = str(item.get("que_cuenta", ""))
        que_no_cuenta = str(item.get("que_no_cuenta", ""))

        pos = set(_tokenize(attr))
        pos.update(_tokenize(que_cuenta))
        pos.update(_tokenize(alias_natural_atributo(attr, brief)))
        for phrase in _natural_phrases_for_attr(attr, brief):
            pos.update(_tokenize(phrase))
        neg = set(_tokenize(que_no_cuenta))

        out[attr] = {
            "keywords_pos": sorted(pos),
            "keywords_neg": sorted(neg),
        }
    return out


def detect_candidates_lexicon(answer: str, lexicon: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    ans_tokens = set(_tokenize(answer))
    candidates: List[Dict[str, Any]] = []
    semantic_by_attr = {
        str(m.get("atributo", "")).strip(): {
            "similaridad": float(m.get("similaridad", 0.0)),
            "semantic_class": str(m.get("semantic_class", "rejected")).strip().lower(),
            "boosted": bool(m.get("boosted", False)),
        }
        for m in semantic_attribute_match(answer, list(lexicon.keys()))
        if isinstance(m, dict) and str(m.get("atributo", "")).strip()
    }

    for attr, kw in lexicon.items():
        pos = set(kw.get("keywords_pos", []))
        if not pos:
            continue
        hits = sorted(ans_tokens.intersection(pos))
        hit_count = len(hits)
        pos_size = len(pos)
        attr_tokens = set(_tokenize(attr))
        direct_attr_hit = len(ans_tokens.intersection(attr_tokens)) > 0
        min_hit = 1 if (pos_size <= 6 or direct_attr_hit) else 2
        sem_obj = semantic_by_attr.get(attr, {})
        semantic_sim = float(sem_obj.get("similaridad", 0.0))
        semantic_class = str(sem_obj.get("semantic_class", "rejected")).lower()
        semantic_ok = semantic_class in {"accepted", "review"}
        if semantic_ok:
            score = int(round(semantic_sim * 4)) + hit_count + (1 if direct_attr_hit else 0)
            if semantic_class == "accepted":
                score += 1
            if bool(sem_obj.get("boosted", False)):
                score += 1
            candidates.append(
                {
                    "atributo": attr,
                    "score": score,
                    "hits_pos": hits,
                    "direct_attr_hit": direct_attr_hit,
                    "semantic_similarity": round(semantic_sim, 3),
                    "semantic_class": semantic_class,
                }
            )

    candidates.sort(key=lambda x: int(x.get("score", 0)), reverse=True)
    return candidates


def detect_attribute_candidates(answer: str, lexicon: Dict[str, Dict[str, List[str]]]) -> List[Dict[str, Any]]:
    return detect_candidates_lexicon(answer, lexicon)


def es_respuesta_vacia(texto: str) -> bool:
    low = _normalize_text(texto)
    if not low:
        return True
    return any(m in low for m in EMPTY_MARKERS)


def _is_vague_answer(text: str) -> bool:
    low = _normalize_text(text)
    token_count = len(_tokenize(low))
    return any(m in low for m in VAGUE_MARKERS) and token_count < 8


def _contains_vague_adjective(text: str) -> bool:
    low = _normalize_text(text)
    vague_adjs = {"buena", "bueno", "interesante", "completa", "completo"}
    toks = set(_tokenize(low))
    return len(toks.intersection(vague_adjs)) >= 1


def es_respuesta_muy_breve_o_vaga(respuesta: str) -> bool:
    low = _normalize_text(respuesta)
    tokens = _tokenize(respuesta)
    if len(tokens) <= 2:
        return True
    base_markers = [
        "si",
        "no",
        "normal",
        "depende",
        "no se",
        "no estoy seguro",
        "pues",
        "algo asi",
        "esta bien",
        "me gusta",
        "esta padre",
    ]
    if low in base_markers:
        return True
    if any(m in low for m in base_markers) and len(tokens) < 8:
        return True
    return _is_vague_answer(respuesta)


def _is_out_of_focus_response(respuesta: str, foco: str, brief: Dict[str, Any], attrs_detected: List[str]) -> bool:
    if not foco or attrs_detected:
        return False
    if _confusion_level(respuesta) != "none":
        return False
    if es_respuesta_muy_breve_o_vaga(respuesta):
        return False
    resp_tokens = set(_tokenize(respuesta))
    if len(resp_tokens) < 6:
        return False
    foco_tokens = set(_tokenize(foco)).union(set(_tokenize(alias_natural_atributo(foco, brief))))
    return len(resp_tokens.intersection(foco_tokens)) == 0


def validate_counts(
    answer: str,
    candidates: List[Dict[str, Any]],
    lexicon: Dict[str, Dict[str, List[str]]],
    question: str = "",
    brief: Optional[Dict[str, Any]] = None,
    foco_actual: str = "",
) -> List[Dict[str, Any]]:
    confusion_level = _confusion_level(answer)
    if confusion_level == "hard":
        return []
    ans_tokens = set(_tokenize(answer))
    vague = _is_vague_answer(answer)
    empty = es_respuesta_vacia(answer)
    has_expl = _has_explanation(answer)
    has_ex = _has_example(answer)
    items: List[Dict[str, Any]] = []

    answer_tokens_list = _tokenize(answer)
    token_count = len(answer_tokens_list)
    short_without_support = token_count < 6 and (not has_expl) and (not has_ex)
    question_tokens = set(_tokenize(question))
    focus_from_question = foco_actual or _infer_focus_from_question(question, brief or {})
    categoria = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
    producto_categoria = categoria in {"soda", "auto"}
    universidad_categoria = categoria == "universidad"
    for cand in candidates:
        attr = str(cand.get("atributo", ""))
        if not attr:
            continue
        score = int(cand.get("score", 0))
        semantic_class = str(cand.get("semantic_class", "rejected")).strip().lower()
        semantic_accepted = semantic_class == "accepted"
        semantic_signal = bool(cand.get("has_university_phrase_signal", False)) or bool(cand.get("boosted", False))
        semantic_source = str(cand.get("phrase_signal_detected", "") or cand.get("semantic_word_detected", "")).strip()
        hits_pos = cand.get("hits_pos", []) if isinstance(cand.get("hits_pos", []), list) else []
        neg = set(lexicon.get(attr, {}).get("keywords_neg", []))
        neg_hits = sorted(ans_tokens.intersection(neg))
        neg_count = len(neg_hits)

        detectado = (not empty) and (score > 0 or (universidad_categoria and semantic_accepted and semantic_signal))
        ev_candidate = _short_evidence(answer, max_words=8) if (has_ex or has_expl) else (" ".join(hits_pos[:3]) or _short_evidence(answer, max_words=8))
        aligned_ev = is_attribute_aligned_evidence(ev_candidate, attr, brief)
        weak_ev = is_weak_evidence(ev_candidate)
        if aligned_ev:
            weak_ev = False
        short_response_promoted = False
        if short_without_support:
            confirmado = bool(detectado and aligned_ev and (not weak_ev) and (not vague))
            evidencia = bool(confirmado and aligned_ev and (not weak_ev))
            if producto_categoria and semantic_accepted and detectado:
                confirmado = True
                # en producto permitimos evidencia breve si semantic accepted
                evidencia = bool(ev_candidate.strip())
                weak_ev = False
            if universidad_categoria and semantic_accepted and (detectado or aligned_ev):
                confirmado = True
                evidencia = bool(ev_candidate.strip()) and (aligned_ev or semantic_signal)
                weak_ev = False
                short_response_promoted = True
        else:
            confirmado = detectado and (has_expl or has_ex or aligned_ev)
            evidencia = confirmado and (has_ex or has_expl or aligned_ev) and (not weak_ev)
        if vague and not has_expl:
            if not aligned_ev:
                confirmado = False
                evidencia = False
        if neg_count >= max(2, score):
            detectado = False
            confirmado = False
            evidencia = False
        if focus_from_question and token_count < 12 and attr != focus_from_question:
            ev_tokens = set(_tokenize(ev_candidate))
            attr_anchor_tokens = set(_tokenize(attr)).union(set(_tokenize(alias_natural_atributo(attr, brief))))
            off_focus_clear_anchor = len(ev_tokens.intersection(attr_anchor_tokens)) >= 2
            if not off_focus_clear_anchor:
                confirmado = False
                evidencia = False
                if detectado:
                    confianza = "baja"

        if evidencia:
            confianza = "alta"
        elif confirmado:
            confianza = "media"
        elif detectado:
            confianza = "baja"
        else:
            confianza = "baja"

        ev_text = ev_candidate if (confirmado or detectado) else ""
        if not ev_text and detectado:
            ev_text = " ".join(hits_pos[:2])

        weak_ev = is_weak_evidence(ev_text)
        if is_attribute_aligned_evidence(ev_text, attr, brief):
            weak_ev = False
        attr_tokens = set(_tokenize(attr))
        anchored_by_question = len(question_tokens.intersection(attr_tokens)) > 0
        if weak_ev and not anchored_by_question and not (producto_categoria and semantic_accepted):
            confirmado = False
            evidencia = False
            if confianza in {"media", "alta"}:
                confianza = "baja"
            if detectado and not ev_text:
                ev_text = " ".join(hits_pos[:1])

        profundidad_minima = bool(detectado and not empty and confusion_level != "hard" and (is_attribute_aligned_evidence(ev_text, attr, brief) or has_expl or has_ex))
        profundidad_fuerte = bool(not weak_ev and (has_expl or has_ex))
        accepted_reason = "rejected_sin_senal"
        if detectado and not confirmado:
            accepted_reason = "review_por_senal_parcial"
        if confirmado and not evidencia:
            accepted_reason = "accepted_por_senal_semantica"
        if confirmado and evidencia:
            accepted_reason = "accepted_por_evidencia_util"
        if short_response_promoted:
            accepted_reason = "accepted_por_respuesta_corta_semantica"

        items.append(
            {
                "atributo": attr,
                "_score": score,
                "detectado": detectado,
                "confirmado": confirmado,
                "evidencia_nivel": evidencia,
                "cuenta": detectado,
                "evidencia": ev_text,
                "confianza": confianza,
                "weak_evidence": weak_ev,
                "semantic_class": semantic_class,
                "semantic_signal": semantic_signal,
                "semantic_source": semantic_source,
                "semantic_hit": bool(semantic_signal or semantic_class in {"accepted", "review"}),
                "matched_attribute": attr,
                "accepted_reason": accepted_reason,
                "short_response_promoted": bool(short_response_promoted),
                "profundidad_minima": profundidad_minima,
                "profundidad_fuerte": profundidad_fuerte,
                "razon_corta": (
                    "Evidencia debil: se mantiene como mencion."
                    if weak_ev and not anchored_by_question
                    else "Nivel 1 detectado, nivel 2 confirmado, nivel 3 evidencia."
                ),
            }
        )

    # En respuestas cortas, maximo un atributo confirmado para evitar deteccion optimista.
    if token_count < 10:
        ranked = sorted(items, key=lambda x: int(x.get("_score", 0)), reverse=True)
        if ranked:
            def _rank_item(it: Dict[str, Any]) -> Any:
                ev = str(it.get("evidencia", ""))
                overlap = len(set(_tokenize(ev)).intersection(ans_tokens))
                return (overlap, int(it.get("_score", 0)), len(ev))
            ranked = sorted(ranked, key=_rank_item, reverse=True)
        top_attr = ranked[0].get("atributo") if ranked else ""
        for it in items:
            if it.get("atributo") != top_attr and bool(it.get("confirmado", False)):
                it["confirmado"] = False
                it["evidencia_nivel"] = False
                if it.get("confianza") in {"media", "alta"}:
                    it["confianza"] = "baja"
                it["razon_corta"] = "Respuesta corta: solo un atributo puede quedar confirmado."

    for it in items:
        it.pop("_score", None)
    return items


def _llm_validate_attribute_items(
    answer: str,
    brief: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    question: str = "",
    foco_actual: str = "",
) -> List[Dict[str, Any]]:
    client = _get_client()
    if client is None:
        return []

    attrs = _allowed_attributes(brief)
    allow_emergent = bool(brief.get("config", {}).get("permitir_atributos_emergentes", False))
    candidate_attrs = [str(c.get("atributo", "")).strip() for c in candidates if str(c.get("atributo", "")).strip()]
    payload = {
        "respuesta": answer,
        "atributos_objetivo": attrs,
        "candidatos_lexicon": candidates,
        "atributos_a_evaluar": sorted(set(candidate_attrs or attrs)),
        "permitir_atributos_emergentes": allow_emergent,
        "definiciones_operativas": brief.get("definiciones_operativas", {}),
    }
    system_prompt = (
        "Evalua atributos para moderacion cualitativa. "
        "Devuelve SOLO JSON valido del schema. "
        "Atributo debe estar en atributos_objetivo; no inventes. "
        "Considera valido lo espontaneo, comparaciones y asociaciones del participante. "
        "Marca cuenta=false solo si esta fuera de tema o es totalmente vacio."
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": ATTR_ITEMS_SCHEMA["name"],
                    "strict": ATTR_ITEMS_SCHEMA["strict"],
                    "schema": ATTR_ITEMS_SCHEMA["schema"],
                }
            },
        )
        data = json.loads(response.output_text or "{}")
        items = data.get("items", [])
        if not isinstance(items, list):
            return []
        answer_tokens_list = _tokenize(answer)
        answer_tokens = set(answer_tokens_list)
        token_count = len(answer_tokens_list)
        short_without_support = token_count < 6 and (not _has_explanation(answer)) and (not _has_example(answer))
        focus_from_question = foco_actual or _infer_focus_from_question(question, brief)
        categoria = detectar_objeto_estudio(brief or {}).get("categoria", "producto_generico")
        producto_categoria = categoria in {"soda", "auto"}
        semantic_class_by_attr = {
            str(c.get("atributo", "")).strip(): str(c.get("semantic_class", "rejected")).strip().lower()
            for c in (candidates if isinstance(candidates, list) else [])
            if isinstance(c, dict) and str(c.get("atributo", "")).strip()
        }
        allowed = set(attrs)
        clean: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            attr = str(it.get("atributo", "")).strip()
            if attr not in allowed:
                continue
            conf = str(it.get("confianza", "baja")).strip().lower()
            if conf not in {"baja", "media", "alta"}:
                conf = "baja"
            evidence = _short_evidence(str(it.get("evidencia", "")).strip()[:120], max_words=8)
            evidence_tokens = set(_tokenize(evidence))
            if evidence and len(answer_tokens.intersection(evidence_tokens)) < 1:
                evidence = ""
            cuenta = bool(it.get("cuenta", False))
            semantic_accepted = semantic_class_by_attr.get(attr, "rejected") == "accepted"
            weak_ev = is_weak_evidence(evidence)
            aligned_ev = is_attribute_aligned_evidence(evidence, attr, brief)
            if aligned_ev:
                weak_ev = False
            anch_text = len(answer_tokens.intersection(set(_tokenize(alias_natural_atributo(attr, brief))).union(set(_tokenize(attr))))) >= 1
            if short_without_support and not aligned_ev:
                cuenta = False
                if producto_categoria and semantic_accepted:
                    cuenta = True
            if not evidence or weak_ev or (not aligned_ev and not anch_text):
                cuenta = False
                if producto_categoria and semantic_accepted and (evidence or anch_text):
                    cuenta = True
                    weak_ev = False
            if focus_from_question and token_count < 12 and attr != focus_from_question:
                off_focus_anchor = len(answer_tokens.intersection(set(_tokenize(attr)).union(set(_tokenize(alias_natural_atributo(attr, brief)))))) >= 2
                if not off_focus_anchor:
                    cuenta = False
            clean.append(
                {
                    "atributo": attr,
                    "detectado": bool(cuenta),
                    "confirmado": bool(cuenta and conf in {"media", "alta"}),
                    "evidencia_nivel": bool(cuenta and conf == "alta"),
                    "cuenta": cuenta,
                    "evidencia": evidence,
                    "confianza": conf,
                    "weak_evidence": weak_ev,
                    "profundidad_minima": bool(cuenta and (aligned_ev or conf in {"media", "alta"})),
                    "profundidad_fuerte": bool(cuenta and not weak_ev and conf == "alta"),
                    "razon_corta": str(it.get("razon_corta", ""))[:160],
                }
            )
        if token_count < 10:
            confirmed = [x for x in clean if bool(x.get("confirmado", False))]
            if len(confirmed) > 1:
                confirmed_sorted = sorted(
                    confirmed,
                    key=lambda x: len(set(_tokenize(str(x.get("evidencia", "")))).intersection(answer_tokens)),
                    reverse=True,
                )
                keep = confirmed_sorted[0].get("atributo")
                for x in clean:
                    if x.get("atributo") != keep:
                        x["confirmado"] = False
                        x["evidencia_nivel"] = False
                        if x.get("confianza") in {"media", "alta"}:
                            x["confianza"] = "baja"
                        x["razon_corta"] = "Respuesta corta: confirmado limitado a un atributo."
        return clean
    except Exception:
        return []


def _critical_pendings(plan: Dict[str, Any]) -> bool:
    blueprint = plan.get("engine_blueprint", {})
    for v in blueprint.get("validaciones", []):
        if not isinstance(v, dict):
            continue
        regla = str(v.get("regla", "")).lower()
        estado = str(v.get("estado", "")).lower()
        criticidad = str(v.get("criticidad", "")).lower()
        if estado == "pendiente" and criticidad == "alta":
            if "target" in regla or "consistencia" in regla or "atributos_objetivo" in regla or "target_vs_lista" in regla:
                return True

    for p in plan.get("pendientes", []):
        campo = str(p.get("campo_faltante", "")).lower()
        if (
            "objetivo_principal" in campo
            or "antecedente" in campo
            or "max_preguntas" in campo
            or "target" in campo
            or "consistencia" in campo
            or "atributos_objetivo" in campo
        ):
            return True
    return False


def _blueprint(plan: Dict[str, Any]) -> Dict[str, Any]:
    return plan.get("engine_blueprint", {}) if isinstance(plan.get("engine_blueprint", {}), dict) else {}


def _blueprint_params(plan: Dict[str, Any]) -> Dict[str, Any]:
    bp = _blueprint(plan)
    p = bp.get("parametros", {})
    return p if isinstance(p, dict) else {}


def _decision_rules(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = _blueprint(plan).get("reglas_de_decision", [])
    if not isinstance(rules, list):
        return []
    cleaned = [r for r in rules if isinstance(r, dict)]
    return sorted(cleaned, key=lambda x: int(x.get("prioridad", 999)))


def _templates_by_action(plan: Dict[str, Any]) -> Dict[str, List[str]]:
    t = _blueprint(plan).get("templates_por_accion", {})
    if not isinstance(t, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for action, items in t.items():
        if isinstance(items, list):
            out[action] = [str(x).strip() for x in items if str(x).strip()]
    return out


def _blueprint_stages(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stages = _blueprint(plan).get("etapas", [])
    if not isinstance(stages, list):
        return {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for s in stages:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", s.get("id_etapa", ""))).strip()
        if sid:
            by_id[sid] = s
    return by_id


def _allowed_actions_for_stage(plan: Dict[str, Any], etapa_id: str) -> List[str]:
    st = _blueprint_stages(plan).get(etapa_id, {})
    acts = st.get("acciones_permitidas", [])
    if not isinstance(acts, list):
        return ACCIONES_VALIDAS
    cleaned = [str(a).strip() for a in acts if str(a).strip() in ACCIONES_VALIDAS]
    return cleaned or ACCIONES_VALIDAS


def _step_tipo_for_stage(plan: Dict[str, Any], etapa_id: str, etapa_index: int) -> str:
    client_plan = plan.get("client_plan", {}) if isinstance(plan, dict) else {}
    etapas = client_plan.get("etapas_visuales", []) if isinstance(client_plan, dict) else []
    if not isinstance(etapas, list):
        etapas = []
    target_paso = int(etapa_index) + 1
    for e in etapas:
        if not isinstance(e, dict):
            continue
        eid = str(e.get("id_etapa", "")).strip()
        paso = int(e.get("paso", 0) or 0)
        if eid == etapa_id or paso == target_paso:
            t = str(e.get("tipo", "")).strip().lower()
            if t in {"explorar", "profundizar", "reconducir", "comparar", "cerrar"}:
                return t
    fallback = {
        "apertura_y_encuadre": "explorar",
        "exploracion_espontanea": "explorar",
        "probing_por_atributos_objetivo": "profundizar",
        "evidencia_y_ejemplos": "profundizar",
        "priorizacion_y_contraste": "comparar",
        "confirmacion": "reconducir",
        "cierre": "cerrar",
    }
    return fallback.get(etapa_id, "explorar")


def _next_uncovered_attribute(state: Dict[str, Any], allowed_attrs: List[str]) -> str:
    uncovered = [a for a in allowed_attrs if a not in state.get("atributos_detectados_unicos", [])]
    if not uncovered:
        return random.choice(allowed_attrs) if allowed_attrs else ""
    idx = int(state.get("attribute_rotation_index", 0)) % len(uncovered)
    state["attribute_rotation_index"] = idx + 1
    return uncovered[idx]


def _detect_emergent_attribute(answer: str, brief: Dict[str, Any], lexicon: Dict[str, Dict[str, List[str]]]) -> str:
    if not bool(brief.get("config", {}).get("permitir_atributos_emergentes", False)):
        return ""
    if _is_confused_response(answer):
        return ""
    tokens = _tokenize(answer)
    if not tokens or len(tokens) < 6:
        return ""
    known = set()
    for kw in lexicon.values():
        known.update(kw.get("keywords_pos", []))
        known.update(kw.get("keywords_neg", []))
    known.update(STOPWORDS_ES)
    emergent_candidates = [
        t
        for t in tokens
        if t not in known and len(t) >= 4 and t not in EMERGENT_BAD_TOKENS and not t.isdigit()
    ]
    if not emergent_candidates:
        return ""
    cand = emergent_candidates[0]
    if any(m in cand for m in ["entiend", "segur", "gust", "normal", "refier"]):
        return ""
    return f"emergente:{cand}"


def seleccionar_foco_dinamico(state: Dict[str, Any], atributo_detectado_reciente: str = "") -> str:
    allowed_attrs = _allowed_attributes(state.get("brief", {}))
    pendientes = state.get("atributos_pendientes", []) or []
    if pendientes:
        return pendientes[0]

    if atributo_detectado_reciente:
        return atributo_detectado_reciente

    emergentes = state.get("_atributos_emergentes_detectados", []) or []
    if bool(state.get("brief", {}).get("config", {}).get("permitir_atributos_emergentes", False)) and emergentes:
        return emergentes[-1]

    return random.choice(allowed_attrs) if allowed_attrs else ""


def _detect_complaint_repetition(answer: str) -> bool:
    low = _normalize_text(answer)
    markers = ["misma pregunta", "lo mismo", "me sigues preguntando", "vuelves a preguntar", "me repites"]
    return any(m in low for m in markers)


def apply_blueprint_rules(state: Dict[str, Any], blueprint: Dict[str, Any]) -> Dict[str, str]:
    ctx = state.get("_turn_ctx", {})
    attrs_detectados = ctx.get("attrs_detectados", [])
    flags = ctx.get("flags", {})

    plan = state["plan"]
    rules = _decision_rules({"engine_blueprint": blueprint} if blueprint else plan)
    etapa_actual = state.get("etapa_actual", ETAPAS_FIJAS[0])
    allowed_actions = _allowed_actions_for_stage(plan, etapa_actual)
    allowed_attrs = _allowed_attributes(state["brief"])
    foco_default = seleccionar_foco_dinamico(state, attrs_detectados[0] if attrs_detectados else "")

    coverage_ratio = float(state.get("coverage_ratio", 0.0))
    umbral = float(state.get("umbral_cierre", 0.8))
    turn_idx_candidate = int(state.get("turn_index", 0)) + 1
    max_preguntas = int(state.get("max_preguntas", 12))
    max_stagnation = int(_blueprint_params(plan).get("max_turnos_sin_atributos_nuevos", 2) or 2)

    if bool(state.get("close_after_confirmation")):
        action_close = "cerrar" if "cerrar" in allowed_actions else allowed_actions[0]
        return {
            "accion": action_close,
            "regla_disparada": "si confirmacion_previa -> cerrar",
            "foco": seleccionar_foco_dinamico(state, ""),
            "razon_corta": "Se realizo confirmacion; corresponde cerrar entrevista.",
        }

    for r in rules:
        rid = str(r.get("id", "")).strip()
        action = str(r.get("accion", "")).strip()
        if action not in ACCIONES_VALIDAS:
            continue

        applies = False
        foco = foco_default
        if rid == "queja_repeticion":
            applies = bool(flags.get("queja_repeticion"))
        elif rid == "sin_atributo":
            applies = bool(flags.get("sin_atributo"))
        elif rid == "atributo_repetido":
            applies = bool(flags.get("atributo_repetido"))
            if applies:
                foco = seleccionar_foco_dinamico(state, "")
        elif rid == "estancamiento":
            applies = int(state.get("turns_without_new_attrs", 0)) >= max_stagnation
            if applies:
                foco = seleccionar_foco_dinamico(state, "")
        elif rid == "respuesta_vaga":
            applies = bool(flags.get("respuesta_vaga"))
        elif rid == "cierre_umbral":
            applies = coverage_ratio >= umbral
        elif rid == "cierre_max":
            applies = turn_idx_candidate >= max_preguntas

        if not applies:
            continue

        if action not in allowed_actions and action != "cerrar":
            action = allowed_actions[0]

        return {
            "accion": action,
            "regla_disparada": f"si {rid} -> {action}",
            "foco": foco,
            "razon_corta": f"Regla aplicada: {rid}",
        }

    default_action = "profundizar" if "profundizar" in allowed_actions else allowed_actions[0]
    return {
        "accion": default_action,
        "regla_disparada": "default_etapa",
        "foco": foco_default,
        "razon_corta": "No se activo regla prioritaria; se aplica accion permitida por etapa.",
    }


def _criteria_salida_met(criteria: str, state: Dict[str, Any], trace: Dict[str, Any]) -> bool:
    c = str(criteria).lower()
    if "participante orientado" in c:
        return int(trace.get("turn_index", 0)) >= 1
    if "atributos iniciales detectados" in c:
        return len(state.get("atributos_detectados_unicos", [])) >= 1
    if "atributo profundizado" in c:
        return bool(trace.get("atributos_detectados")) and trace.get("accion") in ["profundizar", "pedir_ejemplo", "aclarar"]
    if "ejemplo concreto obtenido" in c:
        return trace.get("accion") == "pedir_ejemplo" and len(str(trace.get("evidencia", ""))) >= 20
    if "prioridad comparativa clara" in c:
        return state.get("coverage_ratio", 0.0) >= 0.5
    if "confirmacion del participante" in c:
        return int(trace.get("turn_index", 0)) >= max(int(state.get("max_preguntas", 12)) - 2, 1)
    if "entrevista finalizada" in c:
        return bool(state.get("done"))
    return False


def update_stage(state: Dict[str, Any], trace: Dict[str, Any]) -> None:
    if trace.get("accion") == "cerrar":
        state["etapa_index"] = len(ETAPAS_FIJAS) - 1
        state["etapa_actual"] = ETAPAS_FIJAS[-1]
        return

    current_stage = state.get("etapa_actual", ETAPAS_FIJAS[0])
    stg = _blueprint_stages(state["plan"]).get(current_stage, {})
    salida = stg.get("criterios_salida", [])
    if not isinstance(salida, list) or not salida:
        return

    if all(_criteria_salida_met(s, state, trace) for s in salida):
        current_idx = int(state.get("etapa_index", 0))
        next_idx = min(current_idx + 1, len(ETAPAS_FIJAS) - 1)
        if next_idx == len(ETAPAS_FIJAS) - 1 and trace.get("accion") != "cerrar":
            return
        state["etapa_index"] = next_idx
        state["etapa_actual"] = ETAPAS_FIJAS[next_idx]


def _is_similar_question(q1: str, q2: str, threshold: float) -> bool:
    a = _normalize_text(q1)
    b = _normalize_text(q2)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _base_templates_fallback(estilo: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    e = estilo if isinstance(estilo, dict) else {}
    lenguaje = str(e.get("lenguaje", "medio")).lower()
    if lenguaje == "simple":
        return {
            "profundizar": ["Por que dices eso?", "Que tiene eso que te gusta?", "Que es lo que mas notas?"],
            "pedir_ejemplo": ["Me puedes dar un ejemplo?", "Te ha pasado algo que te haga pensar eso?", "Cuando lo notaste?"],
            "aclarar": ["Que quieres decir con eso?", "Me lo puedes explicar mas facil?", "A que te refieres?"],
            "explorar_nuevo_tema": ["Aparte de eso, hay algo mas en lo que te fijes?", "Cuando comparas opciones, que otra cosa notas?", "Y en cuanto a {nuevo_foco}, que piensas?"],
            "reformular": ["Te lo pregunto de otra forma: que opinas de {foco}?", "Vamos de nuevo: que es lo que mas te importa de {foco}?"],
            "reconducir": ["Volviendo al tema, que es lo mas importante para ti?", "Si eliges una sola cosa, cual seria?"],
            "confirmacion": ["Entonces, eso seria lo mas importante para ti?", "Dirias que eso pesa mas?"],
            "cerrar": ["Para terminar, que te gustaria destacar?", "Antes de cerrar, que es lo mas importante?"],
        }
    if lenguaje == "avanzado":
        return {
            "profundizar": ["Por que ese punto pesa para ti?", "Que hace que eso sea tan importante para ti?"],
            "pedir_ejemplo": ["Me cuentas un caso concreto?", "Recuerdas una situacion que te haga pensar eso?"],
            "aclarar": ["Cuando dices eso, a que te refieres?", "Como lo explicarias con tus palabras?"],
            "explorar_nuevo_tema": ["Aparte de eso, que otra cosa notas?", "Cuando comparas opciones, que otra diferencia ves?", "Y en cuanto a {nuevo_foco}, que piensas?"],
            "reformular": ["Te lo pregunto distinto: que opinas de {foco}?", "Si miras {foco}, que te importa mas?"],
            "reconducir": ["Volvamos al tema principal, que te importa mas?", "Si priorizas una cosa, cual eliges?"],
            "confirmacion": ["Entonces, eso seria lo principal para ti?", "Confirmas que ese punto pesa mas?"],
            "cerrar": ["Para cerrar, que te gustaria dejar como idea final?", "Antes de terminar, que fue lo mas importante?"],
        }
    return {
        "profundizar": ["Por que eso es importante para ti?", "Que tiene eso que te llama la atencion?"],
        "pedir_ejemplo": ["Me puedes dar un ejemplo?", "Te ha pasado algo que te haga pensar eso?"],
        "aclarar": ["Que quieres decir con eso?", "Me lo puedes explicar con otras palabras?"],
        "explorar_nuevo_tema": ["Aparte de eso, hay algo mas en lo que te fijes?", "Cuando comparas opciones, que otra cosa notas?", "Y en cuanto a {nuevo_foco}, que piensas?"],
        "reformular": ["Te lo pregunto de otra forma: que opinas de {foco}?", "Intentemos otra vez: que es lo mas importante de {foco}?"],
        "reconducir": ["Volviendo al tema, que es lo mas importante para ti?", "Si tuvieras que elegir una sola cosa, cual seria?"],
        "confirmacion": ["Entonces, dirias que eso es lo mas importante para ti?", "Para estar seguros, eso seria lo principal?"],
        "cerrar": ["Para terminar, que te gustaria destacar?", "Antes de cerrar, que es lo mas importante?"],
    }


def _default_question_for_stage(etapa: str, estilo: Optional[Dict[str, Any]] = None) -> str:
    mapping = {
        "apertura_y_encuadre": "Cuando piensas en elegir, que cosas son las mas importantes para ti?",
        "exploracion_espontanea": "Que cosas miras primero cuando comparas opciones?",
        "probing_por_atributos_objetivo": "De lo que dijiste, que te importa mas y por que?",
        "evidencia_y_ejemplos": "Me puedes dar un ejemplo?",
        "priorizacion_y_contraste": "Si tuvieras que elegir, que pondrias primero?",
        "confirmacion": "Entonces, eso seria lo principal para ti?",
        "cierre": "Antes de cerrar, que te gustaria destacar?",
    }
    return simplificar_pregunta(mapping.get(etapa, "Que es lo mas importante para ti en este punto?"), estilo=estilo)


def _build_participant_prompt(question: str, brief: Dict[str, Any]) -> List[Dict[str, str]]:
    base = brief.get("brief", {})
    target = brief.get("target_participante", {})
    if not isinstance(target, dict):
        target = {}
    perfil = {
        "rango_edad": str(target.get("rango_edad", "16-18")),
        "genero": str(target.get("genero", "mixto")),
        "perfil": str(target.get("perfil", "estudiante")),
        "nivel_conocimiento": str(target.get("nivel_conocimiento", "bajo")),
        "nivel_profundizacion": str(target.get("nivel_profundizacion", "baja")),
    }
    return [
        {
            "role": "system",
            "content": (
                "Eres un participante de entrevista cualitativa. "
                "Responde en lenguaje simple y natural en 1 a 3 frases, sin tecnicismos. "
                "Si tu nivel de profundizacion es bajo, responde corto. "
                "Si es medio, explica un poco. Si es alto, da mas detalle. "
                "No hagas listas largas ni parrafos largos. "
                "Si el rango de edad es 13-15 o 16-18, usa frases cortas y vocabulario simple."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contexto: {base.get('antecedente', '')}\n"
                f"Objetivo: {base.get('objetivo_principal', '')}\n"
                f"Perfil participante: {json.dumps(perfil, ensure_ascii=False)}\n"
                f"Pregunta: {question}"
            ),
        },
    ]


def participant_simulator(question: str, brief: Dict[str, Any], perfil_si_existe: Optional[Dict[str, Any]] = None) -> str:
    target = {}
    if isinstance(brief.get("target_participante", {}), dict):
        target = brief.get("target_participante", {})
    elif isinstance(brief.get("modelo_participante", {}), dict):
        target = brief.get("modelo_participante", {})

    prob_confusion = float(brief.get("config", {}).get("probabilidad_confusion", 0.15) or 0.15)
    if random.random() < max(0.0, min(prob_confusion, 1.0)):
        return random.choice(["no entendi la pregunta", "puedes explicarlo mejor?", "no estoy seguro", "a que te refieres?"])

    # Para perfiles jovenes o baja profundidad, forzar respuestas simples y cortas.
    nivel_profundizacion = str(target.get("nivel_profundizacion", "baja")).strip().lower()
    rango_edad = str(target.get("rango_edad", "")).strip()
    nivel_conocimiento = str(target.get("nivel_conocimiento", "bajo")).strip().lower()
    should_force_simple = (
        nivel_profundizacion == "baja"
        or rango_edad in {"13-15", "16-18"}
        or nivel_conocimiento == "bajo"
    )
    if should_force_simple:
        return _humanize_simulated_response(generar_respuesta_simulada(question, target, brief), brief)

    llm_client = _get_client()
    if llm_client is None:
        return _humanize_simulated_response(generar_respuesta_simulada(question, target, brief), brief)

    resp = llm_client.responses.create(model="gpt-4o-mini", input=_build_participant_prompt(question, brief))
    text = (resp.output_text or "").strip()
    final_text = text or generar_respuesta_simulada(question, target, brief)
    return _humanize_simulated_response(final_text, brief)


def question_generator(
    accion: str,
    foco: str,
    brief: Dict[str, Any],
    plan: Dict[str, Any],
    guardrails: Dict[str, Any],
    etapa_actual: str,
    nuevo_foco: str = "",
    lista_atributos_no_cubiertos: Optional[List[str]] = None,
    concepto: str = "",
    template_index: int = 0,
    recent_questions: Optional[List[str]] = None,
    recent_questions_by_action: Optional[Dict[str, List[str]]] = None,
    step_tipo: str = "",
) -> str:
    estilo = _estilo_from_brief(brief)
    foco_alias = alias_natural_atributo(foco or "", brief)
    nuevo_foco_alias = alias_natural_atributo(nuevo_foco or foco or "", brief)
    concept_alias = alias_natural_atributo(concepto or foco or "", brief)
    exp_phrase = frase_experiencia_categoria(brief)
    cmp_phrase = frase_comparacion_categoria(brief)
    effective_tipo = str(step_tipo or "").strip().lower()
    if effective_tipo in {"explorar", "profundizar", "comparar", "reconducir", "cerrar"}:
        step_context = {"brief": brief, "etapa_actual": etapa_actual, "accion": accion}
        action_templates = generate_question_by_step_type(
            effective_tipo,
            foco or concept_alias,
            step_context,
        )

    if accion == "profundizar":
        prompts = [
            generar_pregunta_profundizacion(foco or concept_alias, brief),
            f"Que significa eso para ti sobre {concept_alias}?",
            f"Me puedes dar un ejemplo de {concept_alias}?",
            _frase_importa(concept_alias),
            f"Como notas {concept_alias} {exp_phrase}?",
        ]
        if "action_templates" not in locals():
            action_templates = prompts
    elif accion == "pedir_ejemplo":
        cat = detectar_objeto_estudio(brief).get("categoria", "producto_generico")
        prompts = [
            f"Te paso alguna vez con {concept_alias}?",
            f"Recuerdas una vez sobre {concept_alias} que te hiciera pensar eso?",
            f"Cuando te das cuenta de {concept_alias}?",
            f"En que momento notas mas {concept_alias}?",
            f"Me puedes dar un ejemplo de {concept_alias}?",
        ]
        if cat == "universidad":
            prompts.insert(0, "Que aspecto te hace pensar que es buena universidad?")
        if "action_templates" not in locals():
            action_templates = prompts
    elif accion == "explorar_nuevo_tema":
        prompts = [
            f"Y en cuanto a {nuevo_foco_alias}, que piensas?",
            f"Que diferencia notas en {nuevo_foco_alias}?",
            f"Y sobre {nuevo_foco_alias}, que te importa mas?",
            f"Aparte de eso, que mas notas {cmp_phrase}?",
        ]
        if "action_templates" not in locals():
            action_templates = prompts
    elif accion == "aclarar":
        cat = detectar_objeto_estudio(brief).get("categoria", "producto_generico")
        if "action_templates" not in locals():
            action_templates = [
                f"Que quieres decir con eso de {concept_alias}?",
                f"Me lo puedes explicar con otras palabras sobre {concept_alias}?",
                f"Como lo dirias mas facil sobre {concept_alias}?",
                "A que te refieres exactamente?",
            ]
            if cat == "universidad":
                action_templates.insert(0, "Que caracteristica especifica te hace pensar eso?")
    elif accion == "reformular":
        if "action_templates" not in locals():
            action_templates = [
            f"Te lo pregunto de otra forma: que opinas de {foco_alias}?",
            f"Dicho simple, que piensas de {foco_alias}?",
            f"Si lo vemos facil, que te parece {foco_alias}?",
            ]
    elif accion == "cerrar":
        return normalizar_pregunta("Para terminar, que es lo que mas te gustaria destacar?", brief, estilo)

    templates = _templates_by_action(plan)
    if "action_templates" not in locals():
        action_templates = templates.get(accion, [])
        if not action_templates:
            action_templates = _base_templates_fallback(estilo).get(accion, [])

    if not action_templates:
        return normalizar_pregunta(_default_question_for_stage(etapa_actual, estilo=estilo), brief, estilo)

    allowed_attrs = _allowed_attributes(brief)
    effective_foco = foco or (allowed_attrs[0] if allowed_attrs else "")
    effective_nuevo_foco = nuevo_foco or effective_foco or (allowed_attrs[0] if allowed_attrs else "")

    idx = template_index % len(action_templates)
    uncovered_join = ", ".join((lista_atributos_no_cubiertos or [])[:3])
    concept_value = concepto or effective_foco or "eso"
    data = {
        "foco": alias_natural_atributo(effective_foco, brief),
        "nuevo_foco": alias_natural_atributo(effective_nuevo_foco, brief),
        "concepto": concept_value,
        "lista_atributos_no_cubiertos": uncovered_join or "temas pendientes",
    }

    recent = recent_questions or []
    recent_by_action = (recent_questions_by_action or {}).get(accion, []) if isinstance(recent_questions_by_action, dict) else []
    params = _blueprint_params(plan)
    threshold = max(0.85, float(params.get("anti_repeticion_similitud_umbral", params.get("anti_repeticion_umbral", 0.78)) or 0.78))

    for offset in range(len(action_templates)):
        chosen = action_templates[(idx + offset) % len(action_templates)]
        question = chosen
        for k, v in data.items():
            question = question.replace("{" + k + "}", str(v))
        question = " ".join(question.split())
        if not any(_is_similar_question(question, q, threshold=threshold) for q in recent[-3:]) and not any(
            _is_similar_question(question, q, threshold=max(0.85, threshold)) for q in recent_by_action[-3:]
        ):
            return normalizar_pregunta(question, brief, estilo)

    if accion != "pedir_ejemplo":
        fallback = _base_templates_fallback(estilo)["pedir_ejemplo"][1]
        return normalizar_pregunta(fallback, brief, estilo)
    fallback = _base_templates_fallback(estilo)["explorar_nuevo_tema"][0]
    return normalizar_pregunta(fallback.replace("{nuevo_foco}", effective_nuevo_foco or "otro tema"), brief, estilo)


def _coverage_state(state: Dict[str, Any]) -> Dict[str, Any]:
    target_decl = int(state["target_declarado"])
    target_op = int(state["target_operativo"])
    detectados = len(state.get("atributos_detectados_unicos", []))
    denom = max(target_op, 1)
    ratio = round(detectados / denom, 4)
    return {
        "detectados": detectados,
        "target_declarado": target_decl,
        "target_operativo": target_op,
        "ratio": ratio,
    }


def start_session(brief: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    brief_session = deepcopy(brief)
    if isinstance(brief_session, dict):
        for transient_key in ["_sim_response_pattern_counts", "_response_pattern_counts", "case_identity_prev", "_categoria_previa"]:
            brief_session.pop(transient_key, None)
    coverage_plan = plan.get("client_plan", {}).get("medicion_resumida", {})
    if not isinstance(coverage_plan, dict):
        coverage_plan = {}
    params = _blueprint_params(plan)
    attrs = _allowed_attributes(brief)
    target_decl = int(brief.get("atributos_objetivo", {}).get("target", 1) or 1)
    target_op = len(attrs) if len(attrs) > 0 else max(target_decl, 1)
    max_preguntas = int(
        params.get("max_preguntas", brief.get("config", {}).get("max_preguntas", coverage_plan.get("max_preguntas", 12))) or 12
    )
    umbral = float(params.get("umbral_cierre", coverage_plan.get("umbral_cierre", 0.8)) or 0.8)
    min_turnos_antes_cierre = int(params.get("min_turnos_antes_cierre", 6) or 6)
    min_atributos_confirmados = int(
        params.get("min_atributos_confirmados", max(2, ceil(max(target_op, 1) * 0.5))) or max(2, ceil(max(target_op, 1) * 0.5))
    )

    primera_plan = ""
    try:
        primera_plan = str(plan.get("client_plan", {}).get("primera_pregunta_inicio", "")).strip()
    except Exception:
        primera_plan = ""

    state = {
        "brief": brief_session,
        "plan": deepcopy(plan),
        "identidad_caso": identidad_caso(brief_session),
        "turn_index": 0,
        "etapa_index": 0,
        "etapa_actual": ETAPAS_FIJAS[0],
        "max_preguntas": max_preguntas,
        "umbral_cierre": umbral,
        "target_declarado": max(target_decl, 1),
        "target_operativo": max(target_op, 1),
        "estilo_moderacion": _estilo_from_brief(brief),
        "min_turnos_antes_cierre": max(min_turnos_antes_cierre, 1),
        "min_atributos_confirmados": max(min_atributos_confirmados, 1),
        "project_lexicon": build_project_lexicon(brief),
        "detected_attributes": [],
        "atributos_detectados_unicos": [],
        "_atributos_confirmados_set": set(),
        "_atributos_detectados_set": set(),
        "_atributos_con_evidencia_set": set(),
        "_atributos_mencionados_set": set(),
        "_evidencias_por_atributo": {},
        "_profundidad_por_atributo": {},
        "_depth_hits_by_attr": {},
        "_atributos_emergentes_detectados": [],
        "atributos_pendientes": attrs,
        "coverage_ratio": 0.0,
        "foco_actual": attrs[0] if attrs else "",
        "transcript": [],
        "traces": [],
        "next_question": postprocesar_pregunta_final(
            (
                simplificar_pregunta(primera_plan, estilo=_estilo_from_brief(brief))
                if primera_plan
                else _default_question_for_stage(ETAPAS_FIJAS[0], estilo=_estilo_from_brief(brief))
            ),
            brief_session,
        ),
        "recent_questions": [],
        "recent_questions_by_action": {},
        "turns_without_new_attrs": 0,
        "attribute_rotation_index": 0,
        "template_usage": {},
        "_response_pattern_counts": {},
        "_followup_attempts_same_focus": 0,
        "followups_por_atributo": {},
        "ejemplo_concreto_por_atributo": {},
        "turnos_sin_evidencia_por_atributo": {},
        "done": False,
        "final_reason": "",
        "close_after_confirmation": False,
        "premature_closure_prevented": 0,
    }
    return state


def _process_turn(state: Dict[str, Any], respuesta: str, respuesta_key: str) -> Dict[str, Any]:
    if state.get("done"):
        return {"pregunta": "", respuesta_key: "", "trace": None, "state": state, "done": True}

    turn_index = int(state["turn_index"]) + 1
    etapa_actual = ETAPAS_FIJAS[int(state["etapa_index"])]
    pregunta = state.get("next_question") or _default_question_for_stage(etapa_actual, estilo=_estilo_from_brief(state.get("brief", {})))
    pregunta = postprocesar_pregunta_final(str(pregunta), state.get("brief", {}))

    allowed_attrs = _allowed_attributes(state["brief"])
    lexicon = state.get("project_lexicon", {})
    confusion_level = _confusion_level(respuesta)
    confused = confusion_level == "hard"
    if confused:
        candidates = []
        attr_items = []
    else:
        candidates = detect_attribute_candidates(respuesta, lexicon)
        attr_items_heur = validate_counts(
            respuesta,
            candidates,
            lexicon,
            question=pregunta,
            brief=state["brief"],
            foco_actual=str(state.get("foco_actual", "")),
        )
        llm_items = _llm_validate_attribute_items(
            respuesta,
            state["brief"],
            candidates,
            question=pregunta,
            foco_actual=str(state.get("foco_actual", "")),
        )
        if llm_items:
            by_attr: Dict[str, Dict[str, Any]] = {
                str(it.get("atributo", "")).strip(): dict(it)
                for it in attr_items_heur
                if str(it.get("atributo", "")).strip()
            }
            for lit in llm_items:
                attr = str(lit.get("atributo", "")).strip()
                if not attr:
                    continue
                base = by_attr.get(attr, {"atributo": attr, "detectado": False, "confirmado": False, "evidencia_nivel": False, "cuenta": False, "evidencia": "", "confianza": "baja", "razon_corta": ""})
                base["detectado"] = bool(base.get("detectado", False)) or bool(lit.get("detectado", False)) or bool(lit.get("cuenta", False))
                base["confirmado"] = bool(lit.get("confirmado", False)) or (bool(lit.get("cuenta", False)) and str(lit.get("confianza", "")).lower() in {"media", "alta"})
                base["evidencia_nivel"] = bool(lit.get("evidencia_nivel", False)) or (base["confirmado"] and bool(str(lit.get("evidencia", "")).strip()))
                base["cuenta"] = bool(base["detectado"])
                if str(lit.get("evidencia", "")).strip():
                    base["evidencia"] = str(lit.get("evidencia", "")).strip()[:120]
                base["confianza"] = str(lit.get("confianza", base.get("confianza", "baja"))).lower()
                base["razon_corta"] = str(lit.get("razon_corta", base.get("razon_corta", "")))[:160]
                base["weak_evidence"] = bool(base.get("weak_evidence", False)) or bool(lit.get("weak_evidence", False))
                by_attr[attr] = base
            attr_items = list(by_attr.values())
        else:
            attr_items = attr_items_heur
        for item in attr_items:
            if not str(item.get("evidencia", "")).strip() and bool(item.get("evidencia_nivel", False)):
                item["evidencia_nivel"] = False
                item["confianza"] = "media"
            item["weak_evidence"] = bool(item.get("weak_evidence", False)) or is_weak_evidence(str(item.get("evidencia", "")))

    attrs_detected = [x["atributo"] for x in attr_items if bool(x.get("detectado", False))]
    attrs_confirmed = [x["atributo"] for x in attr_items if bool(x.get("confirmado", False))]
    attrs_with_evidence = [x["atributo"] for x in attr_items if bool(x.get("evidencia_nivel", False))]
    attrs = attrs_detected
    recent_detected_attr = attrs[-1] if attrs else ""

    emergente = _detect_emergent_attribute(respuesta, state["brief"], lexicon)
    if emergente:
        emergentes = state.get("_atributos_emergentes_detectados", [])
        if emergente not in emergentes:
            emergentes.append(emergente)
        state["_atributos_emergentes_detectados"] = emergentes

    updated = set(state.get("detected_attributes", []))
    detected_set = set(state.get("_atributos_detectados_set", set()))
    confirmed_set = set(state.get("_atributos_confirmados_set", set()))
    evidence_set = set(state.get("_atributos_con_evidencia_set", set()))
    mentioned_set = set(state.get("_atributos_mencionados_set", set()))
    if not detected_set and state.get("atributos_detectados_unicos"):
        detected_set = set(state.get("atributos_detectados_unicos", []))
    prev_detected_count = len(updated)
    prev_unique_count = len(detected_set)

    evidencias = state.get("_evidencias_por_atributo", {})
    profundidad = state.get("_profundidad_por_atributo", {})
    depth_hits = state.get("_depth_hits_by_attr", {})
    ejemplo_por_atributo = state.get("ejemplo_concreto_por_atributo", {})
    if not isinstance(evidencias, dict):
        evidencias = {}
    if not isinstance(profundidad, dict):
        profundidad = {}
    if not isinstance(depth_hits, dict):
        depth_hits = {}
    if not isinstance(ejemplo_por_atributo, dict):
        ejemplo_por_atributo = {}

    for item in attr_items:
        attr = str(item.get("atributo", "")).strip()
        if not attr:
            continue
        updated.add(attr)
        if bool(item.get("detectado", False)):
            detected_set.add(attr)
        else:
            mentioned_set.add(attr)
        if bool(item.get("confirmado", False)):
            confirmed_set.add(attr)
        ev = str(item.get("evidencia", "")).strip()
        weak_ev = bool(item.get("weak_evidence", False)) or is_weak_evidence(ev)
        if is_attribute_aligned_evidence(ev, attr, state.get("brief", {})):
            weak_ev = False
        profundidad_minima = bool(item.get("profundidad_minima", False))
        profundidad_fuerte = bool(item.get("profundidad_fuerte", False))
        depth_counted = False
        if profundidad_minima and not weak_ev:
            depth_counted = True
        if bool(item.get("evidencia_nivel", False)) and ev and not weak_ev:
            depth_counted = True
        elif profundidad_fuerte and not weak_ev:
            depth_counted = True
        elif bool(item.get("confirmado", False)) and not weak_ev and (_has_explanation(respuesta) or _has_example(respuesta)):
            depth_counted = True
        if depth_counted:
            evidence_set.add(attr)
            ev_to_store = ev or _short_evidence(respuesta)
            evidencias.setdefault(attr, [])
            if ev_to_store and ev_to_store not in evidencias[attr]:
                evidencias[attr].append(ev_to_store)
            depth_hits[attr] = int(depth_hits.get(attr, 0)) + 1
        if bool(item.get("detectado", False)) and _has_example(respuesta) and not weak_ev:
            ejemplo_por_atributo[attr] = True
        item["depth_counted"] = depth_counted
        if attr in evidencias:
            profundidad[attr] = max(len(evidencias[attr]), int(depth_hits.get(attr, 0)))
        if not bool(item.get("detectado", False)) or (bool(item.get("detectado", False)) and not bool(item.get("confirmado", False))):
            mentioned_set.add(attr)

    state["detected_attributes"] = sorted(updated)
    state["_atributos_detectados_set"] = detected_set
    state["_atributos_confirmados_set"] = confirmed_set
    state["_atributos_con_evidencia_set"] = evidence_set
    state["_atributos_mencionados_set"] = mentioned_set
    state["_evidencias_por_atributo"] = evidencias
    state["_profundidad_por_atributo"] = profundidad
    state["_depth_hits_by_attr"] = depth_hits
    state["ejemplo_concreto_por_atributo"] = ejemplo_por_atributo
    state["atributos_detectados_unicos"] = sorted(detected_set)
    state["atributos_pendientes"] = [a for a in allowed_attrs if a not in detected_set]
    state["atributos_mencionados"] = sorted(mentioned_set)

    new_attrs_added = len(updated) > prev_detected_count
    new_unique_added = len(detected_set) > prev_unique_count
    if new_unique_added:
        state["turns_without_new_attrs"] = 0
    else:
        state["turns_without_new_attrs"] = int(state.get("turns_without_new_attrs", 0)) + 1

    cov = _coverage_state(state)
    state["coverage_ratio"] = cov["ratio"]
    flags = {
        "respuesta_vaga": _is_vague_answer(respuesta),
        "respuesta_muy_breve_o_vaga": es_respuesta_muy_breve_o_vaga(respuesta),
        "adjetivo_vago": _contains_vague_adjective(respuesta),
        "no_entendio": confused,
        "confusion_suave": confusion_level == "soft",
        "queja_repeticion": _detect_complaint_repetition(respuesta),
        "dice_no_se": es_respuesta_vacia(respuesta),
        "sin_atributo": len(attrs_detected) == 0,
        "respuesta_vacia": es_respuesta_vacia(respuesta),
        "atributo_repetido": len(attrs_detected) > 0 and (not new_unique_added),
    }
    has_aligned_attr = any(
        bool(it.get("detectado", False)) and not bool(it.get("weak_evidence", False))
        for it in attr_items
    )
    state["_turn_ctx"] = {"question": pregunta, "answer": respuesta, "attrs_detectados": attrs, "flags": flags}

    pending_attrs = state.get("atributos_pendientes", [])
    focus = seleccionar_foco_dinamico(state, recent_detected_attr)
    min_response_tokens = 4
    categoria_turno = detectar_objeto_estudio(state.get("brief", {})).get("categoria", "producto_generico")
    max_followup_same_focus = 2 if categoria_turno == "universidad" else 3
    foco_prev = str(state.get("foco_actual", "")).strip()
    foco_eval = recent_detected_attr or foco_prev
    foco_depth = int(state.get("_profundidad_por_atributo", {}).get(foco_eval, 0)) if foco_eval else 0
    foco_confirmed = foco_eval in set(state.get("_atributos_confirmados_set", set())) if foco_eval else False
    foco_shallow = bool(foco_eval) and (foco_depth < 2)
    estancado_real = int(state.get("turns_without_new_attrs", 0)) >= 2
    prof_map = state.get("_profundidad_por_atributo", {})
    if not isinstance(prof_map, dict):
        prof_map = {}
    attrs_depth_ge2 = [a for a, d in prof_map.items() if int(d) >= 2]
    attrs_depth_ge1 = [a for a, d in prof_map.items() if int(d) >= 1]
    followups_por_atributo = state.get("followups_por_atributo", {})
    ejemplo_por_atributo = state.get("ejemplo_concreto_por_atributo", {})
    turnos_sin_evidencia_por_atributo = state.get("turnos_sin_evidencia_por_atributo", {})
    if not isinstance(followups_por_atributo, dict):
        followups_por_atributo = {}
    if not isinstance(ejemplo_por_atributo, dict):
        ejemplo_por_atributo = {}
    if not isinstance(turnos_sin_evidencia_por_atributo, dict):
        turnos_sin_evidencia_por_atributo = {}
    foco_has_example = bool(ejemplo_por_atributo.get(foco_prev, False)) if foco_prev else False
    foco_ready_to_rotate = (not foco_prev) or foco_depth >= 2 or foco_has_example
    outside_focus = _is_out_of_focus_response(
        respuesta=respuesta,
        foco=foco_prev,
        brief=state.get("brief", {}),
        attrs_detected=attrs_detected,
    )
    foco_has_clear_signal = bool(
        foco_prev
        and any(
            str(it.get("atributo", "")).strip() == foco_prev
            and bool(it.get("detectado", False))
            and not bool(it.get("weak_evidence", False))
            for it in attr_items
        )
    )
    if foco_prev:
        if foco_has_clear_signal:
            turnos_sin_evidencia_por_atributo[foco_prev] = 0
        else:
            turnos_sin_evidencia_por_atributo[foco_prev] = int(turnos_sin_evidencia_por_atributo.get(foco_prev, 0)) + 1
    foco_turnos_sin_evidencia = int(turnos_sin_evidencia_por_atributo.get(foco_prev, 0)) if foco_prev else 0
    short_semantic_signal_in_focus = bool(
        foco_prev
        and bool(flags.get("respuesta_muy_breve_o_vaga"))
        and any(
            str(it.get("atributo", "")).strip() == foco_prev
            and str(it.get("semantic_class", "rejected")).strip().lower() in {"accepted", "review"}
            and (bool(it.get("semantic_signal", False)) or bool(it.get("detectado", False)))
            for it in attr_items
        )
    )
    respuesta_breve_sin_attr = bool(flags.get("respuesta_muy_breve_o_vaga")) and len(attrs_detected) == 0 and (not short_semantic_signal_in_focus)
    shallow_detected = sorted(
        [a for a in state.get("atributos_detectados_unicos", []) if int(prof_map.get(a, 0)) < 2],
        key=lambda a: (int(prof_map.get(a, 0)), int(followups_por_atributo.get(a, 0))),
    )
    attr_depth_one_pending = ""
    for a in shallow_detected:
        if int(prof_map.get(a, 0)) == 1 and int(followups_por_atributo.get(a, 0)) < 2:
            attr_depth_one_pending = a
            break
    min_followup_attrs = 2
    min_depth_consolidated = min(max(2, int(state.get("min_atributos_confirmados", 1))), max(int(state.get("target_operativo", 1)), 1))
    can_close_by_quality = (
        turn_index >= int(state.get("min_turnos_antes_cierre", 6))
        and len(attrs_depth_ge2) >= min_depth_consolidated
        and len(attrs_depth_ge1) >= min_followup_attrs
    )
    can_close = cov["ratio"] >= float(state["umbral_cierre"]) and can_close_by_quality

    if confused:
        action = "reformular"
        rule = "no_entendio -> reformular"
        reason = "Participante no entendio la pregunta."
    elif foco_prev and foco_turnos_sin_evidencia >= 2 and pending_attrs:
        action = "explorar_nuevo_tema"
        focus = next((a for a in pending_attrs if a != foco_prev), "") or _select_pending_focus(state, pending_attrs) or focus
        rule = "foco_sin_senal_2_turnos -> explorar_nuevo_tema"
        reason = "Se cambia de foco para evitar estancamiento sin evidencia."
    elif outside_focus and foco_prev:
        action = "reconducir"
        focus = foco_prev
        rule = "fuera_de_foco -> reconducir_suave"
        reason = "Se reconduce de forma suave sin perder el hilo."
    elif short_semantic_signal_in_focus and foco_prev and foco_turnos_sin_evidencia <= 1:
        action = "profundizar"
        focus = foco_prev
        rule = "respuesta_corta_con_senal_semantica -> profundizar"
        reason = "Hay senal util del foco actual; se profundiza antes de rotar."
    elif respuesta_breve_sin_attr and foco_prev and foco_turnos_sin_evidencia <= 1:
        action = "reconducir"
        focus = foco_prev
        rule = "respuesta_breve_sin_atributo_primer_intento -> reconducir"
        reason = "Se intenta una reconduccion suave antes de cambiar de tema."
    elif respuesta_breve_sin_attr and foco_prev and foco_turnos_sin_evidencia >= 2 and pending_attrs:
        action = "explorar_nuevo_tema"
        focus = next((a for a in pending_attrs if a != foco_prev), "") or _select_pending_focus(state, pending_attrs) or focus
        rule = "respuesta_breve_sin_atributo_reiterada -> explorar_nuevo_tema"
        reason = "Tras reconduccion sin senal, se explora otro atributo."
    elif flags.get("respuesta_muy_breve_o_vaga") and foco_prev:
        focus = foco_prev
        if len(_tokenize(respuesta)) <= 3 or flags.get("confusion_suave"):
            action = "aclarar"
            rule = "respuesta_muy_breve_o_vaga -> aclarar"
            reason = "La respuesta es breve; se pide una aclaracion concreta."
        else:
            action = "pedir_ejemplo"
            rule = "respuesta_muy_breve_o_vaga -> pedir_ejemplo"
            reason = "La respuesta es vaga; se pide ejemplo antes de cambiar de tema."
    elif flags.get("adjetivo_vago") and not has_aligned_attr:
        action = "aclarar"
        focus = foco_prev or focus
        rule = "adjetivo_vago -> aclarar_especifico"
        reason = "La respuesta usa adjetivos vagos y requiere detalle concreto."
    elif flags.get("confusion_suave") and not has_aligned_attr:
        action = "aclarar"
        rule = "confusion_suave -> aclarar"
        reason = "Hay duda; se aclara sin reformular por completo."
    elif bool(state.get("close_after_confirmation")) and can_close_by_quality:
        action = "cerrar"
        rule = "si confirmacion_previa_y_calidad -> cerrar"
        reason = "Ya hubo confirmacion y la calidad minima se cumplio."
    elif bool(state.get("close_after_confirmation")) and not can_close_by_quality:
        action = "profundizar"
        rule = "confirmacion_previa_sin_calidad -> profundizar"
        reason = "Aun falta profundidad minima antes del cierre."
        state["close_after_confirmation"] = False
        state["premature_closure_prevented"] = int(state.get("premature_closure_prevented", 0)) + 1
    elif can_close:
        action = "confirmacion"
        rule = "prioridad_1_cobertura -> confirmacion"
        reason = "Cobertura objetivo alcanzada."
    elif cov["ratio"] >= float(state["umbral_cierre"]) and not can_close_by_quality:
        action = "pedir_ejemplo" if shallow_detected else "profundizar"
        if shallow_detected:
            focus = shallow_detected[0]
        rule = "cobertura_sin_profundidad_suficiente -> consolidar_profundidad"
        reason = "Aun falta consolidar atributos con profundidad util antes de cerrar."
        state["premature_closure_prevented"] = int(state.get("premature_closure_prevented", 0)) + 1
    elif foco_prev and (atributo_necesita_profundizacion(state, foco_prev) and not foco_has_example):
        action = "profundizar"
        focus = foco_prev
        rule = "foco_actual_sin_profundidad_minima -> profundizar"
        reason = "Se mantiene el foco hasta lograr profundidad minima."
    elif attr_depth_one_pending and int(state.get("_followup_attempts_same_focus", 0)) < max_followup_same_focus:
        action = "profundizar"
        focus = attr_depth_one_pending
        rule = "profundidad_1_requiere_followup -> profundizar"
        reason = "Se busca consolidar profundidad minima antes de cambiar de tema."
    elif shallow_detected and not flags.get("confusion_suave", False) and int(state.get("_followup_attempts_same_focus", 0)) < max_followup_same_focus:
        action = "pedir_ejemplo"
        focus = shallow_detected[0]
        rule = "atributo_detectado_sin_profundidad -> pedir_ejemplo"
        reason = "Se consolida evidencia util antes de abrir otro tema."
    elif new_unique_added and foco_shallow:
        action = "profundizar"
        rule = "atributo_nuevo_superficial -> profundizar"
        reason = "Se detecto un tema nuevo y falta una razon minima."
    elif foco_shallow and pending_attrs and not estancado_real and not flags.get("dice_no_se", False):
        action = "pedir_ejemplo"
        rule = "profundidad_minima_antes_de_cambiar -> pedir_ejemplo"
        reason = "Antes de cambiar de tema, se busca un ejemplo breve."
    elif pending_attrs and (
        (not foco_shallow and foco_ready_to_rotate)
        or estancado_real
        or flags.get("dice_no_se", False)
        or int(state.get("_followup_attempts_same_focus", 0)) >= max_followup_same_focus
    ):
        if shallow_detected and cov["ratio"] >= 0.7:
            action = "pedir_ejemplo"
            focus = shallow_detected[0]
            rule = "profundidad_debil_con_cobertura_alta -> pedir_ejemplo"
            reason = "Se prioriza consolidar profundidad util antes de seguir ampliando cobertura."
        else:
            action = "explorar_nuevo_tema"
            focus = _select_pending_focus(state, pending_attrs) or focus
            rule = "prioridad_2_pendientes -> explorar_nuevo_tema"
            reason = "Aun hay atributos pendientes por cubrir."
    elif len(attrs_detected) == 0 and len(_tokenize(respuesta)) < min_response_tokens:
        action = "reconducir"
        rule = "sin_atributos_y_respuesta_corta -> reconducir"
        reason = "No hay atributos confirmados y la respuesta es demasiado corta."
    elif flags.get("respuesta_vaga"):
        action = "pedir_ejemplo"
        rule = "prioridad_3_respuesta_vaga -> pedir_ejemplo"
        reason = "La respuesta es vaga y requiere evidencia concreta."
    else:
        action = "profundizar"
        rule = "prioridad_4_default -> profundizar"
        reason = "Hay contenido util para profundizar."

    state["foco_actual"] = focus

    foco_followup_attempts = int(state.get("_followup_attempts_same_focus", 0))
    foco_depth_after = int(state.get("_profundidad_por_atributo", {}).get(foco_eval, 0)) if foco_eval else 0
    gained_depth_this_turn = any(
        bool(it.get("depth_counted", False))
        for it in attr_items
        if str(it.get("atributo", "")).strip() == str(foco_eval).strip()
    )
    if action in {"profundizar", "pedir_ejemplo", "aclarar", "reformular"} and foco_eval and foco_depth_after <= 1 and not gained_depth_this_turn:
        foco_followup_attempts += 1
    else:
        foco_followup_attempts = 0
    state["_followup_attempts_same_focus"] = foco_followup_attempts
    if focus:
        if action in {"profundizar", "pedir_ejemplo", "aclarar", "reformular"}:
            followups_por_atributo[focus] = int(followups_por_atributo.get(focus, 0)) + 1
        if int(state.get("_profundidad_por_atributo", {}).get(focus, 0)) >= 2:
            followups_por_atributo[focus] = 0
    state["followups_por_atributo"] = followups_por_atributo
    state["turnos_sin_evidencia_por_atributo"] = turnos_sin_evidencia_por_atributo

    if turn_index >= int(state["max_preguntas"]):
        action = "cerrar"
        rule = "si cierre_max -> cerrar"
        reason = "Se alcanzo max_preguntas."

    if action == "confirmacion":
        state["close_after_confirmation"] = True

    salto_forzado_atributo = bool(
        action == "explorar_nuevo_tema"
        and bool(foco_prev)
        and bool(foco_eval)
        and foco_shallow
    )

    naturalidad = 1.0
    question_low = _normalize_text(pregunta)
    response_low = _normalize_text(respuesta)
    tech_terms = ["atributo", "percepcion", "determinante", "variable", "diferenciador", "evaluacion", "posicionamiento"]
    if any(t in question_low for t in tech_terms):
        naturalidad -= 0.25
    if foco_eval and _normalize_text(foco_eval) in response_low:
        naturalidad -= 0.25
    if salto_forzado_atributo:
        naturalidad -= 0.2
    if action == "explorar_nuevo_tema" and int(state.get("turns_without_new_attrs", 0)) == 0:
        naturalidad -= 0.15
    if not new_unique_added and not _has_explanation(respuesta) and not _has_example(respuesta):
        naturalidad -= 0.12

    resp_counts = state.get("_response_pattern_counts", {})
    if not isinstance(resp_counts, dict):
        resp_counts = {}
    response_signature = " ".join(_tokenize(respuesta)[:5]) or response_low
    if response_signature:
        resp_counts[response_signature] = int(resp_counts.get(response_signature, 0)) + 1
    generic_repeats = int(resp_counts.get(response_signature, 0))
    state["_response_pattern_counts"] = resp_counts
    if _is_generic_response(respuesta):
        naturalidad -= 0.12
    if generic_repeats >= 3:
        naturalidad -= 0.15

    recent_q = [q for q in state.get("recent_questions", []) if isinstance(q, str)]
    repeated_q_count = 0
    for rq in recent_q[-3:]:
        if _is_similar_question(pregunta, rq, threshold=0.9):
            repeated_q_count += 1
    if repeated_q_count >= 2:
        naturalidad -= 0.12
    naturalidad = max(0.0, min(1.0, round(naturalidad, 2)))

    prev_q = ""
    prev_a = ""
    if state.get("transcript"):
        prev_q = str(state["transcript"][-1].get("pregunta", ""))
        prev_a = str(state["transcript"][-1].get("respuesta", ""))
    no_repeticion = 0.0 if (prev_q and _is_similar_question(pregunta, prev_q, threshold=0.92)) else 1.0
    if generic_repeats >= 3:
        no_repeticion = min(no_repeticion, 0.5)
    prev_answer_tokens = set(_tokenize(prev_a))
    q_tokens = set(_tokenize(pregunta))
    continuidad_contextual = 1.0 if (prev_answer_tokens and len(prev_answer_tokens.intersection(q_tokens)) >= 1) else 0.0
    if action in {"profundizar", "pedir_ejemplo", "aclarar", "reformular"}:
        continuidad_contextual = max(continuidad_contextual, 1.0)
    profundidad_turno = 1.0 if action in {"profundizar", "pedir_ejemplo", "aclarar"} else 0.0
    human_like_turno = round(
        (0.35 * naturalidad) + (0.25 * continuidad_contextual) + (0.20 * profundidad_turno) + (0.20 * no_repeticion),
        2,
    )

    if naturalidad < 0.65 and action in {"confirmacion", "cerrar"}:
        action = "profundizar"
        rule = "naturalidad_baja_evitar_cierre -> profundizar"
        reason = "Se prioriza naturalidad y continuidad antes de cerrar."
        state["close_after_confirmation"] = False
        state["premature_closure_prevented"] = int(state.get("premature_closure_prevented", 0)) + 1

    done_reason = ""
    if action == "cerrar":
        if turn_index >= int(state["max_preguntas"]):
            done_reason = "max_preguntas"
        elif cov["ratio"] >= float(state["umbral_cierre"]):
            done_reason = "cobertura_umbral"
        else:
            done_reason = "accion_cerrar"

    semantic_debug_rows = [
        {
            "matched_attribute": str(it.get("matched_attribute", it.get("atributo", ""))),
            "semantic_source": str(it.get("semantic_source", "")),
            "accepted_reason": str(it.get("accepted_reason", "")),
        }
        for it in attr_items
        if isinstance(it, dict) and bool(it.get("semantic_hit", False))
    ]
    semantic_hit = bool(semantic_debug_rows)
    semantic_source = ", ".join([x["semantic_source"] for x in semantic_debug_rows if x.get("semantic_source")][:3])
    matched_attribute = ""
    if attrs_confirmed:
        matched_attribute = str(attrs_confirmed[0])
    elif semantic_debug_rows:
        matched_attribute = str(semantic_debug_rows[0].get("matched_attribute", ""))
    accepted_reason = "rejected_sin_senal"
    if attrs_confirmed:
        accepted_reason = "accepted_con_evidencia"
    elif attrs_detected:
        accepted_reason = "review_por_senal_parcial"
    if any(bool(it.get("short_response_promoted", False)) for it in attr_items if isinstance(it, dict)):
        accepted_reason = "accepted_por_respuesta_corta_semantica"
    focus_kept = bool(foco_prev and focus == foco_prev and action != "explorar_nuevo_tema")
    short_response_promoted = bool(
        any(bool(it.get("short_response_promoted", False)) for it in attr_items if isinstance(it, dict))
    )

    trace = {
        "turn_index": turn_index,
        "etapa_actual": etapa_actual,
        "pregunta": pregunta,
        "respuesta": respuesta,
        "atributos_items": attr_items,
        "atributos_detectados": attrs_detected,
        "atributos_detectados_unicos": state.get("atributos_detectados_unicos", []),
        "atributos_confirmados_unicos": sorted(state.get("_atributos_confirmados_set", [])),
        "atributos_confirmados_set": sorted(state.get("_atributos_confirmados_set", [])),
        "atributos_con_evidencia_set": sorted(state.get("_atributos_con_evidencia_set", [])),
        "atributos_pendientes": state.get("atributos_pendientes", []),
        "coverage_ratio": state.get("coverage_ratio", 0.0),
        "profundidad_por_atributo": {k: int(v) for k, v in sorted(state.get("_profundidad_por_atributo", {}).items())},
        "ejemplo_concreto_por_atributo": dict(state.get("ejemplo_concreto_por_atributo", {})),
        "accion": action,
        "regla_disparada": rule,
        "foco": focus,
        "cobertura": cov,
        "razon_corta": reason,
        "evidencia": _short_evidence(respuesta),
        "naturalidad_turno": naturalidad,
        "salto_forzado_atributo": salto_forzado_atributo,
        "no_repeticion": no_repeticion,
        "profundidad_turno": profundidad_turno,
        "continuidad_contextual": continuidad_contextual,
        "human_like_turno": human_like_turno,
        "profundidad_foco_actual": foco_depth,
        "semantic_hit": semantic_hit,
        "semantic_source": semantic_source,
        "matched_attribute": matched_attribute,
        "accepted_reason": accepted_reason,
        "focus_kept": focus_kept,
        "short_response_promoted": short_response_promoted,
        "semantic_debug_rows": semantic_debug_rows,
    }

    state["turn_index"] = turn_index
    state["transcript"].append({"turn_index": turn_index, "pregunta": pregunta, "respuesta": respuesta})
    state["traces"].append(trace)
    state["recent_questions"] = (state.get("recent_questions", []) + [pregunta])[-3:]

    if action == "cerrar":
        state["done"] = True
        state["final_reason"] = done_reason or "accion_cerrar"
        state["etapa_index"] = len(ETAPAS_FIJAS) - 1
        state["etapa_actual"] = ETAPAS_FIJAS[-1]
        state["next_question"] = ""
    else:
        update_stage(state, trace)
        uncovered = [a for a in allowed_attrs if a not in state.get("atributos_detectados_unicos", [])]
        template_usage = state.get("template_usage", {})
        t_idx = int(template_usage.get(action, 0))
        next_focus = str(focus or "").strip() or seleccionar_foco_dinamico(state, recent_detected_attr)
        step_tipo = _step_tipo_for_stage(state.get("plan", {}), state.get("etapa_actual", ETAPAS_FIJAS[0]), int(state.get("etapa_index", 0)))
        if action in {"profundizar", "pedir_ejemplo", "aclarar"}:
            step_tipo = "profundizar"
        elif action == "reconducir":
            step_tipo = "reconducir"
        elif action == "cerrar":
            step_tipo = "cerrar"
        state["next_question"] = question_generator(
            accion=action,
            foco=next_focus,
            brief=state["brief"],
            plan=state["plan"],
            guardrails=state["brief"].get("guardrails", {}),
            etapa_actual=state["etapa_actual"],
            nuevo_foco=uncovered[0] if uncovered else next_focus,
            lista_atributos_no_cubiertos=uncovered,
            concepto=_extract_concept(respuesta, next_focus, lexicon),
            template_index=t_idx,
            recent_questions=state.get("recent_questions", []),
            recent_questions_by_action=state.get("recent_questions_by_action", {}),
            step_tipo=step_tipo,
        )
        if state.get("recent_questions"):
            last_q = str(state["recent_questions"][-1]).strip()
            new_q = str(state.get("next_question", "")).strip()
            if new_q and _is_similar_question(new_q, last_q, threshold=0.99):
                alt_action = "pedir_ejemplo" if action != "pedir_ejemplo" else "reformular"
                state["next_question"] = question_generator(
                    accion=alt_action,
                    foco=next_focus,
                    brief=state["brief"],
                    plan=state["plan"],
                    guardrails=state["brief"].get("guardrails", {}),
                    etapa_actual=state["etapa_actual"],
                    nuevo_foco=uncovered[0] if uncovered else next_focus,
                    lista_atributos_no_cubiertos=uncovered,
                    concepto=_extract_concept(respuesta, next_focus, lexicon),
                    template_index=t_idx + 1,
                    recent_questions=state.get("recent_questions", []),
                    recent_questions_by_action=state.get("recent_questions_by_action", {}),
                    step_tipo=step_tipo,
                )
        state["next_question"] = postprocesar_pregunta_final(str(state.get("next_question", "")), state.get("brief", {}))
        template_usage[action] = t_idx + 1
        state["template_usage"] = template_usage
        by_action = state.get("recent_questions_by_action", {})
        if not isinstance(by_action, dict):
            by_action = {}
        aq = by_action.get(action, [])
        if not isinstance(aq, list):
            aq = []
        nq = str(state.get("next_question", "")).strip()
        if nq:
            aq = (aq + [nq])[-4:]
        by_action[action] = aq
        state["recent_questions_by_action"] = by_action

    state.pop("_turn_ctx", None)
    return {"pregunta": pregunta, respuesta_key: respuesta, "trace": trace, "state": state, "done": bool(state["done"])}


def step(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("next_question") or _default_question_for_stage(
        state.get("etapa_actual", ETAPAS_FIJAS[0]),
        estilo=_estilo_from_brief(state.get("brief", {})),
    )
    question = postprocesar_pregunta_final(str(question), state.get("brief", {}))
    respuesta = participant_simulator(question, state["brief"])
    return _process_turn(state, respuesta=respuesta, respuesta_key="respuesta_simulada")


def step_with_human_answer(state: Dict[str, Any], human_answer: str) -> Dict[str, Any]:
    safe_answer = (human_answer or "").strip()
    if not safe_answer:
        safe_answer = "No se."
    return _process_turn(state, respuesta=safe_answer, respuesta_key="respuesta_humana")


def run_full(state: Dict[str, Any]) -> Dict[str, Any]:
    guard = int(state.get("max_preguntas", 12)) + 2
    while not state.get("done") and guard > 0:
        step(state)
        guard -= 1

    coverage = _coverage_state(state)
    traces = state.get("traces", [])
    human_vals = [float(t.get("human_like_turno", 0.0)) for t in traces if isinstance(t, dict)]
    natural_vals = [float(t.get("naturalidad_turno", 0.0)) for t in traces if isinstance(t, dict)]
    human_like_index = round(sum(human_vals) / len(human_vals), 3) if human_vals else 0.0
    naturalidad_promedio = round(sum(natural_vals) / len(natural_vals), 3) if natural_vals else 0.0
    prof_map = state.get("_profundidad_por_atributo", {})
    attrs_depth_suf = [a for a, d in (prof_map.items() if isinstance(prof_map, dict) else []) if int(d) >= 2]
    confusion_markers = ["no entendi", "puedes explicarlo mejor", "no estoy seguro", "a que te refieres"]
    confusion_turns = 0
    for t in state.get("transcript", []):
        if not isinstance(t, dict):
            continue
        ans = str(t.get("respuesta", "")).strip().lower()
        if any(m in ans for m in confusion_markers):
            confusion_turns += 1
    porcentaje_turnos_confusion = round(confusion_turns / max(len(state.get("transcript", [])), 1), 3)
    return {
        "transcript": state.get("transcript", []),
        "traces": traces,
        "resumen_final": {
            "done": bool(state.get("done")),
            "motivo_cierre": state.get("final_reason", ""),
            "turnos": int(state.get("turn_index", 0)),
            "cobertura": coverage,
            "atributos_detectados_unicos": state.get("atributos_detectados_unicos", []),
            "atributos_pendientes": state.get("atributos_pendientes", []),
            "coverage_ratio": state.get("coverage_ratio", 0.0),
            "profundidad_por_atributo": {k: int(v) for k, v in sorted(state.get("_profundidad_por_atributo", {}).items())},
            "motivo_cierre_detalle": state.get("final_reason", ""),
            "human_like_index": human_like_index,
            "naturalidad_promedio": naturalidad_promedio,
            "atributos_con_profundidad_suficiente": attrs_depth_suf,
            "cierres_prematuros_evitados": int(state.get("premature_closure_prevented", 0)),
            "premature_closure_prevented": int(state.get("premature_closure_prevented", 0)),
            "porcentaje_turnos_confusion": porcentaje_turnos_confusion,
        },
        "atributos_detectados_unicos": state.get("atributos_detectados_unicos", []),
        "atributos_pendientes": state.get("atributos_pendientes", []),
        "coverage_ratio": state.get("coverage_ratio", 0.0),
        "motivo_cierre": state.get("final_reason", ""),
        "human_like_index": human_like_index,
        "naturalidad_promedio": naturalidad_promedio,
        "state": state,
    }
