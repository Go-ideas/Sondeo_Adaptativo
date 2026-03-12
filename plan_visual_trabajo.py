import json
import os
import re
import unicodedata
from copy import deepcopy
from math import ceil
from typing import Any, Dict, List, Tuple

from openai import OpenAI

ETAPAS_ENTREVISTA_FIJAS = [
    "apertura_y_encuadre",
    "exploracion_espontanea",
    "probing_por_atributos_objetivo",
    "evidencia_y_ejemplos",
    "priorizacion_y_contraste",
    "confirmacion",
    "cierre",
]

CICLO_POR_TURNO_FIJO = [
    "preguntar",
    "evaluar_respuesta",
    "detectar_atributos",
    "decidir_accion",
    "generar_siguiente_pregunta",
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

PLAN_V32_REQUIRED = {
    "client_plan": ["resumen_ejecutivo", "etapas_visuales", "medicion_resumida"],
    "engine_blueprint": [
        "etapas",
        "ciclo_por_turno",
        "reglas_de_decision",
        "templates_por_accion",
        "parametros",
        "guardrails",
        "validaciones",
    ],
}


def _get_client() -> Any:
    key = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=key) if key else None


def limpiar_texto_input(texto: str) -> str:
    value = str(texto or "").strip()
    if not value:
        return ""
    value = value.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2018", "'").replace("\u2019", "'")
    value = re.sub(r"^\s*\\?['\"]+", "", value)
    value = re.sub(r"\\?['\"]+\s*$", "", value)
    value = re.sub(r"[ \t]+", " ", value).strip()
    return value


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
    objetivo = limpiar_texto_input(base.get("objetivo_principal", "")) or limpiar_texto_input(base.get("antecedente", ""))
    slug_objetivo = _slugify(objetivo)
    slug_categoria = _slugify(categoria)
    slug_caso = f"{slug_categoria}_{slug_objetivo}".strip("_")
    return {
        "categoria": categoria,
        "slug_categoria": slug_categoria,
        "slug_objetivo": slug_objetivo,
        "slug_caso": slug_caso[:120],
        "sustantivo_singular": str(obj.get("sustantivo_singular", "opcion")),
        "sustantivo_plural": str(obj.get("sustantivo_plural", "opciones")),
    }


def unir_prep_articulo(frase: str) -> str:
    out = " ".join(str(frase or "").split())
    out = re.sub(r"\bde el\b", "del", out, flags=re.IGNORECASE)
    out = re.sub(r"\ba el\b", "al", out, flags=re.IGNORECASE)
    return out.strip()


def alias_natural_atributo(attr: str, brief: Dict[str, Any] | None = None) -> str:
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

    if any(x in low for x in ["nivel de azucar", "azucar", "dulce"]):
        return "que tan dulce es"
    if any(x in low for x in ["burbujeo", "burbuja", "carbonat", "gas"]):
        return "el gas"
    if any(x in low for x in ["precio", "costo"]):
        return "el precio"
    if "sabor" in low:
        return "el sabor"

    clean = " ".join(str(attr or "").strip().split()).lower()
    if not clean:
        return "este tema"
    if clean.startswith(("el ", "la ", "los ", "las ", "que ")):
        return unir_prep_articulo(clean)
    return unir_prep_articulo(clean)


def obtener_estilo_moderacion(target_participante: Dict[str, Any]) -> Dict[str, Any]:
    target = target_participante if isinstance(target_participante, dict) else {}
    rango = str(target.get("rango_edad", "")).strip()
    conocimiento = str(target.get("nivel_conocimiento", "medio")).strip().lower()
    profundizacion = str(target.get("nivel_profundizacion", "media")).strip().lower()
    perfil = str(target.get("perfil", "consumidor_general")).strip().lower()

    lenguaje = "medio"
    max_palabras = 16
    tono = "claro y conversacional"
    usar_comparaciones_simples = True
    evitar_abstracciones = False

    if rango in {"13-15", "16-18"} or conocimiento == "bajo" or profundizacion == "baja":
        lenguaje = "simple"
        max_palabras = 14
        tono = "cotidiano"
        evitar_abstracciones = True
    elif perfil == "usuario_experto" and conocimiento == "alto" and profundizacion == "alta":
        lenguaje = "avanzado"
        max_palabras = 20
        tono = "detallado pero claro"
        usar_comparaciones_simples = False

    return {
        "lenguaje": lenguaje,
        "max_palabras_pregunta": max_palabras,
        "usar_comparaciones_simples": usar_comparaciones_simples,
        "evitar_abstracciones": evitar_abstracciones,
        "tono": tono,
    }


def _simplificar_pregunta_por_estilo(texto: str, estilo: Dict[str, Any]) -> str:
    t = limpiar_texto_input(texto)
    if not t:
        return ""
    low = t.lower()
    replacements = {
        "atributo": "cosa",
        "percepcion": "opinion",
        "determinante": "importante",
        "evaluacion": "opinion",
        "variable": "tema",
        "posicionamiento": "imagen",
        "diferenciador": "diferencia",
        "factor": "tema",
        "elemento": "cosa",
        "dimension": "tema",
    }
    for k, v in replacements.items():
        low = re.sub(rf"\b{k}\b", v, low)
    low = re.split(r"[.!?]", low)[0].strip()
    low = re.sub(r"\b(cual es|podrias indicar|de que manera)\b", "que", low).strip()
    low = re.sub(r"\b(en terminos generales|de forma general)\b", "", low).strip()
    low = " ".join(low.split())

    max_words = int(estilo.get("max_palabras_pregunta", 16) or 16)
    words = low.split()
    if len(words) > max_words:
        low = " ".join(words[:max_words])
    if not low.endswith("?"):
        low = low.rstrip(",;:") + "?"
    low = low[:1].upper() + low[1:] if low else low
    return low


def validar_primera_pregunta(pregunta: str, target_participante: Dict[str, Any]) -> str:
    estilo = obtener_estilo_moderacion(target_participante)
    out = _simplificar_pregunta_por_estilo(pregunta, estilo)
    out = re.sub(r"viene a\?$", "viene a la mente?", out, flags=re.IGNORECASE)
    if not out:
        out = "Cuando piensas en esto, que es lo primero que se te viene a la mente?"
    return out


def sanitizar_pregunta_por_categoria(pregunta: str, brief_cliente: Dict[str, Any], estilo: Dict[str, Any]) -> str:
    q = limpiar_texto_input(pregunta)
    if not q:
        return q
    cat = detectar_objeto_estudio(brief_cliente).get("categoria", "producto_generico")
    low = _normalize_text(q)
    group_markers = [
        "estan listos",
        "todos estan de acuerdo",
        "para ustedes",
        "quieren compartir",
        "participantes",
        "estamos de acuerdo",
        "discuten entre si",
        "si pudieras elegir solo tres atributos",
        "se sienten relajados",
        "los participantes",
    ]
    if any(m in low for m in group_markers):
        if cat == "universidad":
            q = "Que te gusta o no te gusta de una universidad?"
        elif cat == "soda":
            q = "Que te gusta o no te gusta de una soda?"
        else:
            q = "Que te gusta o no te gusta de esto?"

    if cat == "universidad":
        q = re.sub(r"\bsoda(s)?\b|\brefresco(s)?\b|\bbebida(s)?\b", "universidad", q, flags=re.IGNORECASE)
        q = re.sub(r"cuando la tomas", "cuando la eliges", q, flags=re.IGNORECASE)
    elif cat == "soda":
        q = re.sub(r"\buniversidad(es)?\b", "soda", q, flags=re.IGNORECASE)
        q = re.sub(r"cuando la eliges", "cuando la tomas", q, flags=re.IGNORECASE)
    elif cat == "auto":
        q = re.sub(r"\bsoda(s)?\b|\brefresco(s)?\b|\bbebida(s)?\b", "auto", q, flags=re.IGNORECASE)
        q = re.sub(r"\buniversidad(es)?\b", "auto", q, flags=re.IGNORECASE)

    q = q.replace("de el", "del").replace("a el", "al")
    q = re.sub(r"\bque tan dulce es\s+que tan dulce es\b", "que tan dulce es", q, flags=re.IGNORECASE)
    q = _simplificar_pregunta_por_estilo(q, estilo)
    q = re.sub(r"viene a\?$", "viene a la mente?", q, flags=re.IGNORECASE)
    return q


def sanitizar_etapas_visuales(etapas: List[Dict[str, Any]], brief_cliente: Dict[str, Any], estilo: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for etapa in etapas if isinstance(etapas, list) else []:
        if not isinstance(etapa, dict):
            continue
        item = dict(etapa)
        item["tipo"] = str(item.get("tipo", "explorar")).strip().lower() or "explorar"
        if item["tipo"] not in {"explorar", "profundizar", "reconducir", "comparar", "cerrar"}:
            item["tipo"] = "explorar"
        item["pregunta_guia"] = sanitizar_pregunta_por_categoria(str(item.get("pregunta_guia", "")), brief_cliente, estilo)
        proposito = limpiar_texto_input(item.get("proposito", ""))
        proposito_low = _normalize_text(proposito)
        if any(
            m in proposito_low
            for m in [
                "participantes",
                "discuten entre",
                "grupo",
                "se sienten relajados",
                "estan listos",
                "todos",
            ]
        ):
            proposito = "Guiar la conversacion 1-a-1 y obtener respuestas claras."
        item["proposito"] = proposito
        criterios = item.get("criterios_exito", [])
        if not isinstance(criterios, list):
            criterios = []
        clean_criterios: List[str] = []
        for c in criterios:
            txt = limpiar_texto_input(c)
            low = _normalize_text(txt)
            if any(m in low for m in ["participantes", "discuten", "grupo", "todos", "estan listos"]):
                continue
            if txt:
                clean_criterios.append(txt)
        item["criterios_exito"] = clean_criterios[:3] or ["Salida accionable para siguiente turno"]
        out.append(item)
    return out


def sanitizar_definiciones_operativas(brief: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    defs = brief.get("definiciones_operativas", {}) if isinstance(brief, dict) else {}
    if not isinstance(defs, dict):
        return {}
    cat = detectar_objeto_estudio(brief).get("categoria", "producto_generico")
    quant_markers = [
        "ranking",
        "rankings",
        "tasa",
        "tasas",
        "estadistica",
        "estadisticas",
        "dato verificable",
        "datos verificables",
        "colocacion laboral demostrada",
        "porcentaje",
    ]
    out: Dict[str, Dict[str, str]] = {}
    for attr, item in defs.items():
        attr_name = limpiar_texto_input(attr)
        row = item if isinstance(item, dict) else {}
        qc = limpiar_texto_input(row.get("que_cuenta", ""))
        qn = limpiar_texto_input(row.get("que_no_cuenta", ""))
        if cat == "universidad":
            low_c = _normalize_text(qc)
            low_n = _normalize_text(qn)
            if (not qc) or any(m in low_c for m in quant_markers):
                qc = (
                    f"Percepciones, asociaciones, opiniones, comparaciones, expectativas y experiencias sobre {attr_name}; "
                    "tambien frases subjetivas como 'tiene nombre', 'me da confianza', 'se siente buena opcion'."
                )
            if (not qn) or any(m in low_n for m in quant_markers):
                qn = f"Respuestas fuera de tema o vacias sobre {attr_name}."
        else:
            if not qc:
                qc = f"Menciones, opiniones, comparaciones o experiencias relacionadas con {attr_name}."
            if not qn:
                qn = f"Respuestas fuera de tema o totalmente vacias sobre {attr_name}."
        out[attr_name] = {"que_cuenta": qc, "que_no_cuenta": qn}
    return out


DEFINICIONES_AI_SCHEMA: Dict[str, Any] = {
    "name": "definiciones_operativas_por_atributo",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["definiciones"],
        "properties": {
            "definiciones": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["atributo", "que_cuenta", "que_no_cuenta"],
                    "properties": {
                        "atributo": {"type": "string"},
                        "que_cuenta": {"type": "string"},
                        "que_no_cuenta": {"type": "string"},
                    },
                },
            }
        },
    },
}

ETAPAS_VISUALES_SCHEMA: Dict[str, Any] = {
    "name": "etapas_visuales_moderador",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["etapas"],
        "properties": {
            "etapas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id_etapa", "tipo", "proposito", "criterios_exito", "pregunta_guia"],
                    "properties": {
                        "id_etapa": {"type": "string"},
                        "tipo": {"type": "string", "enum": ["explorar", "profundizar", "reconducir", "comparar", "cerrar"]},
                        "proposito": {"type": "string"},
                        "criterios_exito": {"type": "array", "items": {"type": "string"}},
                        "pregunta_guia": {"type": "string"},
                    },
                },
            }
        },
    },
}


def generate_definiciones_operativas_ai(
    antecedente: str,
    objetivo_principal: str,
    tipo_sesion: str,
    lista_atributos: List[str],
    model: str = "gpt-4o-mini",
) -> Dict[str, Dict[str, str]]:
    cleaned_atributos = [limpiar_texto_input(x) for x in lista_atributos if limpiar_texto_input(x)]
    if not cleaned_atributos:
        raise ValueError("No hay atributos para generar definiciones operativas")

    llm_client = _get_client()
    if llm_client is None:
        cat = detectar_objeto_estudio({"brief": {"antecedente": antecedente, "objetivo_principal": objetivo_principal}}).get("categoria", "producto_generico")
        if cat == "universidad":
            return {
                atributo: {
                    "que_cuenta": (
                        f"Percepciones, asociaciones, opiniones, comparaciones, expectativas y experiencias sobre {atributo}; "
                        "incluye expresiones subjetivas validas del participante."
                    ),
                    "que_no_cuenta": f"Respuestas fuera de tema o vacias sobre {atributo}.",
                }
                for atributo in cleaned_atributos
            }
        return {
            atributo: {
                "que_cuenta": f"Menciones, opiniones, comparaciones o experiencias relacionadas con {atributo}.",
                "que_no_cuenta": f"Respuestas fuera de tema o totalmente vacias sobre {atributo}.",
            }
            for atributo in cleaned_atributos
        }

    categoria = detectar_objeto_estudio({"brief": {"antecedente": antecedente, "objetivo_principal": objetivo_principal}}).get("categoria", "producto_generico")
    system_prompt = """
Eres especialista en investigacion cualitativa.
Genera definiciones operativas para moderacion IA.
Reglas:
1. "que_cuenta" debe incluir percepciones, asociaciones, opiniones, comparaciones, expectativas y experiencias personales.
2. "que_no_cuenta" solo excluye respuestas fuera de tema o vacias; no excluyas opiniones subjetivas validas.
3. No exigir rankings, datos verificables, estadisticas, tasas, datos de colocacion ni evidencia cuantitativa.
4. En cualitativo SI cuentan frases subjetivas: "se ve buena", "tiene nombre", "se siente segura", "parece cara", "se ve completa".
5. No inventes atributos nuevos.
6. Devuelve exactamente una definicion por cada atributo recibido.
7. Salida estricta en JSON segun schema.
""".strip()
    if categoria == "universidad":
        system_prompt += (
            "\n8. Para universidad, SI cuentan frases subjetivas como: "
            "\"tiene nombre\", \"se escucha buena\", \"se siente segura\", \"parece cara\"."
            "\n9. Para universidad, NO pidas rankings, tasas, estadisticas ni datos verificables."
        )

    user_prompt = (
        "Contexto del estudio:\n"
        f"antecedente={antecedente}\n"
        f"objetivo_principal={objetivo_principal}\n"
        f"tipo_sesion={tipo_sesion}\n"
        f"objeto_estudio={json.dumps(detectar_objeto_estudio({'brief': {'antecedente': antecedente, 'objetivo_principal': objetivo_principal}}), ensure_ascii=False)}\n"
        f"atributos={json.dumps(cleaned_atributos, ensure_ascii=False)}\n\n"
        "Devuelve definiciones operativas para TODOS los atributos recibidos."
    )

    response = llm_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": DEFINICIONES_AI_SCHEMA["name"],
                "strict": DEFINICIONES_AI_SCHEMA["strict"],
                "schema": DEFINICIONES_AI_SCHEMA["schema"],
            }
        },
    )
    payload = json.loads(response.output_text)
    raw_defs = payload.get("definiciones", [])

    by_attr: Dict[str, Dict[str, str]] = {}
    for item in raw_defs:
        atributo = str(item.get("atributo", "")).strip()
        if atributo in cleaned_atributos:
            by_attr[atributo] = {
                "que_cuenta": limpiar_texto_input(item.get("que_cuenta", "")),
                "que_no_cuenta": limpiar_texto_input(item.get("que_no_cuenta", "")),
            }

    for atributo in cleaned_atributos:
        if atributo not in by_attr or not by_attr[atributo]["que_cuenta"] or not by_attr[atributo]["que_no_cuenta"]:
            by_attr[atributo] = {
                "que_cuenta": f"Menciones, opiniones, comparaciones o experiencias relacionadas con {atributo}.",
                "que_no_cuenta": f"Respuestas fuera de tema o totalmente vacias sobre {atributo}.",
            }
    brief_tmp = {
        "brief": {"antecedente": antecedente, "objetivo_principal": objetivo_principal, "tipo_sesion": tipo_sesion},
        "definiciones_operativas": by_attr,
    }
    return sanitizar_definiciones_operativas(brief_tmp)


def base_templates_por_estilo(estilo: Dict[str, Any], brief: Dict[str, Any] | None = None) -> Dict[str, List[str]]:
    lenguaje = str(estilo.get("lenguaje", "medio")).lower()
    obj = detectar_objeto_estudio(brief or {})
    cat = obj.get("categoria", "producto_generico")
    sing = obj.get("sustantivo_singular", "opcion")
    plur = obj.get("sustantivo_plural", "opciones")
    cmp_phrase = (
        f"Cuando comparas {plur}, que otra cosa notas?"
        if cat in {"soda", "universidad", "auto"}
        else "Cuando comparas opciones, que otra cosa notas?"
    )
    exp_phrase = (
        "Que notas cuando la tomas?"
        if cat == "soda"
        else ("Que notas cuando la usas?" if cat == "auto" else "Que notas cuando la eliges?")
    )
    example_phrase = (
        f"Te ha pasado con alguna {sing}?"
        if cat in {"soda", "universidad"}
        else "Te ha pasado alguna vez?"
    )
    if lenguaje == "simple":
        if cat == "universidad":
            return {
                "profundizar": [
                    "Como te das cuenta de que una universidad es buena?",
                    "Que ves o escuchas para pensar que tiene buen nivel?",
                    "Que te haria confiar mas en una universidad?",
                    "Que te hace sentir que vale lo que cuesta?",
                ],
                "pedir_ejemplo": ["Me puedes dar un ejemplo?", "Recuerdas una universidad donde notaste eso?", "Cuando lo notaste mas?"],
                "aclarar": ["Que quieres decir con eso?", "Como lo explicarias mas facil?", "Como lo notas?"],
                "reformular": ["Te lo pregunto de otra forma: que opinas de {foco}?", "Vamos de nuevo: que es lo que mas te importa de {foco}?"],
                "explorar_nuevo_tema": ["Aparte de eso, hay algo mas en lo que te fijas al elegir universidad?", "Y sobre {nuevo_foco}, que opinas?"],
                "reconducir": [
                    "Eso que dices es interesante. Pensando en elegir universidad, que pesa mas para ti?",
                    "Y de eso, que te ayuda a decidir?",
                    "Si lo llevamos a la universidad, que seria lo importante?",
                ],
                "confirmacion": ["Entonces, eso seria lo mas importante para ti?", "Dirias que eso pesa mas?"],
                "cerrar": ["Para terminar, que te gustaria destacar?", "Antes de cerrar, que es lo mas importante?"],
            }
        return {
            "profundizar": ["Que tiene eso que te gusta?", exp_phrase, "Como te das cuenta?", "Que te hace pensar eso?", "Que ves ahi que te importa?"],
            "pedir_ejemplo": ["Me puedes dar un ejemplo?", example_phrase, "Cuando lo notas mas?", "Te acuerdas de una vez reciente?"],
            "aclarar": ["Que quieres decir con eso?", "Me lo puedes explicar mas facil?", "A que te refieres?"],
            "reformular": ["Te lo pregunto de otra forma: que opinas de {foco}?", "Vamos de nuevo: que es lo que mas te importa de {foco}?", "Si lo vemos simple, que piensas de {foco}?"],
            "explorar_nuevo_tema": ["Aparte de eso, hay algo mas en lo que te fijes?", cmp_phrase, "Y sobre {nuevo_foco}, que opinas?"],
            "reconducir": ["Volviendo al tema, que es lo mas importante para ti?", "Si eliges una sola cosa, cual seria?"],
            "confirmacion": ["Entonces, eso seria lo mas importante para ti?", "Dirias que eso pesa mas?"],
            "cerrar": ["Para terminar, que te gustaria destacar?", "Antes de cerrar, que es lo mas importante?"],
        }
    if lenguaje == "avanzado":
        return {
            "profundizar": ["Por que ese punto pesa para ti?", "Que hace que eso te resulte tan importante?"],
            "pedir_ejemplo": ["Me cuentas un caso concreto?", "Recuerdas una situacion que te haga pensar eso?"],
            "aclarar": ["Cuando dices eso, a que te refieres?", "Como lo explicarias con tus palabras?"],
            "reformular": ["Te lo pregunto distinto: que opinas de {foco}?", "Si miras {foco}, que te importa mas?"],
            "explorar_nuevo_tema": ["Pasemos a {nuevo_foco}. Que opinas?", "Y sobre {nuevo_foco}, que te llama la atencion?"],
            "reconducir": ["Volvamos al tema principal, que te importa mas?", "Si priorizas una cosa, cual eliges?"],
            "confirmacion": ["Entonces, eso seria lo principal para ti?", "Confirmas que ese punto pesa mas?"],
            "cerrar": ["Para cerrar, que te gustaria dejar como idea final?", "Antes de terminar, que fue lo mas importante?"],
        }
    return {
        "profundizar": ["Que tiene eso que te gusta?", exp_phrase, "Que te hace decir eso?", "Como lo notas en tu dia a dia?"],
        "pedir_ejemplo": ["Me puedes dar un ejemplo?", example_phrase, "Cuando lo notas mas?", "Que paso la ultima vez?"],
        "aclarar": ["Que quieres decir con eso?", "Me lo puedes explicar con otras palabras?"],
        "reformular": ["Te lo pregunto de otra forma: que opinas de {foco}?", "Intentemos otra vez: que es lo mas importante de {foco}?", "Dicho simple, que piensas de {foco}?"],
        "explorar_nuevo_tema": ["Aparte de eso, hay algo mas en lo que te fijes?", cmp_phrase, "Y en cuanto a {nuevo_foco}, que opinas?"],
        "reconducir": ["Volviendo al tema, que es lo mas importante para ti?", "Si tuvieras que elegir una sola cosa, cual seria?"],
        "confirmacion": ["Entonces, eso seria lo mas importante para ti?", "Dirias que eso pesa mas?"],
        "cerrar": ["Para terminar, que te gustaria destacar?", "Antes de cerrar, que es lo mas importante?"],
    }


def _etapa_nombre(etapa_id: str) -> str:
    names = {
        "apertura_y_encuadre": "Apertura y encuadre",
        "exploracion_espontanea": "Exploracion espontanea",
        "probing_por_atributos_objetivo": "Probing por atributos objetivo",
        "evidencia_y_ejemplos": "Evidencia y ejemplos",
        "priorizacion_y_contraste": "Priorizacion y contraste",
        "confirmacion": "Confirmacion",
        "cierre": "Cierre",
    }
    return names.get(etapa_id, etapa_id)


def _build_etapas_visuales_fallback(estilo: Dict[str, Any], brief_cliente: Dict[str, Any]) -> List[Dict[str, Any]]:
    estilo_visual = dict(estilo or {})
    estilo_visual["max_palabras_pregunta"] = max(int(estilo_visual.get("max_palabras_pregunta", 16) or 16), 18)
    lenguaje = str(estilo.get("lenguaje", "medio")).strip().lower()
    target_simple = lenguaje == "simple"
    obj = detectar_objeto_estudio(brief_cliente)
    cat = obj.get("categoria", "producto_generico")
    if cat == "universidad":
        apertura = "Cuando piensas en elegir universidad, que es lo primero que se te viene a la mente?"
        exploracion = "Que cosas te hacen pensar que una universidad es buena?"
        probing = "Como te das cuenta de que una universidad tiene buen nivel?"
    elif cat == "soda":
        apertura = "Que te gusta de esta soda?"
        exploracion = "Cuando tomas soda, en que te fijas?"
        probing = "Que notas primero?"
    elif cat == "auto":
        apertura = "Cuando piensas en comprar un auto, en que te fijas primero?"
        exploracion = "Que es lo mas importante para ti en un auto?"
        probing = "Que diferencia notas entre un auto y otro?"
    else:
        apertura = "Cuando piensas en esto, que es lo primero que se te viene a la mente?"
        exploracion = "Que te gusta de esto?"
        probing = "Que diferencia notas con otras opciones?"
    by_stage = {
        "apertura_y_encuadre": {
            "tipo": "explorar",
            "proposito": "Romper el hielo y abrir la conversacion sin sesgar.",
            "criterios_exito": ["Participante habla con comodidad", "Aparece al menos un tema inicial"],
            "pregunta_guia": apertura,
        },
        "exploracion_espontanea": {
            "tipo": "explorar",
            "proposito": "Escuchar ideas espontaneas antes de profundizar.",
            "criterios_exito": ["Menciona ideas en sus palabras", "Se identifica foco inicial"],
            "pregunta_guia": exploracion,
        },
        "probing_por_atributos_objetivo": {
            "tipo": "profundizar",
            "proposito": "Profundizar en los atributos objetivo sin sugerir respuestas.",
            "criterios_exito": ["Se cubren atributos pendientes", "Aparecen razones concretas"],
            "pregunta_guia": probing,
        },
        "evidencia_y_ejemplos": {
            "tipo": "profundizar",
            "proposito": "Obtener ejemplos concretos para validar lo dicho.",
            "criterios_exito": ["Da al menos un ejemplo", "Aclara por que piensa eso"],
            "pregunta_guia": "Me puedes dar un ejemplo concreto?",
        },
        "priorizacion_y_contraste": {
            "tipo": "comparar",
            "proposito": "Entender que pesa mas y como compara opciones.",
            "criterios_exito": ["Prioriza entre temas", "Explica diferencia principal"],
            "pregunta_guia": "Que diferencia notas con otras?",
        },
        "confirmacion": {
            "tipo": "reconducir",
            "proposito": "Confirmar hallazgos clave con el participante.",
            "criterios_exito": ["Participante confirma o corrige", "Queda claro lo principal"],
            "pregunta_guia": "Que pesa mas para ti?",
        },
        "cierre": {
            "tipo": "cerrar",
            "proposito": "Cerrar con un resumen final del participante.",
            "criterios_exito": ["Deja idea final clara", "Cierre sin temas pendientes criticos"],
            "pregunta_guia": "Hay algo mas que te importe?",
        },
    }
    if not target_simple:
        by_stage["apertura_y_encuadre"]["pregunta_guia"] = apertura
        by_stage["exploracion_espontanea"]["pregunta_guia"] = exploracion
        by_stage["probing_por_atributos_objetivo"]["pregunta_guia"] = probing
    out: List[Dict[str, Any]] = []
    for i, etapa in enumerate(ETAPAS_ENTREVISTA_FIJAS, start=1):
        row = by_stage.get(etapa, {})
        out.append(
            {
                "paso": i,
                "id_etapa": etapa,
                "tipo": str(row.get("tipo", "explorar")).strip().lower() or "explorar",
                "nombre": _etapa_nombre(etapa),
                "proposito": _simplificar_pregunta_por_estilo(str(row.get("proposito", "Guiar esta etapa de entrevista.")), estilo).rstrip("?"),
                "criterios_exito": [limpiar_texto_input(x) for x in row.get("criterios_exito", ["Salida accionable para siguiente turno"])][:3],
                "pregunta_guia": _sanitize_one_to_one_question(str(row.get("pregunta_guia", "Que opinas sobre este punto?")), brief_cliente, estilo_visual),
            }
        )
    return sanitizar_etapas_visuales(out, brief_cliente, estilo_visual)


def _sanitize_one_to_one_question(text: str, brief_cliente: Dict[str, Any], estilo: Dict[str, Any]) -> str:
    return sanitizar_pregunta_por_categoria(text, brief_cliente, estilo)


def _build_etapas_visuales_ai(brief_cliente: Dict[str, Any], estilo: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = _get_client()
    if client is None:
        return _build_etapas_visuales_fallback(estilo, brief_cliente)
    fallback_rows = _build_etapas_visuales_fallback(estilo, brief_cliente)
    estilo_visual = dict(estilo or {})
    estilo_visual["max_palabras_pregunta"] = max(int(estilo_visual.get("max_palabras_pregunta", 16) or 16), 18)

    payload = {
        "brief": brief_cliente.get("brief", {}),
        "atributos_objetivo": brief_cliente.get("atributos_objetivo", {}),
        "target_participante": brief_cliente.get("target_participante", {}),
        "estilo_moderacion": estilo,
        "etapas_fijas": ETAPAS_ENTREVISTA_FIJAS,
    }
    system_prompt = (
        "Eres moderador cualitativo senior. Genera etapas visuales para cliente, "
        "con lenguaje natural y accionable como moderador humano. "
        "No uses tecnicismos. Mantente en entrevista (sin logistica/analisis tradicional)."
    )
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": ETAPAS_VISUALES_SCHEMA["name"],
                    "strict": ETAPAS_VISUALES_SCHEMA["strict"],
                    "schema": ETAPAS_VISUALES_SCHEMA["schema"],
                }
            },
        )
        data = json.loads(resp.output_text or "{}")
        rows = data.get("etapas", [])
        by_id = {}
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    continue
                sid = limpiar_texto_input(r.get("id_etapa", ""))
                if sid in ETAPAS_ENTREVISTA_FIJAS:
                    by_id[sid] = r

        out: List[Dict[str, Any]] = []
        for i, etapa in enumerate(ETAPAS_ENTREVISTA_FIJAS, start=1):
            row = by_id.get(etapa, {})
            criterios = row.get("criterios_exito", [])
            if not isinstance(criterios, list) or not criterios:
                criterios = ["Salida accionable para siguiente turno"]
            out.append(
                {
                    "paso": i,
                    "id_etapa": etapa,
                    "tipo": str(row.get("tipo", fallback_rows[i - 1].get("tipo", "explorar"))).strip().lower() or "explorar",
                    "nombre": _etapa_nombre(etapa),
                    "proposito": limpiar_texto_input(row.get("proposito", "")) or fallback_rows[i - 1]["proposito"],
                    "criterios_exito": [limpiar_texto_input(x) for x in criterios if limpiar_texto_input(x)][:3],
                    "pregunta_guia": _sanitize_one_to_one_question(
                        limpiar_texto_input(row.get("pregunta_guia", "")) or fallback_rows[i - 1]["pregunta_guia"],
                        brief_cliente,
                        estilo_visual,
                    ),
                }
            )
        return sanitizar_etapas_visuales(out, brief_cliente, estilo_visual)
    except Exception:
        return sanitizar_etapas_visuales(_build_etapas_visuales_fallback(estilo, brief_cliente), brief_cliente, estilo_visual)


def _build_engine_etapas() -> List[Dict[str, Any]]:
    defaults = {
        "apertura_y_encuadre": ("inicio de sesion", "participante orientado"),
        "exploracion_espontanea": ("tema abierto", "atributos iniciales detectados"),
        "probing_por_atributos_objetivo": ("hay atributo candidato", "atributo profundizado"),
        "evidencia_y_ejemplos": ("respuesta general", "ejemplo concreto obtenido"),
        "priorizacion_y_contraste": (">=2 atributos detectados", "prioridad comparativa clara"),
        "confirmacion": ("hallazgos preliminares", "confirmacion del participante"),
        "cierre": ("condicion de cierre", "entrevista finalizada"),
    }
    by_stage_actions = {
        "apertura_y_encuadre": ["reformular", "aclarar", "explorar_nuevo_tema", "reconducir", "confirmacion"],
        "exploracion_espontanea": ["reformular", "profundizar", "explorar_nuevo_tema", "reconducir", "confirmacion"],
        "probing_por_atributos_objetivo": ["reformular", "profundizar", "pedir_ejemplo", "aclarar", "explorar_nuevo_tema", "confirmacion"],
        "evidencia_y_ejemplos": ["reformular", "pedir_ejemplo", "aclarar", "profundizar", "confirmacion"],
        "priorizacion_y_contraste": ["reformular", "explorar_nuevo_tema", "aclarar", "profundizar", "confirmacion"],
        "confirmacion": ["reformular", "aclarar", "confirmacion", "cerrar", "profundizar"],
        "cierre": ["cerrar"],
    }

    etapas = []
    for etapa in ETAPAS_ENTREVISTA_FIJAS:
        entrada, salida = defaults[etapa]
        etapas.append(
            {
                "id": etapa,
                "id_etapa": etapa,
                "criterios_entrada": [entrada],
                "criterios_salida": [salida],
                "acciones_permitidas": by_stage_actions[etapa],
            }
        )
    return etapas


def _build_validaciones_and_pendientes(brief_cliente: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_brief = brief_cliente.get("brief", {})
    atributos = brief_cliente.get("atributos_objetivo", {})
    config = brief_cliente.get("config", {})
    definiciones = brief_cliente.get("definiciones_operativas", {})

    validaciones: List[Dict[str, Any]] = []
    pendientes: List[Dict[str, Any]] = []

    objetivo = limpiar_texto_input(base_brief.get("objetivo_principal", ""))
    antecedente = limpiar_texto_input(base_brief.get("antecedente", ""))
    lista = [limpiar_texto_input(x) for x in atributos.get("lista", []) if limpiar_texto_input(x)]
    target_declarado = int(atributos.get("target", 0) or 0)

    if not objetivo:
        validaciones.append({"regla": "objetivo_principal_presente", "estado": "pendiente", "criticidad": "alta", "detalle": "objetivo_principal vacio"})
        pendientes.append({"campo_faltante": "brief.objetivo_principal", "por_que_importa": "Define foco de moderacion.", "pregunta_para_cliente": "Cual es el objetivo_principal exacto del estudio?"})
    else:
        validaciones.append({"regla": "objetivo_principal_presente", "estado": "ok", "criticidad": "alta", "detalle": "objetivo_principal informado"})

    if not antecedente:
        validaciones.append({"regla": "antecedente_presente", "estado": "pendiente", "criticidad": "media", "detalle": "antecedente vacio"})
        pendientes.append({"campo_faltante": "brief.antecedente", "por_que_importa": "Da contexto a la entrevista.", "pregunta_para_cliente": "Cual es el antecedente del estudio en una frase?"})
    else:
        validaciones.append({"regla": "antecedente_presente", "estado": "ok", "criticidad": "media", "detalle": "antecedente informado"})

    if target_declarado != len(lista):
        validaciones.append({
            "regla": "target_vs_lista",
            "estado": "pendiente",
            "criticidad": "alta",
            "detalle": f"target_declarado={target_declarado}, len_lista={len(lista)}",
        })
        pendientes.append({
            "campo_faltante": "consistencia atributos_objetivo.target",
            "por_que_importa": "La cobertura operativa usa lista real de atributos.",
            "pregunta_para_cliente": f"Deseas ajustar target a {len(lista)} o ampliar lista para llegar a {target_declarado}?",
        })
    else:
        validaciones.append({
            "regla": "target_vs_lista",
            "estado": "ok",
            "criticidad": "alta",
            "detalle": f"target_declarado={target_declarado} consistente con len_lista={len(lista)}",
        })

    defs_fuente = str(config.get("definiciones_operativas_fuente", "no_provistas"))
    defs_ok = bool(isinstance(definiciones, dict) and definiciones)
    if defs_ok:
        validaciones.append({"regla": "definiciones_operativas_fuente", "estado": "ok", "criticidad": "media", "detalle": defs_fuente})
    else:
        validaciones.append({"regla": "definiciones_operativas_fuente", "estado": "pendiente", "criticidad": "media", "detalle": "no_provistas"})
        pendientes.append({"campo_faltante": "definiciones_operativas", "por_que_importa": "Reduce ambiguedad en deteccion.", "pregunta_para_cliente": "Deseas validar o editar definiciones operativas por atributo?"})

    max_preguntas = int(config.get("max_preguntas", 0) or 0)
    if max_preguntas <= 0:
        validaciones.append({"regla": "max_preguntas_valido", "estado": "pendiente", "criticidad": "alta", "detalle": "max_preguntas invalido"})
        pendientes.append({"campo_faltante": "config.max_preguntas", "por_que_importa": "Define cierre por limite de turnos.", "pregunta_para_cliente": "Cual es el max_preguntas permitido?"})
    else:
        validaciones.append({"regla": "max_preguntas_valido", "estado": "ok", "criticidad": "alta", "detalle": "max_preguntas valido"})

    return validaciones, pendientes


def _build_plan_v3_2(brief_cliente: Dict[str, Any]) -> Dict[str, Any]:
    brief_cliente = deepcopy(brief_cliente if isinstance(brief_cliente, dict) else {})
    brief_cliente["definiciones_operativas"] = sanitizar_definiciones_operativas(brief_cliente)
    atributos = brief_cliente.get("atributos_objetivo", {})
    lista = [limpiar_texto_input(x) for x in atributos.get("lista", []) if limpiar_texto_input(x)]
    target_declarado = int(atributos.get("target", len(lista) or 1) or 1)
    target_operativo = len(lista) if len(lista) > 0 else max(target_declarado, 1)

    config = brief_cliente.get("config", {})
    target_participante = brief_cliente.get("target_participante", {})
    rangos_edad = target_participante.get("rangos_edad", [])
    if not isinstance(rangos_edad, list) or not rangos_edad:
        rangos_edad = [target_participante.get("rango_edad", "-")]
    rangos_txt = ", ".join([str(x) for x in rangos_edad if str(x).strip()]) or "-"
    perfiles = target_participante.get("perfiles", [])
    if not isinstance(perfiles, list) or not perfiles:
        perfiles = [target_participante.get("perfil", "-")]
    perfiles_txt = ", ".join([str(x) for x in perfiles if str(x).strip()]) or "-"
    max_preguntas = int(config.get("max_preguntas", 12) or 12)
    umbral = 0.8
    primera_pregunta_modo = str(config.get("primera_pregunta_modo", "ia")).strip().lower()
    primera_pregunta_manual = limpiar_texto_input(config.get("primera_pregunta_manual", ""))
    estilo = obtener_estilo_moderacion(target_participante)
    templates = base_templates_por_estilo(estilo, brief_cliente)
    case_identity = identidad_caso(brief_cliente)

    primera_pregunta_inicio = sanitizar_pregunta_por_categoria(validar_primera_pregunta(_generar_primera_pregunta(
        brief_cliente=brief_cliente,
        modo=primera_pregunta_modo,
        manual=primera_pregunta_manual,
    ), target_participante), brief_cliente, estilo)
    etapas_visuales = sanitizar_etapas_visuales(_build_etapas_visuales_ai(brief_cliente, estilo), brief_cliente, estilo)

    validaciones, pendientes = _build_validaciones_and_pendientes(brief_cliente)

    guardrails = ["no_preguntas_sugestivas", "no_preguntas_dobles"]
    if bool(brief_cliente.get("guardrails", {}).get("no_preguntas_temporales", True)):
        guardrails.append("no_preguntas_temporales")
    guardrails.append("no_inventar_atributos_fuera_de_lista_salvo_emergentes")

    if target_operativo <= 4:
        min_atributos_confirmados = 3
    elif target_operativo <= 8:
        min_atributos_confirmados = ceil(target_operativo * 0.6)
    else:
        min_atributos_confirmados = ceil(target_operativo * 0.5)

    plan = {
        "version": "3.2",
        "titulo_plan": "Plan Visual de Moderacion IA V3.2",
        "identidad_caso": case_identity,
        "client_plan": {
            "resumen_ejecutivo": [
                "Blueprint ejecutable por turnos para entrevista 1-a-1.",
                "Etapas limitadas al flujo de moderacion, sin analisis tradicional.",
                "Cobertura transparente con target_declarado y target_operativo.",
                (
                    "Participante simulado: "
                    f"edades_objetivo={rangos_txt}, "
                    f"edad_autoplay={target_participante.get('rango_edad','-')}, "
                    f"genero={target_participante.get('genero','-')}, "
                    f"perfiles_objetivo={perfiles_txt}, "
                    f"perfil_autoplay={target_participante.get('perfil','-')}, "
                    f"profundizacion={target_participante.get('nivel_profundizacion','-')}, "
                    f"prob_confusion={config.get('probabilidad_confusion', 0.15)}"
                ),
                f"Primera pregunta de inicio ({'IA' if primera_pregunta_modo == 'ia' else 'manual'}): {primera_pregunta_inicio}",
            ],
            "target_participante_resumen": {
                "rango_edad": target_participante.get("rango_edad", "-"),
                "rangos_edad": rangos_edad,
                "genero": target_participante.get("genero", "-"),
                "perfil": target_participante.get("perfil", "-"),
                "perfiles": perfiles,
                "nivel_conocimiento": target_participante.get("nivel_conocimiento", "-"),
                "nivel_profundizacion": target_participante.get("nivel_profundizacion", "-"),
                "probabilidad_confusion": config.get("probabilidad_confusion", 0.15),
            },
            "estilo_moderacion_aplicado": estilo,
            "identidad_caso": case_identity,
            "etapas_visuales": etapas_visuales,
            "primera_pregunta_inicio": primera_pregunta_inicio,
            "preguntas_ejemplo_estilo": {
                "profundizar": _simplificar_pregunta_por_estilo(templates.get("profundizar", ["Por que dices eso?"])[0], estilo),
                "reformular": _simplificar_pregunta_por_estilo(
                    templates.get("reformular", ["Vamos de nuevo: que opinas de {foco}?"])[0].replace(
                        "{foco}",
                        alias_natural_atributo(lista[0] if lista else "este tema", brief_cliente),
                    ),
                    estilo,
                ),
            },
            "medicion_resumida": {
                "target_declarado": max(target_declarado, 1),
                "target_operativo": max(target_operativo, 1),
                "umbral_cierre": umbral,
                "max_preguntas": max(max_preguntas, 1),
                "como_se_calcula": "cobertura = atributos_detectados_unicos / target_operativo",
            },
        },
        "engine_blueprint": {
            "etapas": _build_engine_etapas(),
            "ciclo_por_turno": CICLO_POR_TURNO_FIJO,
            "reglas_de_decision": [
                {"id": "no_entendio", "si": "si respuesta contiene no entendi o reformula", "entonces": "reformular", "accion": "reformular", "prioridad": 1},
                {"id": "queja_repeticion", "si": "si queja_repeticion", "entonces": "pedir ejemplo concreto", "accion": "pedir_ejemplo", "prioridad": 2},
                {"id": "sin_atributo", "si": "si no hay atributo detectado", "entonces": "reconducir al objetivo", "accion": "reconducir", "prioridad": 3},
                {"id": "atributo_repetido", "si": "si atributo detectado ya cubierto", "entonces": "explorar atributo pendiente", "accion": "explorar_nuevo_tema", "prioridad": 4},
                {"id": "estancamiento", "si": "si 2 turnos sin atributos nuevos", "entonces": "explorar nuevo tema en atributo no cubierto", "accion": "explorar_nuevo_tema", "prioridad": 5},
                {"id": "respuesta_vaga", "si": "si respuesta vaga", "entonces": "pedir ejemplo", "accion": "pedir_ejemplo", "prioridad": 6},
                {"id": "cierre_umbral", "si": "si coverage_ratio >= umbral_cierre", "entonces": "confirmacion", "accion": "confirmacion", "prioridad": 7},
                {"id": "cierre_max", "si": "si turnos == max_preguntas", "entonces": "cerrar", "accion": "cerrar", "prioridad": 8},
            ],
            "templates_por_accion": templates,
            "parametros": {
                "umbral_cierre": umbral,
                "max_preguntas": max(max_preguntas, 1),
                "anti_repeticion_similitud_umbral": 0.78,
                "anti_repeticion_umbral": 0.78,
                "max_turnos_sin_atributos_nuevos": 2,
                "min_turnos_antes_cierre": 6,
                "min_atributos_confirmados": max(1, int(min_atributos_confirmados)),
            },
            "guardrails": guardrails,
            "validaciones": validaciones,
        },
        "pendientes": pendientes,
    }
    return plan


def generar_primera_pregunta_por_categoria(brief_cliente: Dict[str, Any], target_participante: Dict[str, Any]) -> str:
    cat = detectar_objeto_estudio(brief_cliente).get("categoria", "producto_generico")
    if cat == "universidad":
        options = [
            "Cuando piensas en elegir una universidad, que es lo primero que te viene a la mente?",
            "Que es lo mas importante para ti al elegir una universidad?",
            "Si comparas universidades, que notas primero?",
        ]
    elif cat == "soda":
        options = [
            "Cuando tomas una soda, que es lo primero que notas?",
            "Que tiene que tener una soda para que te guste?",
            "Si comparas sodas, que notas primero?",
        ]
    elif cat == "auto":
        options = [
            "Cuando piensas en comprar un auto, en que te fijas primero?",
            "Que es lo mas importante para ti en un auto?",
        ]
    else:
        options = [
            "Cuando piensas en esto, que es lo primero que te viene a la mente?",
            "Que es lo mas importante para ti aqui?",
        ]
    return sanitizar_pregunta_por_categoria(validar_primera_pregunta(options[0], target_participante), brief_cliente, obtener_estilo_moderacion(target_participante))


def _generar_primera_pregunta(brief_cliente: Dict[str, Any], modo: str, manual: str) -> str:
    target_participante = brief_cliente.get("target_participante", {})
    estilo = obtener_estilo_moderacion(target_participante)
    if modo == "manual" and manual:
        return validar_primera_pregunta(manual if manual.endswith("?") else f"{manual}?", target_participante)

    attrs = [limpiar_texto_input(x) for x in brief_cliente.get("atributos_objetivo", {}).get("lista", []) if limpiar_texto_input(x)]
    objetivo = limpiar_texto_input(brief_cliente.get("brief", {}).get("objetivo_principal", ""))
    tipo_sesion = limpiar_texto_input(brief_cliente.get("brief", {}).get("tipo_sesion", ""))
    tipo_norm = _normalize_text(tipo_sesion)
    if "exploracion por atributos" in tipo_norm and str(estilo.get("lenguaje", "medio")) == "simple":
        return generar_primera_pregunta_por_categoria(brief_cliente, target_participante)

    client = _get_client()
    if client is not None:
        try:
            prompt = {
                "objetivo_principal": objetivo,
                "tipo_sesion": tipo_sesion,
                "atributos_objetivo": attrs,
                "estilo_participante": estilo,
                "reglas": [
                    "pregunta abierta",
                    "lenguaje cotidiano",
                    "maximo 15 palabras",
                    "sin tecnicismos",
                    "si estudio exploratorio, prioriza apertura espontanea",
                ],
            }
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Genera una sola pregunta inicial para una entrevista cualitativa. "
                            "Debe sonar como moderador humano, abierta, natural, no tecnica y facil de entender."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            )
            q = " ".join((resp.output_text or "").strip().split())
            if q:
                return validar_primera_pregunta(q if q.endswith("?") else f"{q}?", target_participante)
        except Exception:
            pass

    if "exploracion por atributos" in _normalize_text(tipo_sesion):
        return generar_primera_pregunta_por_categoria(brief_cliente, target_participante)
    if attrs:
        return validar_primera_pregunta(
            f"Cuando piensas en {alias_natural_atributo(attrs[0], brief_cliente)}, que te llama mas la atencion?",
            target_participante,
        )
    return generar_primera_pregunta_por_categoria(brief_cliente, target_participante)


def generar_primera_pregunta_inicio(brief_cliente: Dict[str, Any], modo: str = "ia", manual: str = "") -> str:
    return _generar_primera_pregunta(brief_cliente=brief_cliente, modo=modo, manual=manual)


def _categoria_base_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "categorias")


def _token_keywords(texto: str) -> List[str]:
    toks = [t for t in _normalize_text(texto).split() if len(t) >= 3 and t not in {"para", "con", "por", "que", "como"}]
    return list(dict.fromkeys(toks))[:12]


def generar_atributos_iniciales(
    categoria: str,
    objetivo: str,
    target_atributos: int = 8,
    atributos_usuario: List[str] | None = None,
    model: str = "gpt-4o-mini",
) -> List[str]:
    attrs_user = [limpiar_texto_input(x) for x in (atributos_usuario or []) if limpiar_texto_input(x)]
    if attrs_user:
        return attrs_user[: max(1, int(target_atributos or len(attrs_user)))]

    n = max(1, int(target_atributos or 8))
    client = _get_client()
    if client:
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Eres especialista en investigacion cualitativa. "
                            "Devuelve solo JSON con una lista de atributos claros, concretos y no redundantes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "categoria": categoria,
                                "objetivo": objetivo,
                                "target_atributos": n,
                                "formato_salida": {"atributos": ["string"]},
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "atributos_iniciales",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["atributos"],
                            "properties": {
                                "atributos": {
                                    "type": "array",
                                    "minItems": n,
                                    "maxItems": max(n, 12),
                                    "items": {"type": "string"},
                                }
                            },
                        },
                    },
                },
            )
            parsed = json.loads(resp.output_text or "{}")
            attrs = [limpiar_texto_input(x) for x in parsed.get("atributos", []) if limpiar_texto_input(x)]
            if attrs:
                return attrs[:n]
        except Exception:
            pass

    # fallback deterministico
    base_pool = [
        "Calidad percibida",
        "Precio",
        "Experiencia de uso",
        "Confianza",
        "Diferenciacion",
        "Disponibilidad",
        "Servicio",
        "Reputacion",
        "Beneficio principal",
        "Facilidad de acceso",
    ]
    cat_hint = limpiar_texto_input(categoria).title()
    attrs = [f"{cat_hint} - {x}" for x in base_pool[:n]]
    return attrs


def _generar_semantic_seed(categoria: str, atributos: List[str]) -> Dict[str, Any]:
    by_attr: Dict[str, List[str]] = {}
    for attr in atributos:
        alias = alias_natural_atributo(attr, {"brief": {"antecedente": categoria}})
        kws = list(dict.fromkeys(_token_keywords(attr) + _token_keywords(alias)))
        by_attr[attr] = kws[:10]
    flat_map: Dict[str, str] = {}
    for attr, kws in by_attr.items():
        for kw in kws:
            flat_map.setdefault(kw, _normalize_text(attr))
    return {"categoria": _slugify(categoria), "keywords_por_atributo": by_attr, "semantic_map_seed": flat_map}


def generar_suite_laboratorio(
    categoria: str,
    objetivo: str,
    atributos: List[str],
    n_ejemplos: int = 12,
) -> Dict[str, Any]:
    n = min(20, max(10, int(n_ejemplos or 12)))
    perfiles = ["claro", "vago", "redundante", "fuera_de_tema", "contradictorio", "confuso", "muy_breve", "comparativo"]
    casos: List[Dict[str, Any]] = []
    cat = _slugify(categoria)
    for i in range(1, n + 1):
        perfil = perfiles[(i - 1) % len(perfiles)]
        foco = atributos[(i - 1) % max(len(atributos), 1)] if atributos else "Tema principal"
        casos.append(
            {
                "case_name": f"{cat}_{perfil}_{i:03d}",
                "perfil_respuesta_simulada": perfil,
                "prompt_seed": {
                    "categoria": categoria,
                    "objetivo": objetivo,
                    "foco": foco,
                    "respuesta_ejemplo": f"Me fijo en {alias_natural_atributo(foco, {'brief': {'antecedente': categoria}})}.",
                },
            }
        )
    return {"suite": "categoria_seed_v1", "categoria": cat, "n_casos": len(casos), "cases": casos}


def registrar_categoria(categoria: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    root = os.path.dirname(__file__)
    reg_path = os.path.join(root, "categorias_registradas.json")
    payload = []
    if os.path.exists(reg_path):
        try:
            loaded = json.loads(open(reg_path, "r", encoding="utf-8").read())
            if isinstance(loaded, list):
                payload = loaded
        except Exception:
            payload = []
    slug = _slugify(categoria)
    row = {
        "categoria": slug,
        "categoria_display": limpiar_texto_input(categoria),
        "fecha_registro": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        **(metadata if isinstance(metadata, dict) else {}),
    }
    payload = [x for x in payload if str(x.get("categoria", "")) != slug]
    payload.append(row)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return row


def crear_categoria_proyecto(
    categoria: str,
    objetivo: str,
    target_atributos: int = 8,
    atributos_usuario: List[str] | None = None,
    n_ejemplos_suite: int = 12,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    categoria_clean = limpiar_texto_input(categoria) or "categoria_nueva"
    objetivo_clean = limpiar_texto_input(objetivo) or "Objetivo no especificado"
    slug = _slugify(categoria_clean)

    categoria_dir = os.path.join(_categoria_base_dir(), slug)
    os.makedirs(categoria_dir, exist_ok=True)

    atributos = generar_atributos_iniciales(
        categoria=categoria_clean,
        objetivo=objetivo_clean,
        target_atributos=target_atributos,
        atributos_usuario=atributos_usuario,
        model=model,
    )
    semantic_seed = _generar_semantic_seed(categoria_clean, atributos)
    suite_lab = generar_suite_laboratorio(
        categoria=categoria_clean,
        objetivo=objetivo_clean,
        atributos=atributos,
        n_ejemplos=n_ejemplos_suite,
    )

    atributos_payload = {
        "categoria": slug,
        "objetivo": objetivo_clean,
        "target_atributos": int(target_atributos or len(atributos)),
        "atributos": atributos,
    }

    atributos_path = os.path.join(categoria_dir, "atributos.json")
    suite_path = os.path.join(categoria_dir, "suite_lab.json")
    semantic_path = os.path.join(categoria_dir, "semantic_seed.json")

    with open(atributos_path, "w", encoding="utf-8") as f:
        json.dump(atributos_payload, f, ensure_ascii=False, indent=2)
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(suite_lab, f, ensure_ascii=False, indent=2)
    with open(semantic_path, "w", encoding="utf-8") as f:
        json.dump(semantic_seed, f, ensure_ascii=False, indent=2)

    registro = registrar_categoria(
        categoria_clean,
        {
            "slug_categoria": slug,
            "ruta_categoria": os.path.relpath(categoria_dir, os.path.dirname(__file__)).replace("\\", "/"),
            "n_atributos": len(atributos),
            "n_casos_suite": int(suite_lab.get("n_casos", 0)),
            "archivos": {
                "atributos": os.path.basename(atributos_path),
                "suite_lab": os.path.basename(suite_path),
                "semantic_seed": os.path.basename(semantic_path),
            },
            "objetivo": objetivo_clean,
        },
    )

    return {
        "ok": True,
        "categoria": slug,
        "categoria_dir": categoria_dir,
        "archivos": {
            "atributos.json": atributos_path,
            "suite_lab.json": suite_path,
            "semantic_seed.json": semantic_path,
        },
        "atributos_generados": atributos,
        "suite_generada": suite_lab.get("n_casos", 0),
        "registro": registro,
        "mensaje": "categoria creada y lista para laboratorio",
    }


def _ensure_plan_v32(plan: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("Plan invalido: no es objeto")
    for top in ["version", "titulo_plan", "client_plan", "engine_blueprint", "pendientes"]:
        if top not in plan:
            raise ValueError(f"Plan invalido: falta {top}")
    for block, keys in PLAN_V32_REQUIRED.items():
        node = plan.get(block, {})
        if not isinstance(node, dict):
            raise ValueError(f"Plan invalido: {block} no es objeto")
        for k in keys:
            if k not in node:
                raise ValueError(f"Plan invalido: falta {block}.{k}")
    return plan


def generar_plan_visual(brief_cliente: Dict[str, Any], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    # V3.2: blueprint ejecutable estable y deterministico.
    # Se conserva la firma para compatibilidad con la app.
    _ = model
    return _ensure_plan_v32(_build_plan_v3_2(brief_cliente))


if __name__ == "__main__":
    ejemplo = {
        "brief": {
            "antecedente": "Buscamos diferenciadores de una universidad privada.",
            "objetivo_principal": "Identificar drivers de eleccion.",
            "tipo_sesion": "Exploracion por atributos",
        },
        "atributos_objetivo": {"target": 10, "lista": ["Calidad Academica", "Carreras que cuenta la universidad", "Colegiatura", "Instalaciones"]},
        "config": {"max_preguntas": 12, "definiciones_operativas_fuente": "generadas_por_ia"},
        "guardrails": {"evitar_sugestivas": True, "no_preguntas_temporales": True},
    }
    print(json.dumps(generar_plan_visual(ejemplo), ensure_ascii=False, indent=2))
