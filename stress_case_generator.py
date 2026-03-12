import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from plan_visual_trabajo import generar_plan_visual


CATEGORIES: Dict[str, Dict[str, Any]] = {
    "universidad": {
        "entity_singular": "universidad",
        "entity_plural": "universidades",
        "atributos": [
            "Calidad academica",
            "Prestigio y reputacion",
            "Infraestructura y recursos",
            "Costo y accesibilidad",
            "Empleabilidad y oportunidades laborales",
            "Ambiente universitario y experiencia estudiantil",
        ],
    },
    "soda": {
        "entity_singular": "soda",
        "entity_plural": "sodas",
        "atributos": ["Sabor", "Burbujeo", "Nivel de azucar", "Precio", "Disponibilidad", "Marca"],
    },
    "auto": {
        "entity_singular": "auto",
        "entity_plural": "autos",
        "atributos": ["Seguridad", "Rendimiento", "Precio", "Diseno", "Tecnologia", "Mantenimiento"],
    },
    "banco": {
        "entity_singular": "banco",
        "entity_plural": "bancos",
        "atributos": ["Confianza", "Comisiones", "App movil", "Atencion", "Cobertura", "Beneficios"],
    },
    "seguro": {
        "entity_singular": "seguro",
        "entity_plural": "seguros",
        "atributos": ["Cobertura", "Precio", "Facilidad de reclamo", "Confianza", "Claridad de poliza", "Atencion"],
    },
    "retail": {
        "entity_singular": "tienda",
        "entity_plural": "tiendas",
        "atributos": ["Precio", "Surtido", "Atencion", "Ubicacion", "Promociones", "Experiencia de compra"],
    },
    "tecnologia": {
        "entity_singular": "plataforma",
        "entity_plural": "plataformas",
        "atributos": ["Usabilidad", "Velocidad", "Soporte", "Precio", "Confiabilidad", "Integraciones"],
    },
}

RESPONSE_PROFILES = [
    "claro",
    "vago",
    "redundante",
    "fuera_de_tema",
    "contradictorio",
    "confuso",
    "muy_breve",
    "comparativo",
]

RANGOS_EDAD = ["13-15", "16-18", "19-24", "25-34", "35-49", "50+"]
GENEROS = ["mujer", "hombre", "mixto", "no_especificado"]
PERFILES = ["estudiante", "padre_familia", "profesionista", "consumidor_general", "usuario_experto"]
NIVELES = ["bajo", "medio", "alto"]
PROFUNDIDADES = ["baja", "media", "alta"]


def _build_brief(category: str, profile: str, idx: int, rnd: random.Random) -> Dict[str, Any]:
    cat = CATEGORIES[category]
    attrs = list(cat["atributos"])
    rnd.shuffle(attrs)
    chosen_attrs = attrs[: rnd.randint(4, 6)]
    max_preguntas = rnd.randint(10, 16)
    nivel_conocimiento = rnd.choice(NIVELES)
    nivel_profundizacion = rnd.choice(PROFUNDIDADES)
    rango_edad = rnd.choice(RANGOS_EDAD)
    perfil = rnd.choice(PERFILES)

    brief = {
        "categoria_estudio": category,
        "perfil_respuesta_simulada": profile,
        "brief": {
            "antecedente": (
                f"Estudio cualitativo para entender como personas eligen {cat['entity_singular']}es "
                f"y que factores pesan en la evaluacion."
            ),
            "objetivo_principal": (
                f"Identificar atributos clave que influyen en la eleccion de {cat['entity_singular']}es "
                f"y comparacion entre {cat['entity_plural']}."
            ),
            "tipo_sesion": "Exploracion por atributos",
        },
        "atributos_objetivo": {
            "target": rnd.randint(len(chosen_attrs), len(chosen_attrs) + 2),
            "lista": chosen_attrs,
        },
        "definiciones_operativas": {},
        "config": {
            "max_preguntas": max_preguntas,
            "profundidad": rnd.choice(["Alta", "Media"]),
            "probabilidad_confusion": round(rnd.uniform(0.05, 0.35), 2),
            "primera_pregunta_modo": "ia",
            "primera_pregunta_manual": "",
            "definiciones_operativas_fuente": "generadas_por_ia",
        },
        "target_participante": {
            "rango_edad": rango_edad,
            "rangos_edad": [rango_edad],
            "genero": rnd.choice(GENEROS),
            "perfil": perfil,
            "perfiles": [perfil],
            "nivel_conocimiento": nivel_conocimiento,
            "nivel_profundizacion": nivel_profundizacion,
        },
        "guardrails": {
            "evitar_sugestivas": True,
            "no_preguntas_temporales": True,
            "permitir_atributos_emergentes": True,
            "max_emergentes": 3,
        },
        "contexto": f"stress_case_{idx:03d}",
    }
    return brief


def generate_stress_suite(n_cases: int = 42, seed: int = 20260310) -> Dict[str, Any]:
    rnd = random.Random(seed)
    os.environ.pop("OPENAI_API_KEY", None)
    categories = list(CATEGORIES.keys())
    cases: List[Dict[str, Any]] = []

    for i in range(1, max(30, min(n_cases, 50)) + 1):
        category = categories[(i - 1) % len(categories)]
        profile = RESPONSE_PROFILES[(i - 1) % len(RESPONSE_PROFILES)]
        brief = _build_brief(category=category, profile=profile, idx=i, rnd=rnd)
        plan = generar_plan_visual(brief)
        case_name = f"{category}_{profile}_{i:03d}"
        cases.append(
            {
                "case_name": case_name,
                "brief": brief,
                "plan": plan,
                "perfil_respuesta_simulada": profile,
            }
        )

    return {
        "suite": "stress_test_v1",
        "seed": seed,
        "n_cases": len(cases),
        "cases": cases,
    }


def main() -> None:
    out_path = Path("lab_validation_suite_stress.json")
    suite = generate_stress_suite()
    out_path.write_text(json.dumps(suite, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
