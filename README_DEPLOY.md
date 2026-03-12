# README_DEPLOY

## Objetivo
Desplegar esta app Streamlit desde GitHub en Streamlit Community Cloud usando `streamlit_app.py` como entrypoint.

## 1) Subir repo a GitHub
1. Inicializa o reutiliza el repositorio Git.
2. Confirma que `streamlit_app.py` esta en la raiz.
3. Sube los archivos nuevos: `deployment_config.json`, `published_categories.json`, `.streamlit/config.toml`, `requirements.txt`.

## 2) Conectar en Streamlit Community Cloud
1. Entra a Streamlit Community Cloud.
2. Selecciona `New app`.
3. Elige tu repo y branch.
4. En `Main file path` selecciona `streamlit_app.py`.

## 3) Configurar secrets (opcional)
Si necesitas proteger modo admin por password, agrega en `Secrets`:

```toml
ADMIN_PASSWORD = "tu_password_segura"
```

## 4) Desplegar
1. Pulsa `Deploy`.
2. Espera build e instalacion de dependencias.

## 5) Compartir URL cliente
1. Usa la URL publica generada por Streamlit Cloud.
2. Con `deployment_config.json` en modo `client`, el cliente vera solo experiencia limpia.

## 6) Operacion cliente/admin
1. `deployment_config.json` controla modo por defecto y visibilidad admin.
2. Para entrar a admin, activa `show_admin_tools=true` o define `ADMIN_PASSWORD`.
3. La seccion `Publicacion cliente` permite publicar/retirar categorias y aprobar/rechazar manualmente.

## 7) Estructura de categorias entrenadas
El runtime busca categorias en:

```text
trained_categories/<categoria>/
  brief.json
  latest_plan.json
  model_config.json
  latest_metrics.json
  semantic_seed.json
  metadata.json
```

## 8) Sesiones cliente
Si `save_client_sessions=true`, la app guarda sesiones en:

```text
client_sessions/<categoria>/<timestamp>_<session_id>.json
```
