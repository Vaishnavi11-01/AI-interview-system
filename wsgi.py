"""Production WSGI entry point. Requires credentials when publicly hosted."""

import os

from web_app import create_app


username = os.environ.get("WEB_USERNAME")
password = os.environ.get("WEB_PASSWORD")
if not username or not password:
    raise RuntimeError("Set WEB_USERNAME and WEB_PASSWORD before starting the hosted application")

app = create_app(
    config_path=os.environ.get("APP_CONFIG", "render_config.json"),
    use_gpu=os.environ.get("USE_GPU", "false").strip().lower() in {"1", "true", "yes", "on"},
    username=username,
    password=password,
)