from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    config_path = Path(path) if path else ROOT / "settings.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    ollama = config.setdefault("ollama", {})
    ollama["base_url"] = os.getenv("OLLAMA_BASE_URL", ollama.get("base_url", "http://localhost:11434")).rstrip("/")
    # 配置中的相对路径统一相对于项目根目录，避免从其他目录运行时写错位置。
    for section, key in [("app", "output_dir"), ("app", "log_dir"),
                         ("database", "path"), ("vector_store", "path")]:
        if section in config and key in config[section]:
            p = Path(config[section][key]).expanduser()
            config[section][key] = str(p if p.is_absolute() else ROOT / p)
    return config