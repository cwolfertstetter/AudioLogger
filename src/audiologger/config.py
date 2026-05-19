"""Config dataclass with YAML persistence and default-fill semantics."""
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    hotkey: str = "ctrl+alt+r"
    output_dir: Path = field(default_factory=lambda: Path("./recordings"))
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    diarization_enabled: bool = True
    huggingface_token: str | None = None
    audio_source: str = "all"  # "all" | "apps"
    filtered_app_names: list[str] = field(default_factory=list)
    notification_enabled: bool = True
    dictation_hotkey: str = "ctrl+alt+d"
    dictation_model: str = "medium"
    worker_prewarm: bool = True
    worker_warm_seconds: int = 600  # 10 minutes


def _to_yaml_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)
    d["output_dir"] = str(cfg.output_dir)
    return d


def _from_yaml_dict(d: dict[str, Any]) -> Config:
    """Build Config from possibly-partial dict, filling defaults."""
    valid = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in d.items() if k in valid}
    if "output_dir" in filtered and filtered["output_dir"] is not None:
        filtered["output_dir"] = Path(filtered["output_dir"])
    return Config(**filtered)


def save_config(path: Path, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_yaml_dict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config(path: Path) -> Config:
    """Load config. If missing, write defaults. If partial, fill with defaults."""
    if not path.exists():
        cfg = Config()
        save_config(path, cfg)
        return cfg
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = _from_yaml_dict(data)
    # Re-write so missing fields are added to disk for future hand-editing
    save_config(path, cfg)
    return cfg
