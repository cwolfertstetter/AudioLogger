"""Config dataclass with YAML persistence and default-fill semantics."""
import logging
import shutil
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class Config:
    hotkey: str = "ctrl+alt+r"
    output_dir: Path = field(default_factory=lambda: Path("./recordings"))
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    # Fallback language. WhisperX auto-detects per file, but the mic track is
    # mostly silence and detection there is unreliable -- see transcribe_worker.
    language: str = "de"
    diarization_enabled: bool = True
    huggingface_token: str | None = None
    audio_source: str = "all"  # "all" | "apps"
    filtered_app_names: list[str] = field(default_factory=list)
    notification_enabled: bool = True
    dictation_hotkey: str = "ctrl+alt+d"
    extend_hotkey: str = "ctrl+alt+e"
    dictation_model: str = "medium"
    worker_prewarm: bool = True
    worker_warm_seconds: int = 600  # 10 minutes


def _to_yaml_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)
    d["output_dir"] = str(cfg.output_dir)
    return d


def _has_default_none(f) -> bool:
    return f.default is None


def _from_yaml_dict(d: dict[str, Any]) -> Config:
    """Build Config from possibly-partial dict, filling defaults."""
    by_name = {f.name: f for f in fields(Config)}
    filtered = {k: v for k, v in d.items() if k in by_name}
    # A key left empty in the file ("output_dir:") reads as None; that means
    # "not set", not "set to nothing" -- except where None is the default.
    filtered = {
        k: v for k, v in filtered.items()
        if v is not None or _has_default_none(by_name[k])
    }
    if "output_dir" in filtered:
        filtered["output_dir"] = Path(filtered["output_dir"])
    # YAML 1.1 reads an unquoted `no` (Norwegian) as False.
    if filtered.get("language") is False:
        filtered["language"] = "no"
    if isinstance(filtered.get("filtered_app_names"), str):
        filtered["filtered_app_names"] = [filtered["filtered_app_names"]]
    return Config(**filtered)


def save_config(path: Path, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_yaml_dict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config_checked(path: Path) -> tuple[Config, str | None]:
    """Like load_config, plus a message for the user when the file was unusable.

    An unreadable file is set aside as config.yaml.broken -- never silently
    overwritten -- and the defaults are used, so the app still starts.
    """
    if not path.exists():
        cfg = Config()
        save_config(path, cfg)
        return cfg, None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"expected a mapping of settings, got {type(data).__name__}")
        cfg = _from_yaml_dict(data)
    except Exception as e:
        log.exception("config %s is unusable; starting with defaults", path)
        broken = path.with_name(path.name + ".broken")
        try:
            shutil.copyfile(path, broken)
        except OSError:
            log.exception("Could not keep a copy of the broken config")
        cfg = Config()
        save_config(path, cfg)
        return cfg, f"{path.name} could not be read ({e}). Defaults are in use; your file was kept as {broken.name}."
    # Re-write so missing fields are added to disk for future hand-editing
    save_config(path, cfg)
    return cfg, None


def load_config(path: Path) -> Config:
    """Load config. If missing, write defaults. If partial, fill with defaults."""
    return load_config_checked(path)[0]
