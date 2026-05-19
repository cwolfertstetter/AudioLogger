from pathlib import Path

from audiologger.config import Config, load_config, save_config


def test_default_values():
    c = Config()
    assert c.hotkey == "ctrl+alt+r"
    assert c.whisper_model == "large-v3"
    assert c.device == "cuda"
    assert c.compute_type == "float16"
    assert c.diarization_enabled is True
    assert c.huggingface_token is None
    assert c.audio_source == "all"
    assert c.filtered_app_names == []
    assert c.notification_enabled is True
    assert c.dictation_hotkey == "ctrl+alt+d"
    assert c.dictation_model == "medium"
    assert c.worker_prewarm is True
    assert c.worker_warm_seconds == 600


def test_load_creates_default_when_missing(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg_file.exists()  # auto-written


def test_load_fills_missing_fields(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("hotkey: f8\n")  # only hotkey set
    cfg = load_config(cfg_file)
    assert cfg.hotkey == "f8"
    assert cfg.whisper_model == "large-v3"  # default filled in


def test_load_round_trip(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    original = Config(hotkey="f9", diarization_enabled=False)
    save_config(cfg_file, original)
    loaded = load_config(cfg_file)
    assert loaded.hotkey == "f9"
    assert loaded.diarization_enabled is False


def test_save_writes_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    save_config(cfg_file, Config(hotkey="ctrl+f1"))
    text = cfg_file.read_text()
    assert "hotkey: ctrl+f1" in text


def test_output_dir_is_path(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("output_dir: C:/Recordings\n")
    cfg = load_config(cfg_file)
    assert cfg.output_dir == Path("C:/Recordings")
