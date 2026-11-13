import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hifiserver


class TestParseServerCliArgs:
    """Test pure CLI argument parsing in hifiserver.parse_server_cli_args."""

    def test_defaults_without_args_or_env(self, monkeypatch):
        monkeypatch.delenv("HIFISAMPLER_PORT", raising=False)
        args = hifiserver.parse_server_cli_args([])
        assert args.register_only is False
        assert args.managed is False
        assert args.idle_timeout is None
        assert args.port == 8572

    def test_env_port_used_when_cli_port_omitted(self, monkeypatch):
        monkeypatch.setenv("HIFISAMPLER_PORT", "9100")
        args = hifiserver.parse_server_cli_args([])
        assert args.port == 9100

    def test_cli_port_overrides_env_port(self, monkeypatch):
        monkeypatch.setenv("HIFISAMPLER_PORT", "9100")
        args = hifiserver.parse_server_cli_args(["--port", "9200"])
        assert args.port == 9200

    def test_managed_and_idle_timeout_and_register_only(self, monkeypatch):
        monkeypatch.delenv("HIFISAMPLER_PORT", raising=False)
        args = hifiserver.parse_server_cli_args(
            ["--register-only", "--managed", "--idle-timeout", "450", "--port", "8600"]
        )
        assert args.register_only is True
        assert args.managed is True
        assert args.idle_timeout == 450
        assert args.port == 8600

    def test_idle_timeout_secs_alias(self, monkeypatch):
        monkeypatch.delenv("HIFISAMPLER_PORT", raising=False)
        args = hifiserver.parse_server_cli_args(["--managed", "--idle-timeout-secs", "720"])
        assert args.managed is True
        assert args.idle_timeout == 720

    def test_invalid_port_raises(self, monkeypatch):
        monkeypatch.delenv("HIFISAMPLER_PORT", raising=False)
        with pytest.raises(ValueError, match="--port must be between 1 and 65535"):
            hifiserver.parse_server_cli_args(["--port", "0"])
        with pytest.raises(ValueError, match="--port must be between 1 and 65535"):
            hifiserver.parse_server_cli_args(["--port", "70000"])

    def test_negative_idle_timeout_raises(self, monkeypatch):
        monkeypatch.delenv("HIFISAMPLER_PORT", raising=False)
        with pytest.raises(ValueError, match="--idle-timeout must be >= 0"):
            hifiserver.parse_server_cli_args(["--idle-timeout", "-5"])


class TestApplyManagedRuntimeConfig:
    """Test runtime idle_timeout overrides without modifying config.yaml."""

    def test_unmanaged_preserves_existing_idle_timeout(self):
        cfg = SimpleNamespace(idle_timeout=0)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=False, idle_timeout=None)
        assert result == 0
        assert cfg.idle_timeout == 0

    def test_managed_defaults_to_600_when_config_idle_timeout_disabled(self):
        cfg = SimpleNamespace(idle_timeout=0)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=None)
        assert result == 600
        assert cfg.idle_timeout == 600

    def test_managed_keeps_positive_config_idle_timeout_when_cli_omitted(self):
        cfg = SimpleNamespace(idle_timeout=300)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=None)
        assert result == 300
        assert cfg.idle_timeout == 300

    def test_managed_cli_idle_timeout_overrides_config(self):
        cfg = SimpleNamespace(idle_timeout=300)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=900)
        assert result == 900
        assert cfg.idle_timeout == 900

    def test_managed_cli_zero_falls_back_to_default_600(self):
        cfg = SimpleNamespace(idle_timeout=0)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=0)
        assert result == 600
        assert cfg.idle_timeout == 600

    def test_unmanaged_cli_idle_timeout_applies_override(self):
        cfg = SimpleNamespace(idle_timeout=0)
        result = hifiserver.apply_managed_runtime_config(cfg, managed=False, idle_timeout=240)
        assert result == 240
        assert cfg.idle_timeout == 240

    def test_config_yaml_not_modified(self):
        config_yaml = hifiserver.PROJECT_ROOT / "config.yaml"
        before = config_yaml.read_text(encoding="utf-8") if config_yaml.exists() else None
        cfg = SimpleNamespace(idle_timeout=0)
        hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=600)
        after = config_yaml.read_text(encoding="utf-8") if config_yaml.exists() else None
        assert after == before


class TestWriteLocatorBeacon:
    """Test ~/.hifisampler/server.json beacon payload including managed flag."""

    def test_beacon_preserves_fields_and_writes_managed_false_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        hifiserver.write_locator_beacon(port=8572, python_exe="C:/fake/python.exe")

        beacon_file = tmp_path / ".hifisampler" / "server.json"
        assert beacon_file.exists()
        data = json.loads(beacon_file.read_text(encoding="utf-8"))
        assert data["server_root"] == str(hifiserver.PROJECT_ROOT)
        assert data["python_exe"] == "C:/fake/python.exe"
        assert data["script_path"] == str(hifiserver.PROJECT_ROOT / "hifiserver.py")
        assert data["port"] == 8572
        assert data["managed"] is False

    def test_beacon_writes_managed_true_and_custom_port(self, monkeypatch, tmp_path):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        hifiserver.write_locator_beacon(port=9123, python_exe="/usr/bin/python3", managed=True)

        beacon_file = tmp_path / ".hifisampler" / "server.json"
        data = json.loads(beacon_file.read_text(encoding="utf-8"))
        assert data["port"] == 9123
        assert data["managed"] is True
        assert data["python_exe"] == "/usr/bin/python3"
        assert data["server_root"] == str(hifiserver.PROJECT_ROOT)
        assert data["script_path"] == str(hifiserver.PROJECT_ROOT / "hifiserver.py")


def test_get_lock_file_path_includes_port():
    lock_path = hifiserver.get_lock_file_path(port=8599)
    assert "hifisampler-8599-" in lock_path.name
    assert lock_path.name.endswith(".lock")
