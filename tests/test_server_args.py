import json
import os
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from backend import server
from backend.server import handle_get, handle_post, split_arguments, translate_wine_path


class TestWinePathTranslation:
    """Test Wine drive path to POSIX path translation."""

    def test_windows_nt_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr(os, "name", "nt")
        path = r"Z:\home\user\audio.wav"
        assert translate_wine_path(path) == path

    def test_null_device_translation_posix(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        assert translate_wine_path("nul") == "/dev/null"
        assert translate_wine_path("NUL") == "/dev/null"
        assert translate_wine_path("/dev/null") == "/dev/null"

    def test_wine_z_drive_posix_translation(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        assert translate_wine_path(r"Z:\home\user\sample.wav") == "/home/user/sample.wav"
        assert translate_wine_path("Z:/home/user/sample.wav") == "/home/user/sample.wav"
        assert translate_wine_path(r"z:\var\audio\test.flac") == "/var/audio/test.flac"

    def test_wine_other_drive_without_dosdevices(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "name", "posix")
        # Point home to a directory without .wine/dosdevices
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        path = r"C:\Voice\oto.ini"
        # Since dosdevice doesn't exist, it should return original path
        assert translate_wine_path(path) == path

    def test_wine_other_drive_with_dosdevices(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "name", "posix")
        fake_c = tmp_path / ".wine" / "dosdevices" / "c:"
        fake_c.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        res = translate_wine_path(r"C:\Voice\sample.wav")
        assert str(fake_c / "Voice" / "sample.wav") in res

    def test_wine_custom_wineprefix_env(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "name", "posix")
        custom_prefix = tmp_path / "custom_wine_prefix"
        fake_c = custom_prefix / "dosdevices" / "c:"
        fake_c.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("WINEPREFIX", str(custom_prefix))

        res = translate_wine_path(r"C:\UTAU\voice.wav")
        assert str(fake_c / "UTAU" / "voice.wav") in res

    def test_empty_or_non_wine_path(self, monkeypatch):
        monkeypatch.setattr(os, "name", "posix")
        assert translate_wine_path("") == ""
        assert translate_wine_path("/usr/bin/local") == "/usr/bin/local"


class TestSplitArguments:
    """Test argument parser for space-separated and JSON payloads."""

    def test_standard_space_separated_arguments(self):
        args = split_arguments("input.wav output.wav C4 100")
        assert args == ["input.wav", "output.wav", "C4", "100"]

    def test_12_arguments_payload(self):
        cmd = "in.wav out.wav C4 100 g-5 0 1000 0 0 100 0 !100 AA"
        args = split_arguments(cmd)
        assert len(args) == 13
        assert args[0] == "in.wav"
        assert args[1] == "out.wav"
        assert args[2] == "C4"
        assert args[3] == "100"
        assert args[4] == "g-5"

    def test_arguments_with_spaces_in_paths(self):
        # When paths have spaces and more than 12 tokens, _AUDIO_EXT_SPLIT_RE handles it
        raw = r"C:\My Voice\in file.wav D:\Out Dir\out file.wav C4 100 g-5 0 1000 0 0 100 0 !100 AA"
        args = split_arguments(raw)
        assert args[0] == r"C:\My Voice\in file.wav"
        assert args[1] == r"D:\Out Dir\out file.wav"
        assert args[2] == "C4"
        assert args[3] == "100"

    def test_json_list_payload(self):
        payload = json.dumps(["in.wav", "out.wav", "C4", "100"])
        args = split_arguments(payload)
        assert args == ["in.wav", "out.wav", "C4", "100"]

    def test_json_object_with_args_payload(self):
        payload = json.dumps({"args": ["in.wav", "out.wav", "C4", "100", "g-5"]})
        args = split_arguments(payload)
        assert args == ["in.wav", "out.wav", "C4", "100", "g-5"]

    def test_null_device_output(self):
        args_nul = split_arguments("in.wav nul C4 100")
        assert args_nul[1].lower() in ("nul", "/dev/null")

        args_devnull = split_arguments("in.wav /dev/null C4 100")
        assert args_devnull[1].lower() in ("nul", "/dev/null")

    def test_empty_payload_raises(self):
        with pytest.raises(ValueError, match="Empty request payload"):
            split_arguments("")
        with pytest.raises(ValueError, match="Empty request payload"):
            split_arguments("    ")

    def test_too_few_arguments_raises(self):
        with pytest.raises(ValueError, match="Expected at least 4 UTAU arguments"):
            split_arguments("in.wav out.wav C4")

    def test_invalid_output_extension_raises(self):
        with pytest.raises(ValueError, match="Output file must end with .wav or be a null device"):
            split_arguments("in.wav output.mp3 C4 100")

    def test_invalid_json_structure_raises(self):
        with pytest.raises(ValueError, match="JSON payload must be a list"):
            split_arguments(json.dumps({"invalid": "format"}))


@pytest.fixture
def aiohttp_app():
    app = web.Application()
    app.add_routes([
        web.get("/", handle_get),
        web.post("/", handle_post),
    ])
    return app


class TestServerRoutes:
    """Test HTTP GET and POST route handlers."""

    @pytest.mark.asyncio
    async def test_handle_get_initializing(self, aiohttp_app):
        server.server_ready = False
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            resp = await client.get("/")
            assert resp.status == 503
            text = await resp.text()
            assert "Server Initializing" in text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_get_ready(self, aiohttp_app):
        server.server_ready = True
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            resp = await client.get("/")
            assert resp.status == 200
            text = await resp.text()
            assert "Server Ready" in text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_post_cross_origin_blocked(self, aiohttp_app):
        server.server_ready = True
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            resp = await client.post("/", headers={"Origin": "http://malicious.com"})
            assert resp.status == 403
            text = await resp.text()
            assert "Forbidden" in text

            resp2 = await client.post("/", headers={"Sec-Fetch-Site": "cross-site"})
            assert resp2.status == 403
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_post_server_not_ready(self, aiohttp_app):
        server.server_ready = False
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            resp = await client.post("/", data="in.wav out.wav C4 100")
            assert resp.status == 503
            text = await resp.text()
            assert "Server initializing" in text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_post_bad_request(self, aiohttp_app):
        server.server_ready = True
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            resp = await client.post("/", data="")
            assert resp.status == 400

            resp2 = await client.post("/", data="in.wav out.wav")
            assert resp2.status == 400
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_post_file_not_found(self, aiohttp_app):
        server.server_ready = True
        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            payload = "nonexistent_input_file_12345.wav output.wav C4 100"
            resp = await client.post("/", data=payload)
            # In handle_post, Resampler is invoked in executor.
            # If in_file does not exist, FileNotFoundError is caught -> 404
            assert resp.status == 404
            text = await resp.text()
            assert "Input file not found" in text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_handle_post_success_with_mock_resampler(self, aiohttp_app, tmp_path):
        server.server_ready = True
        in_file = tmp_path / "test_note.wav"
        in_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
        out_file = tmp_path / "out_note.wav"

        client = TestClient(TestServer(aiohttp_app))
        await client.start_server()
        try:
            with patch("backend.server.Resampler") as mock_resampler:
                mock_resampler.return_value = None
                payload = f"{in_file} {out_file} C4 100"
                resp = await client.post("/", data=payload)
                assert resp.status == 200
                text = await resp.text()
                assert "Success:" in text
                mock_resampler.assert_called_once()
        finally:
            await client.close()
