import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging
import os
from pathlib import Path
import re
import signal
import time
from aiohttp import web
from backend.resampler import Resampler
from config import CONFIG

server_ready = False
infer_executor = None
last_activity_time = time.monotonic()
active_jobs = 0

_AUDIO_EXT_SPLIT_RE = re.compile(
    r"^(.+?\.(?:wav|flac|ogg|mp3|aiff|aif))\s+(.+)$",
    flags=re.IGNORECASE,
)
_WINE_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\/](.*)$")


def translate_wine_path(path_str: str) -> str:
    """Translate Wine paths (e.g. Z:\\home\\user\\foo.wav) to POSIX paths when running on Linux/macOS."""
    if os.name == "nt" or not path_str:
        return path_str
    if path_str.strip().lower() in ("nul", "/dev/null"):
        return "/dev/null"
    m = _WINE_DRIVE_RE.match(path_str)
    if not m:
        return path_str
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    if drive == "z":
        return "/" + rest.lstrip("/")
    wine_dosdevice = Path.home() / ".wine" / "dosdevices" / f"{drive}:"
    if wine_dosdevice.exists():
        return str(wine_dosdevice / rest)
    return path_str


def split_arguments(input_string: str) -> list[str]:
    """Parse either JSON array payload or legacy space-separated UTAU resampler string."""
    stripped = input_string.strip()
    if not stripped:
        raise ValueError("Empty request payload.")

    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
        if isinstance(data, dict) and isinstance(data.get("args"), list):
            args = [str(x) for x in data["args"]]
        elif isinstance(data, list):
            args = [str(x) for x in data]
        else:
            raise ValueError("JSON payload must be a list or an object with an 'args' list.")
    else:
        tokens = stripped.split(" ")
        if len(tokens) < 4:
            raise ValueError(f"Expected at least 4 UTAU arguments, got {len(tokens)}.")
        if len(tokens) > 12:
            otherargs = tokens[-11:]
            file_path_strings = " ".join(tokens[:-11])
            m = _AUDIO_EXT_SPLIT_RE.match(file_path_strings)
            if not m:
                raise ValueError(f"Could not split input/output audio paths from: {file_path_strings}")
            args = [m.group(1), m.group(2)] + otherargs
        else:
            args = tokens

    if len(args) < 4:
        raise ValueError(f"Expected at least 4 UTAU arguments (in_file, out_file, pitch, velocity), got {len(args)}.")

    args[0] = translate_wine_path(args[0])
    args[1] = translate_wine_path(args[1])

    out_lower = args[1].strip().lower()
    if out_lower not in ("nul", "/dev/null") and not out_lower.endswith(".wav"):
        raise ValueError(f"Output file must end with .wav or be a null device, got: {args[1]}")

    return args


async def handle_get(request: web.Request):
    global last_activity_time
    last_activity_time = time.monotonic()
    if server_ready:
        return web.Response(text="Server Ready", status=200)
    else:
        return web.Response(text="Server Initializing", status=503)


async def handle_post(request: web.Request):
    global server_ready, last_activity_time, active_jobs
    last_activity_time = time.monotonic()

    if "Origin" in request.headers or "Sec-Fetch-Site" in request.headers:
        logging.warning("Rejected browser cross-origin POST request.")
        return web.Response(text="Forbidden: Browser requests are not allowed.", status=403)

    if not server_ready:
        logging.warning("POST arrived but server not ready.")
        return web.Response(text="Server initializing, please retry.", status=503)

    post_data_string = await request.text()
    logging.debug(f"Received POST payload: {post_data_string}")

    try:
        sliced = split_arguments(post_data_string)
    except (ValueError, json.JSONDecodeError) as e:
        err = f"Bad request arguments: {e}"
        logging.warning(err)
        return web.Response(text=err, status=400)

    active_jobs += 1
    try:
        in_file_path, out_file_path = Path(sliced[0]), Path(sliced[1])
        note = f"'{in_file_path.stem}' -> '{out_file_path.name}'"
        logging.info(f"Queued {note} ...")

        loop = asyncio.get_running_loop()
        job = partial(Resampler, *sliced)
        await loop.run_in_executor(infer_executor, job)

        return web.Response(text=f"Success: {note}", status=200)

    except FileNotFoundError as e:
        err = f"Error processing: Input file not found ({e})."
        logging.error(err)
        return web.Response(text=err, status=404)

    except Exception as e:
        err = f"Error processing note: {type(e).__name__}: {e}"
        logging.error(err, exc_info=True)
        return web.Response(text=err, status=500)

    finally:
        active_jobs = max(0, active_jobs - 1)
        last_activity_time = time.monotonic()


async def _idle_watchdog(app: web.Application):
    timeout_sec = int(getattr(CONFIG, "idle_timeout", 0) or 0)
    if timeout_sec <= 0:
        return
    logging.info(f"Idle auto-shutdown enabled: server will exit after {timeout_sec}s of inactivity.")
    try:
        while True:
            await asyncio.sleep(5)
            if active_jobs == 0 and (time.monotonic() - last_activity_time) >= timeout_sec:
                logging.info(f"Server idle for {timeout_sec}s with no active jobs. Shutting down to free GPU VRAM...")
                os.kill(os.getpid(), signal.SIGINT)
                break
    except asyncio.CancelledError:
        pass


async def _on_startup(app: web.Application):
    app["idle_task"] = asyncio.create_task(_idle_watchdog(app))


async def _on_cleanup(app: web.Application):
    idle_task = app.get("idle_task")
    if idle_task:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass
    if infer_executor is not None:
        infer_executor.shutdown(wait=False)


def _resolve_max_workers() -> int:
    raw = getattr(CONFIG, "max_workers", 2)
    if str(raw).strip().lower() == "auto":
        return max(1, min(4, (os.cpu_count() or 2) // 2))
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return 2


def run(port: int = 8572, host: str = "127.0.0.1"):
    global server_ready, infer_executor, last_activity_time
    workers = _resolve_max_workers()
    infer_executor = ThreadPoolExecutor(max_workers=workers)
    last_activity_time = time.monotonic()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    app = web.Application()
    app.add_routes([
        web.get("/", handle_get),
        web.post("/", handle_post),
    ])
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    server_ready = True
    logging.info(f"Listening on http://{host}:{port}; aiohttp + inference-threads={workers}")
    web.run_app(app, host=host, port=port, access_log=None)


if __name__ == "__main__":
    run()
