#!/usr/bin/env bash
# Zero-compile Linux/macOS client for HifiSampler in OpenUtau.
# Can be copied or symlinked directly into OpenUtau/Resamplers/hifisampler.sh.
# Named with .sh extension to avoid conflicting with OpenUtau's built-in "hifisampler" tool name.

set -e

PORT="${HIFISAMPLER_PORT:-8572}"
BEACON_FILE="$HOME/.hifisampler/server.json"

if [ $# -lt 1 ]; then
    echo "[hifisampler.sh] Error: No UTAU arguments provided." >&2
    exit 1
fi

shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

probe_server_dir() {
    local dir="$1"
    if [ -n "$dir" ] && [ -d "$dir" ] && { [ -f "$dir/hifiserver.py" ] || [ -f "$dir/start.sh" ]; }; then
        (cd "$dir" && pwd)
        return 0
    fi
    return 1
}

resolve_server_root() {
    # 1. Environment variable
    if [ -n "${HIFISAMPLER_HOME:-}" ]; then
        if probe_server_dir "$HIFISAMPLER_HOME"; then
            return 0
        fi
    fi

    # 2. Symlink target resolution
    if command -v readlink >/dev/null 2>&1; then
        local resolved_self
        resolved_self="$(readlink -f "$0" 2>/dev/null || true)"
        if [ -n "$resolved_self" ]; then
            local link_dir
            link_dir="$(dirname "$resolved_self")"
            if probe_server_dir "$link_dir"; then
                return 0
            fi
        fi
    fi

    # 3. Own directory
    local self_dir
    self_dir="$(cd "$(dirname "$0")" && pwd)"
    if probe_server_dir "$self_dir"; then
        return 0
    fi

    # 4. Global Locator Beacon ~/.hifisampler/server.json
    if [ -f "$BEACON_FILE" ] && command -v python3 >/dev/null 2>&1; then
        local beacon_root
        beacon_root="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d.get("server_root",""))' "$BEACON_FILE" 2>/dev/null || true)"
        if probe_server_dir "$beacon_root"; then
            return 0
        fi
    fi

    return 1
}

is_server_ready() {
    curl -sf --max-time 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1
}

launch_server() {
    local server_root="$1"
    local q_root
    q_root="$(shell_quote "$server_root")"
    local launch_cmd=""
    if [ -x "$server_root/.venv/bin/python" ] && [ -f "$server_root/hifiserver.py" ]; then
        launch_cmd="$(shell_quote "$server_root/.venv/bin/python") $(shell_quote "$server_root/hifiserver.py")"
    else
        launch_cmd="bash $(shell_quote "$server_root/start.sh")"
    fi

    local interactive_cmd="cd $q_root && echo 'Starting HifiSampler Server...' && $launch_cmd ; echo ; echo 'Server exited. Press Enter to close.' ; read -r _"

    if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        local terms=(ptyxis kgx gnome-terminal konsole kitty alacritty wezterm foot xfce4-terminal mate-terminal terminator lxterminal xterm)
        for term in "${terms[@]}"; do
            if command -v "$term" >/dev/null 2>&1; then
                case "$term" in
                    ptyxis|kgx|gnome-terminal)
                        setsid "$term" -- bash -c "$interactive_cmd" >/dev/null 2>&1 < /dev/null 9>&- &
                        return 0
                        ;;
                    konsole|alacritty|lxterminal|xterm)
                        setsid "$term" -e bash -c "$interactive_cmd" >/dev/null 2>&1 < /dev/null 9>&- &
                        return 0
                        ;;
                    xfce4-terminal|mate-terminal|terminator)
                        setsid "$term" -x bash -c "$interactive_cmd" >/dev/null 2>&1 < /dev/null 9>&- &
                        return 0
                        ;;
                    wezterm)
                        setsid "$term" start -- bash -c "$interactive_cmd" >/dev/null 2>&1 < /dev/null 9>&- &
                        return 0
                        ;;
                    kitty|foot)
                        setsid "$term" bash -c "$interactive_cmd" >/dev/null 2>&1 < /dev/null 9>&- &
                        return 0
                        ;;
                esac
            fi
        done
    fi

    # Fallback: background daemon via setsid logging to logs/hifiserver.log (close FD 9 so lock is not inherited)
    mkdir -p "$server_root/logs"
    local q_log
    q_log="$(shell_quote "$server_root/logs/hifiserver.log")"
    setsid bash -c "cd $q_root && exec $launch_cmd" >> "$server_root/logs/hifiserver.log" 2>&1 < /dev/null 9>&- &
}

wait_for_server() {
    local timeout_sec="${1:-45}"
    local elapsed=0
    while [ "$elapsed" -lt "$timeout_sec" ]; do
        if is_server_ready; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

if ! is_server_ready; then
    SERVER_ROOT="$(resolve_server_root || true)"
    if [ -z "$SERVER_ROOT" ]; then
        echo "[hifisampler.sh] Error: Cannot find HifiSampler server root. Run ./start.sh once in the HifiSampler folder to register ~/.hifisampler/server.json, or set HIFISAMPLER_HOME." >&2
        exit 1
    fi

    LOCK_FILE="${TMPDIR:-/tmp}/hifisampler-client-startup-${PORT}-$(id -u).lock"
    if command -v flock >/dev/null 2>&1; then
        (
            flock -w 5 9 || exit 0
            if ! is_server_ready; then
                echo "[hifisampler.sh] Launching HifiSampler server from: $SERVER_ROOT"
                launch_server "$SERVER_ROOT"
                wait_for_server 45 || true
            fi
        ) 9>"$LOCK_FILE"
    else
        launch_server "$SERVER_ROOT"
    fi

    if ! wait_for_server 45; then
        echo "[hifisampler.sh] Error: Server did not become ready within 45s." >&2
        exit 1
    fi
fi

# Build JSON array payload {"args": [...]}
if command -v python3 >/dev/null 2>&1; then
    JSON_PAYLOAD="$(python3 -c 'import json,sys; print(json.dumps({"args": sys.argv[1:]}))' "$@")"
else
    JSON_PAYLOAD='{"args":['
    first=1
    for arg in "$@"; do
        escaped="${arg//\\/\\\\}"
        escaped="${escaped//\"/\\\"}"
        if [ "$first" -eq 1 ]; then
            JSON_PAYLOAD="${JSON_PAYLOAD}\"${escaped}\""
            first=0
        else
            JSON_PAYLOAD="${JSON_PAYLOAD},\"${escaped}\""
        fi
    done
    JSON_PAYLOAD="${JSON_PAYLOAD}]}"
fi

HTTP_CODE="$(curl -sS -o /tmp/hifisampler-resp-$$.txt -w "%{http_code}" --max-time 180 -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "http://127.0.0.1:${PORT}/" || true)"
if [ "$HTTP_CODE" = "200" ]; then
    rm -f "/tmp/hifisampler-resp-$$.txt"
    exit 0
else
    cat "/tmp/hifisampler-resp-$$.txt" >&2 2>/dev/null || true
    rm -f "/tmp/hifisampler-resp-$$.txt"
    exit 1
fi
