#!/usr/bin/env bash
# One-command local launcher for the runnable FinX-Moda prototype.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$PROJECT_ROOT/apps/moda-api"
WEB_DIR="$PROJECT_ROOT/apps/moda-web"
API_PORT="${API_PORT:-3001}"
WEB_PORT="${WEB_PORT:-5173}"
VERIFY=0
NO_OPEN="${NO_OPEN:-0}"

usage() {
    cat <<'USAGE'
Usage: ./run-moda.sh [--verify] [--no-open]

Installs the locked dependencies, builds both Moda apps, starts the local API
and storefront, checks that both answer, and opens the storefront when a local
desktop opener is available. Press Ctrl+C to stop both processes.

Options:
  --verify   Run both application test suites before starting.
  --no-open  Do not open a browser automatically.

Environment:
  API_PORT   API port (default: 3001)
  WEB_PORT   storefront port (default: 5173)
  NO_OPEN=1  same as --no-open
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --verify) VERIFY=1 ;;
        --no-open) NO_OPEN=1 ;;
        -h|--help|help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

for port_name in API_PORT WEB_PORT; do
    port_value="${!port_name}"
    if ! [[ "$port_value" =~ ^(0|[1-9][0-9]{0,4})$ ]] \
       || (( 10#$port_value < 1 || 10#$port_value > 65535 )); then
        echo "$port_name must be an integer between 1 and 65535" >&2
        exit 2
    fi
done
if [ "$API_PORT" = "$WEB_PORT" ]; then
    echo "API_PORT and WEB_PORT must be different" >&2
    exit 2
fi

command -v node >/dev/null 2>&1 || {
    echo "Node.js 22.22.2+, 24.15.0+, or 26+ is required" >&2
    exit 1
}
command -v npm >/dev/null 2>&1 || { echo "npm is required" >&2; exit 1; }
node -e '
  const [major, minor, patch] = process.versions.node.split(".").map(Number);
  const atLeast = (wantedMajor, wantedMinor, wantedPatch) =>
    major === wantedMajor &&
    (minor > wantedMinor || (minor === wantedMinor && patch >= wantedPatch));
  const supported = atLeast(22, 22, 2) || atLeast(24, 15, 0) || major >= 26;
  process.exit(supported ? 0 : 1);
' || {
    echo "Node.js 22.22.2+ on 22.x, 24.15.0+ on 24.x, or 26+ is required; found $(node --version)" >&2
    exit 1
}

port_available() {
    node -e '
      const net = require("node:net");
      const server = net.createServer();
      server.once("error", () => process.exit(1));
      server.listen(Number(process.argv[1]), "127.0.0.1", () => {
        server.close(() => process.exit(0));
      });
    ' "$1" >/dev/null 2>&1
}

for port_name in API_PORT WEB_PORT; do
    port_value="${!port_name}"
    if ! port_available "$port_value"; then
        echo "$port_name=$port_value is already in use on 127.0.0.1" >&2
        exit 2
    fi
done

echo "== installing locked application dependencies"
npm --prefix "$API_DIR" ci
npm --prefix "$WEB_DIR" ci

if [ "$VERIFY" = "1" ]; then
    echo "== verifying API and storefront"
    npm --prefix "$API_DIR" test
    npm --prefix "$WEB_DIR" test
fi

echo "== building API and storefront"
npm --prefix "$API_DIR" run build
npm --prefix "$WEB_DIR" run build

api_pid=""
web_pid=""
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for child_pid in "$api_pid" "$web_pid"; do
        if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
            kill "$child_pid" 2>/dev/null || true
        fi
    done
    for attempt in {1..20}; do
        children_running=0
        for child_pid in "$api_pid" "$web_pid"; do
            if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
                children_running=1
            fi
        done
        [ "$children_running" = "0" ] && break
        sleep 0.1
    done
    for child_pid in "$api_pid" "$web_pid"; do
        if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
            kill -KILL "$child_pid" 2>/dev/null || true
        fi
    done
    for child_pid in "$api_pid" "$web_pid"; do
        if [ -n "$child_pid" ]; then
            wait "$child_pid" 2>/dev/null || true
        fi
    done
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "== starting local services"
PORT="$API_PORT" HOST=127.0.0.1 node "$API_DIR/dist/index.js" &
api_pid=$!
(
    cd "$WEB_DIR"
    exec node node_modules/vite/bin/vite.js preview \
        --host 127.0.0.1 --port "$WEB_PORT" --strictPort
) &
web_pid=$!

probe_url() {
    node -e '
      const http = require("node:http");
      const expected = process.argv[2];
      const request = http.get(process.argv[1], (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          if (body.length < 8192) body += chunk;
        });
        response.on("end", () => {
          if (response.statusCode !== 200) process.exit(1);
          if (expected === "api") {
            try {
              process.exit(JSON.parse(body).status === "ok" ? 0 : 1);
            } catch {
              process.exit(1);
            }
          }
          process.exit(body.includes("<title>MODA —") ? 0 : 1);
        });
      });
      request.setTimeout(250, () => request.destroy());
      request.on("error", () => process.exit(1));
    ' "$1" "$2" >/dev/null 2>&1
}

wait_for_url() {
    label="$1"
    url="$2"
    child_pid="$3"
    expected="$4"
    for attempt in {1..40}; do
        if ! kill -0 "$child_pid" 2>/dev/null; then
            echo "$label stopped before becoming ready" >&2
            return 1
        fi
        if probe_url "$url" "$expected" && kill -0 "$child_pid" 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    echo "$label did not become ready at $url" >&2
    return 1
}

API_URL="http://127.0.0.1:$API_PORT"
WEB_URL="http://127.0.0.1:$WEB_PORT"
wait_for_url "Moda API" "$API_URL/health" "$api_pid" api
wait_for_url "Moda storefront" "$WEB_URL/" "$web_pid" web

if ! kill -0 "$api_pid" 2>/dev/null || ! kill -0 "$web_pid" 2>/dev/null; then
    echo "A Moda service stopped during readiness checks" >&2
    exit 1
fi

echo
echo "FinX-Moda is ready:"
echo "  Storefront  $WEB_URL"
echo "  API health  $API_URL/health"
echo "  Stop both with Ctrl+C"
echo

if [ "$NO_OPEN" != "1" ]; then
    if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
        open "$WEB_URL" >/dev/null 2>&1 || true
    elif [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v xdg-open >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 5s xdg-open "$WEB_URL" >/dev/null 2>&1 || true
        else
            xdg-open "$WEB_URL" >/dev/null 2>&1 &
        fi
    fi
fi

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
    sleep 1
done

echo "A Moda service stopped unexpectedly" >&2
exit 1
