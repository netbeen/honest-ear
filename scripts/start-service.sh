#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_HOST="${HONEST_EAR_HOST:-127.0.0.1}"
DEFAULT_PORT="${HONEST_EAR_PORT:-8000}"

find_port_pids() {
  # Returns PIDs currently listening on the target TCP port.
  lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

wait_for_port_release() {
  # Waits briefly for the target TCP port to stop being listened on.
  local port="$1"
  local attempts=20

  while [ "$attempts" -gt 0 ]; do
    if [ -z "$(find_port_pids "$port")" ]; then
      return 0
    fi
    sleep 0.2
    attempts=$((attempts - 1))
  done

  return 1
}

kill_port_processes() {
  # Stops any process listening on the target TCP port before uvicorn starts.
  local port="$1"
  local pids

  pids="$(find_port_pids "$port")"
  if [ -z "$pids" ]; then
    return 0
  fi

  echo "检测到端口 $port 已被占用，正在停止相关进程: $pids"
  echo "$pids" | xargs kill

  if wait_for_port_release "$port"; then
    return 0
  fi

  pids="$(find_port_pids "$port")"
  if [ -n "$pids" ]; then
    echo "端口 $port 仍未释放，强制停止相关进程: $pids"
    echo "$pids" | xargs kill -9
    wait_for_port_release "$port" || {
      echo "端口 $port 仍未释放，请手动检查占用进程。"
      exit 1
    }
  fi
}

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "未找到 $PROJECT_ROOT/.venv，请先创建虚拟环境并安装依赖。"
  exit 1
fi

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
else
  echo "未找到 .env，将使用当前 shell 环境变量启动服务。"
fi

kill_port_processes "$DEFAULT_PORT"

exec python -m uvicorn honest_ear.api:app \
  --app-dir "$PROJECT_ROOT/src" \
  --host "$DEFAULT_HOST" \
  --port "$DEFAULT_PORT" \
  --reload \
  "$@"
