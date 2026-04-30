#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
  echo "未找到 .venv，请先在项目根目录创建虚拟环境并安装依赖。"
  exit 1
fi

source "$PROJECT_ROOT/.venv/bin/activate"

python -m honest_ear.download_models "$@"
