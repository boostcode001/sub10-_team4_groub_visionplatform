#!/usr/bin/env bash
# GitHub push 전 점검: 올라가면 안 되는 파일이 staging 후보에 있는지 확인
#   bash scripts/check_before_push.sh
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse --git-dir >/dev/null 2>&1 || { echo "git init 먼저 실행하세요 (git init -b main)"; exit 1; }

BAD=$(git add -A --dry-run | sed "s/^add '//; s/'$//" \
  | grep -E '(^|/)\.env$|\.joblib$|\.pt$|\.task$|\.onnx$|\.mp4$|\.mov$|\.avi$|\.mkv$|\.zip$|\.csv$|(^|/)media/|db\.sqlite3$' || true)
if [ -n "$BAD" ]; then
  echo "[STOP] 올리면 안 되는 파일이 포함됩니다:"; echo "$BAD"; exit 1
fi

BIG=$(git add -A --dry-run | sed "s/^add '//; s/'$//" | while read -r f; do
  [ -f "$f" ] && [ "$(stat -c %s "$f")" -gt 5242880 ] && echo "$f ($(du -h "$f" | cut -f1))"; done || true)
if [ -n "$BIG" ]; then
  echo "[STOP] 5MB 넘는 파일:"; echo "$BIG"; exit 1
fi

# 실제 카메라 IP가 코드·문서에 남았는지 (예시 주소 192.168.0.XX 는 허용)
LEAK=$(git add -A --dry-run | sed "s/^add '//; s/'$//" | xargs -r grep -lE '192\.168\.[0-9]+\.[0-9]+:8080' 2>/dev/null \
  | grep -v -E '^backend/analyses/tests.py$' || true)
if [ -n "$LEAK" ]; then
  echo "[확인] 실제 카메라 IP처럼 보이는 주소가 있습니다 (예시 주소면 무시):"; echo "$LEAK"
fi

N=$(git add -A --dry-run | wc -l)
echo "[OK] 올라갈 파일 ${N}개. 모델·영상·.env 없음."
