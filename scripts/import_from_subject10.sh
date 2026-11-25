#!/usr/bin/env bash
# 기존 subject10_pose_action 프로젝트에서 '재학습 없이' 재사용할 파일을 가져온다.
#
# 사용 (WSL):
#   bash scripts/import_from_subject10.sh ~/ai_vision/subject10_pose_action
#
# 가져오는 것
#   src/                      → ai_service/src/         (pose·feature·sequence·realtime·multi_person·common 전체)
#   configs/*.json            → ai_service/configs/     (project.json, day07_realtime.json, day08_multi_person.json)
#   artifacts/day06/rf_*      → ai_service/models/      (joblib + feature_columns + training_info)
#   project.json의 pose_model → ai_service/models/      (YOLO Pose 가중치 .pt)
set -euo pipefail

SRC_ROOT="${1:-$HOME/ai_vision/subject10_pose_action}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
AI="$HERE/ai_service"

if [ ! -d "$SRC_ROOT/src" ]; then
  echo "[오류] $SRC_ROOT/src 가 없습니다. 기존 프로젝트 경로를 인자로 주세요." >&2
  exit 1
fi

mkdir -p "$AI/src" "$AI/configs" "$AI/models"

echo "== 1) src/ 복사 (import 관계를 깨지 않도록 폴더 전체)"
cp -r "$SRC_ROOT/src/." "$AI/src/"
find "$AI/src" -name "__pycache__" -type d -prune -exec rm -rf {} +
touch "$AI/src/__init__.py"
ls "$AI/src"

echo "== 2) 설정 JSON 복사"
for name in day07_realtime.json day08_multi_person.json; do
  cp -v "$SRC_ROOT/configs/$name" "$AI/configs/"
done
if   [ -f "$SRC_ROOT/configs/project.json" ]; then cp -v "$SRC_ROOT/configs/project.json" "$AI/configs/"
elif [ -f "$SRC_ROOT/project.json" ];         then cp -v "$SRC_ROOT/project.json" "$AI/configs/"
else echo "[오류] project.json 을 찾지 못했습니다." >&2; exit 1
fi

echo "== 3) 6일차 모델 산출물 복사 (재학습 없음)"
for name in rf_action_baseline.joblib rf_feature_columns.json rf_training_info.json; do
  cp -v "$SRC_ROOT/artifacts/day06/$name" "$AI/models/"
done

echo "== 4) YOLO Pose 가중치 복사"
POSE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig'))['pose_model'])" "$AI/configs/project.json")
echo "project.json pose_model = $POSE"
FOUND=""
for cand in "$POSE" "$SRC_ROOT/$POSE" "$SRC_ROOT/$(basename "$POSE")" "$SRC_ROOT/models/$(basename "$POSE")" "$SRC_ROOT/weights/$(basename "$POSE")"; do
  if [ -f "$cand" ]; then FOUND="$cand"; break; fi
done
if [ -z "$FOUND" ]; then
  FOUND=$(find "$SRC_ROOT" -name "$(basename "$POSE")" -not -path "*/.venv/*" | head -n 1 || true)
fi
if [ -n "$FOUND" ]; then
  cp -v "$FOUND" "$AI/models/"
else
  echo "[주의] $(basename "$POSE") 파일을 찾지 못했습니다. 공식 가중치 이름이면 컨테이너가 처음 실행될 때 내려받습니다."
fi

echo "== 5) .env 에 넣을 버전 (학습 환경과 일치시키기)"
if [ -x "$SRC_ROOT/.venv/bin/python" ]; then
  "$SRC_ROOT/.venv/bin/python" - <<'PY'
import platform
print(f"PYTHON_VERSION={'.'.join(platform.python_version_tuple()[:2])}")
for pkg, key in (("sklearn", "SKLEARN_VERSION"), ("ultralytics", "ULTRALYTICS_VERSION")):
    try:
        mod = __import__(pkg)
        print(f"{key}={mod.__version__}")
    except ImportError:
        print(f"# {pkg} 미설치")
PY
else
  echo "(.venv 를 찾지 못함: subject10 가상환경에서 pip show scikit-learn ultralytics 로 확인)"
fi

echo
echo "완료. 다음: cd ai_service && python scripts/check_artifacts.py   (T01)"
