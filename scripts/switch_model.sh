#!/usr/bin/env bash
# 사용할 모델 전환 (재빌드 없이 .env 만 바꾸고 재시작)
#   bash scripts/switch_model.sh basic   # 과제 6동작 (모델 v2, YOLO 관절) — 평가 기준 시연
#   bash scripts/switch_model.sh team    # 팀 재활 3동작 (팀원 모델, MediaPipe heavy 관절)
#   bash scripts/switch_model.sh team6   # 팀 재활 6동작 (확장 실험 모델)
#   bash scripts/switch_model.sh day06   # 6일차 원본 모델 그대로 (재학습 없음, 과제 기본 원칙 확인용)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"
case "$MODE" in
  team)
    # 팀원이 학습한 3동작 모델. apply_team6.sh 가 team/backup_3actions 로 옮겨 둠
    T=ai_service/models/team/backup_3actions
    for f in rf_action_team.joblib rf_feature_columns.json rf_training_info.json; do
      [ -f "$T/$f" ] || { echo "[오류] $T/$f 없음. 팀원 3동작 모델 3개 파일을 이 폴더에 두세요." >&2; exit 1; }
    done
    VALUES="ACTION_SET=team MODEL_FILE=team/backup_3actions/rf_action_team.joblib POSE_REFINER=mediapipe_heavy WINDOW_SECONDS=3 CONFIDENCE_THRESHOLD=0.55" ;;
  team6) VALUES="ACTION_SET=team6 MODEL_FILE=team/rf_action_team.joblib POSE_REFINER=mediapipe_heavy WINDOW_SECONDS=3 CONFIDENCE_THRESHOLD=0.55" ;;
  basic) VALUES="ACTION_SET=basic MODEL_FILE=rf_action_baseline.joblib POSE_REFINER=none WINDOW_SECONDS=2.5 CONFIDENCE_THRESHOLD=" ;;
  day06)
    # 6일차 산출물 3개를 수정 없이 사용. 7일차 실시간 Window(5초)·기준(0.55) 그대로.
    D=ai_service/models/backup_day06
    SRC="${SUBJECT10_DIR:-$HOME/ai_vision/subject10_pose_action}/artifacts/day06"
    if [ ! -f "$D/rf_action_baseline.joblib" ] && [ -f "$SRC/rf_action_baseline.joblib" ]; then
      mkdir -p "$D" && cp "$SRC"/rf_action_baseline.joblib "$SRC"/rf_feature_columns.json "$SRC"/rf_training_info.json "$D/"
      echo "6일차 산출물 복사: $SRC → $D"
    fi
    for f in rf_action_baseline.joblib rf_feature_columns.json rf_training_info.json; do
      [ -f "$D/$f" ] || { echo "[오류] $D/$f 없음. subject10의 artifacts/day06 에서 복사하세요." >&2; exit 1; }
    done
    VALUES="ACTION_SET=basic MODEL_FILE=backup_day06/rf_action_baseline.joblib POSE_REFINER=none WINDOW_SECONDS=5 CONFIDENCE_THRESHOLD=" ;;
  *) echo "사용법: bash scripts/switch_model.sh basic|day06|team|team6" >&2; exit 1 ;;
esac
touch .env
for kv in $VALUES; do
  key="${kv%%=*}"
  if grep -q "^${key}=" .env; then sed -i "s|^${key}=.*|${kv}|" .env; else echo "$kv" >> .env; fi
done
echo "[.env] $VALUES"
docker compose up -d
for i in $(seq 1 30); do
  if docker compose ps fastapi --format '{{.Health}}' 2>/dev/null | grep -q healthy; then break; fi
  sleep 5
done
docker compose exec fastapi python scripts/check_artifacts.py --with-pose
