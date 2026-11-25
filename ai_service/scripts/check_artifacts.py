"""T01 모델 불러오기 테스트.

사용:
    (로컬)    cd ai_service && python scripts/check_artifacts.py
    (컨테이너) docker compose exec fastapi python scripts/check_artifacts.py --with-pose

확인 항목: 6개 Class, 1,140 Feature(= sequence_steps × 38), Feature 순서 JSON 길이, 설정 파일 필수 키.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from services.model_loader import ACTION_SETS, ArtifactError, load_bundle  # noqa: E402
from services.settings import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-pose", action="store_true", help="YOLO Pose 가중치까지 로드")
    args = parser.parse_args()

    print(f"MODEL_DIR  = {settings.model_dir}")
    print(f"CONFIG_DIR = {settings.config_dir}")
    try:
        bundle = load_bundle(settings)
    except ArtifactError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] {bundle.model_file} 로드")
    print(f"     classes          : {bundle.classes}")
    print(f"     train_samples    : {bundle.training_info.get('train_samples', bundle.training_info.get('train_clips'))}")
    print(f"     sequence_steps   : {bundle.sequence_steps}")
    print(f"     n_features_in_   : {bundle.feature_count}  (= {bundle.sequence_steps} × 38)")
    print(f"     feature_columns  : {len(bundle.feature_columns)}개, 첫 3개 {bundle.feature_columns[:3]}")
    print(f"     threshold        : {bundle.confidence_threshold}")
    print(f"     tracker          : {bundle.tracking_config['tracker']}")
    print(f"     pose_model       : {bundle.pose_model_path} (device={bundle.device})")
    print(f"     model_file       : {bundle.model_file} ({bundle.training_info.get('model_version', '-')})")
    print(f"     window_seconds   : {bundle.runtime_config['window_seconds']}")
    print(f"     action_set       : {settings.action_set} / pose_refiner: {settings.pose_refiner}")
    for label, code in sorted(ACTION_SETS.get(settings.action_set, {}).items(), key=lambda kv: kv[1]):
        mark = "OK" if label in bundle.classes else "MISSING"
        print(f"     {code} {label:<24} {mark}")
    for warning in bundle.warnings:
        print(f"[WARN] {warning}")

    if args.with_pose:
        from services.pose_pipeline import YoloPoseTracker

        tracker = YoloPoseTracker(bundle)
        print("[OK] YOLO Pose 가중치 로드")
        if tracker.refiner is not None:
            print(f"[OK] MediaPipe heavy 로드 ({settings.model_dir / settings.heavy_model_file})")

    print("T01 PASS" if not bundle.warnings else "T01 PASS (경고 확인 필요)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
