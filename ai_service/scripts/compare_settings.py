"""실제 영상 1개로 Window 길이·신뢰도 기준 조합을 비교한다 (팀 모델 설정 고르기용).

관절 추출(YOLO + ByteTrack, POSE_REFINER=mediapipe_heavy면 MediaPipe heavy)은 1번만 하고,
그 결과로 여러 설정을 다시 집계하므로 빠르다.

사용 (컨테이너 안):
    docker compose cp <영상.mp4> fastapi:/tmp/test.mp4
    docker compose exec fastapi python scripts/compare_settings.py /tmp/test.mp4 --expected side_lying_leg_raise
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from services.action_inference import ActionInferenceSession  # noqa: E402
from services.model_loader import load_bundle  # noqa: E402
from services.pose_pipeline import YoloPoseTracker, iter_observations, probe_video  # noqa: E402
from services.settings import settings  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--windows", type=float, nargs="+", default=[2.5, 3.0, 4.0, 5.0])
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.45, 0.55])
    p.add_argument("--expected", default="", help="영상 속 실제 동작 Label (있으면 정답 여부 표시)")
    args = p.parse_args()

    bundle = load_bundle(settings)
    tracker = YoloPoseTracker(bundle)
    video = Path(args.video)
    meta = probe_video(video, settings.max_video_seconds, 0.0)
    started = time.perf_counter()
    observations = list(iter_observations(video, meta, tracker, settings.target_process_fps, settings.max_video_seconds))
    print(f"영상 {video.name}: {meta.duration_sec:.1f}초, 처리 Frame {len(observations)}, "
          f"관절 추출 {time.perf_counter() - started:.1f}초, {tracker.stats()}")
    print(f"모델 {bundle.model_file} classes={bundle.classes}\n")
    print(f"{'window':>6} {'threshold':>9} | {'track':>5} {'결과':<24} {'conf':>5} {'득표율':>6} {'windows':>7} {'정답':>4}")
    print("-" * 80)
    for window in args.windows:
        for threshold in args.thresholds:
            cfg = {**bundle.runtime_config, "window_seconds": window, "confidence_threshold": threshold}
            session = ActionInferenceSession(replace(bundle, runtime_config=cfg),
                                             settings.min_windows_per_track, settings.min_vote_ratio)
            for obs in observations:
                session.update(obs)
            predictions, dropped = session.summarize()
            if not predictions:
                print(f"{window:>6} {threshold:>9} | {'-':>5} {'(결과 없음: Window 부족)':<24}")
            for pr in predictions:
                mark = ("O" if pr["action_label"] == args.expected else "X") if args.expected else ""
                print(f"{window:>6} {threshold:>9} | {pr['track_id']:>5} {pr['action_label']:<24} "
                      f"{pr['confidence']:>5.2f} {pr['vote_ratio']:>6.2f} {pr['window_count']:>7} {mark:>4}")
    print("\n고른 값은 .env 의 WINDOW_SECONDS, CONFIDENCE_THRESHOLD 에 넣고 `docker compose up -d`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
