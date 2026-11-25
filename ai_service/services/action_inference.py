"""Track ID별 Sliding Window 추론과 '작업 1건 = Track별 결과 1행' 집계.

- Window 관리·예측·안정화는 7·8일차 코드(MultiPersonStateManager, predict_from_window)를 그대로 쓴다.
- 49번 스크립트는 예측 주기마다 라벨을 화면에 그렸지만, 웹 서비스는 DB에 Track당 1행을 저장해야 하므로
  Window별 예측 이력을 모아 다수결로 대표 동작을 정한다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.multi_person_utils import MultiPersonStateManager
from src.realtime_utils import predict_from_window, validate_runtime_values

from .model_loader import ModelBundle
from .pose_pipeline import FrameObservation

UNCERTAIN_LABEL = "UNCERTAIN"


@dataclass
class WindowPrediction:
    timestamp: float
    raw_label: str
    threshold_label: str  # confidence_threshold 미만이면 UNKNOWN
    stable_label: str
    confidence: float


@dataclass
class TrackHistory:
    track_id: int
    first_seen: float
    last_seen: float
    observed_frames: int = 0
    valid_feature_frames: int = 0
    windows: list[WindowPrediction] = field(default_factory=list)


class ActionInferenceSession:
    """영상 1건 동안 유지되는 추론 상태. 요청마다 새로 만든다 (요청 간 상태 공유 없음)."""

    def __init__(self, bundle: ModelBundle, min_windows: int, min_vote_ratio: float):
        self.bundle = bundle
        self.min_windows = int(min_windows)
        self.min_vote_ratio = float(min_vote_ratio)
        cfg = bundle.runtime_config

        validate_runtime_values(
            window_seconds=float(cfg["window_seconds"]),
            confidence_threshold=float(cfg["confidence_threshold"]),
            stabilization_buffer=int(cfg["stabilization_buffer"]),
            stabilization_min_votes=int(cfg["stabilization_min_votes"]),
        )
        self.state_manager = MultiPersonStateManager(
            window_seconds=float(cfg["window_seconds"]),
            stabilization_buffer=int(cfg["stabilization_buffer"]),
            stabilization_min_votes=int(cfg["stabilization_min_votes"]),
            lost_timeout_seconds=float(bundle.tracking_config["lost_timeout_seconds"]),
        )
        self.histories: dict[int, TrackHistory] = {}
        self.processed_frames = 0
        self.frames_with_person = 0
        self.max_simultaneous_persons = 0

    # ------------------------------------------------------------------ 프레임 단위
    def update(self, obs: FrameObservation) -> None:
        cfg = self.bundle.runtime_config
        now = obs.timestamp
        seen_ids: set[int] = set()
        self.processed_frames += 1
        if obs.persons:
            self.frames_with_person += 1
            self.max_simultaneous_persons = max(self.max_simultaneous_persons, len(obs.persons))

        for person in obs.persons:
            track_id = int(person.track_id)
            seen_ids.add(track_id)
            history = self.histories.setdefault(track_id, TrackHistory(track_id, now, now))
            history.last_seen = now
            history.observed_frames += 1
            if person.feature is not None:
                history.valid_feature_frames += 1

            state = self.state_manager.get_or_create(track_id, now)
            state.feature_buffer.append(now, person.feature)

            ready = state.feature_buffer.is_ready(
                min_window_fill_ratio=float(cfg["min_window_fill_ratio"]),
                min_valid_feature_ratio=float(cfg["min_valid_feature_ratio"]),
                min_valid_features=int(cfg["min_valid_features"]),
            )
            due = (now - state.last_prediction_time) >= float(cfg["prediction_interval_sec"])
            if not (ready and due):
                continue

            threshold_label, raw_label, confidence, _ = predict_from_window(
                model=self.bundle.action_model,
                feature_buffer=state.feature_buffer,
                sequence_steps=self.bundle.sequence_steps,
                confidence_threshold=self.bundle.confidence_threshold,
            )
            state.raw_label = raw_label
            state.threshold_label = threshold_label
            state.confidence = float(confidence)
            state.stable_label = state.stabilizer.update(threshold_label)
            state.last_prediction_time = now

            history.windows.append(
                WindowPrediction(
                    timestamp=now,
                    raw_label=str(raw_label),
                    threshold_label=str(threshold_label),
                    stable_label=str(state.stable_label),
                    confidence=float(confidence),
                )
            )

        # 안 보인 사람은 Buffer에 빈 칸을 넣고, lost_timeout 이 지나면 상태 삭제 (8일차와 동일)
        self.state_manager.mark_missing(now, seen_ids)
        self.state_manager.remove_lost(now)

    # ------------------------------------------------------------------ 작업 단위 집계
    def summarize(self) -> tuple[list[dict], list[dict]]:
        """(predictions, dropped_tracks) 반환."""
        predictions: list[dict] = []
        dropped: list[dict] = []
        classes = set(self.bundle.classes)

        for track_id in sorted(self.histories):
            history = self.histories[track_id]
            windows = history.windows
            if len(windows) < self.min_windows:
                dropped.append(
                    {
                        "track_id": track_id,
                        "window_count": len(windows),
                        "reason": "예측 Window 수 부족 (짧게 검출되었거나 ID Switch로 생긴 조각일 수 있음)",
                        "first_seen_sec": round(history.first_seen, 3),
                        "last_seen_sec": round(history.last_seen, 3),
                    }
                )
                continue

            confident = [w for w in windows if w.threshold_label in classes]
            raw_votes = Counter(w.raw_label for w in windows)
            raw_top = raw_votes.most_common(1)[0][0]

            if confident:
                votes = Counter(w.threshold_label for w in confident)
                # 동률이면 평균 confidence가 높은 라벨
                label = max(
                    votes,
                    key=lambda lab: (
                        votes[lab],
                        sum(w.confidence for w in confident if w.threshold_label == lab) / votes[lab],
                    ),
                )
                chosen = [w for w in confident if w.threshold_label == label]
                confidence = sum(w.confidence for w in chosen) / len(chosen)
                vote_ratio = len(chosen) / len(windows)
                is_uncertain = vote_ratio < self.min_vote_ratio
            else:
                # 모든 Window가 confidence_threshold 미만 → 판단 불확실
                label = UNCERTAIN_LABEL
                confidence = sum(w.confidence for w in windows) / len(windows)
                vote_ratio = 0.0
                is_uncertain = True

            predictions.append(
                {
                    "track_id": track_id,
                    "action_label": label,
                    "confidence": round(float(confidence), 4),
                    "raw_label": raw_top,
                    "window_count": len(windows),
                    "vote_ratio": round(float(vote_ratio), 4),
                    "is_uncertain": bool(is_uncertain),
                    "first_seen_sec": round(history.first_seen, 3),
                    "last_seen_sec": round(history.last_seen, 3),
                    "valid_feature_ratio": round(
                        history.valid_feature_frames / max(1, history.observed_frames), 4
                    ),
                }
            )
        return predictions, dropped


# ---------------------------------------------------------------------- 영상 1건 처리 순서
def analyze_video(path, bundle: ModelBundle, tracker, settings) -> dict:
    """영상 1건: 검사 → Tracker 초기화 → Frame 반복 → Track별 집계 → 응답 dict.

    사람 없음은 정상 응답(빈 predictions + message), 손상 영상은 VideoReadError로 구분한다.
    """
    import time

    from .pose_pipeline import iter_observations, probe_video

    started = time.perf_counter()
    # Window 1개를 채우는 데 필요한 최소 길이 (day07 설정: window_seconds × min_window_fill_ratio)
    cfg = bundle.runtime_config
    min_seconds = float(cfg["window_seconds"]) * float(cfg["min_window_fill_ratio"])
    meta = probe_video(path, settings.max_video_seconds, min_seconds)

    session = ActionInferenceSession(
        bundle,
        min_windows=settings.min_windows_per_track,
        min_vote_ratio=settings.min_vote_ratio,
    )
    for obs in iter_observations(
        path, meta, tracker, settings.target_process_fps, settings.max_video_seconds
    ):
        session.update(obs)

    if session.processed_frames == 0:
        from .pose_pipeline import VideoReadError

        raise VideoReadError("처리할 수 있는 Frame이 없습니다.")

    predictions, dropped = session.summarize()
    person_detected = session.frames_with_person > 0
    if not person_detected:
        message = "영상에서 사람을 찾지 못했습니다."
    elif not predictions:
        message = "사람은 검출됐지만 동작을 판단할 만큼 충분히 추적되지 않았습니다. 더 길게(5초 이상) 촬영해 주세요."
    else:
        message = f"{len(predictions)}명의 동작을 예측했습니다."

    return {
        "status": "ok",
        "message": message,
        "person_detected": person_detected,
        "predictions": predictions,
        "dropped_tracks": dropped,
        "video": meta.as_dict(),
        "stats": {
            "processed_frames": session.processed_frames,
            "frames_with_person": session.frames_with_person,
            "max_simultaneous_persons": session.max_simultaneous_persons,
            "unique_track_ids": len(session.histories),
            **(tracker.stats() if hasattr(tracker, "stats") else {}),
        },
        "model": {
            "name": "rf_action_baseline",
            "classes": bundle.classes,
            "sequence_steps": bundle.sequence_steps,
            "confidence_threshold": bundle.confidence_threshold,
        },
        "processing_time_sec": round(time.perf_counter() - started, 3),
    }
