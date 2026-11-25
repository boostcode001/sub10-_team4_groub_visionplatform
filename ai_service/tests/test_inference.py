from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakePoseTracker, write_video


# ------------------------------------------------------------------ T01 모델 불러오기
def test_t01_bundle_loads_with_6_classes_and_1140_features(test_settings):
    from services.model_loader import load_bundle

    bundle = load_bundle(test_settings)
    assert len(bundle.classes) == 6
    assert bundle.feature_count == 1140 == bundle.sequence_steps * 38
    assert bundle.warnings == []


def test_feature_count_mismatch_is_rejected(test_settings, tmp_path):
    import shutil

    from services.model_loader import ArtifactError, load_bundle
    from services.settings import Settings

    bad_models = tmp_path / "models"
    shutil.copytree(test_settings.model_dir, bad_models)
    info = json.loads((bad_models / "rf_training_info.json").read_text())
    info["sequence_steps"] = 20  # 1,140 ≠ 20 × 38
    (bad_models / "rf_training_info.json").write_text(json.dumps(info))

    with pytest.raises(ArtifactError, match="sequence_steps"):
        load_bundle(Settings(model_dir=bad_models, config_dir=test_settings.config_dir))


# ------------------------------------------------------------------ 추론 파이프라인
def _analyze(test_settings, video, scenario):
    from services.action_inference import analyze_video
    from services.model_loader import load_bundle

    bundle = load_bundle(test_settings)
    tracker = FakePoseTracker(bundle, scenario)
    return analyze_video(video, bundle, tracker, test_settings), tracker


def test_t04_two_people_get_separate_predictions(test_settings, sample_video):
    result, tracker = _analyze(test_settings, sample_video, "two_people")
    by_id = {p["track_id"]: p for p in result["predictions"]}
    assert set(by_id) == {1, 2}
    assert by_id[1]["action_label"] == "wave"
    assert by_id[2]["action_label"] == "sit_stand"
    assert all(p["window_count"] >= 2 for p in result["predictions"])
    assert result["person_detected"] is True
    assert result["stats"]["max_simultaneous_persons"] == 2
    assert tracker.reset_count == 1  # 영상마다 Tracker 초기화


def test_no_person_returns_empty_predictions_not_error(test_settings, sample_video):
    result, _ = _analyze(test_settings, sample_video, "nobody")
    assert result["predictions"] == []
    assert result["person_detected"] is False
    assert "사람을 찾지 못했습니다" in result["message"]


def test_short_track_fragment_is_dropped(test_settings, sample_video):
    result, _ = _analyze(test_settings, sample_video, "blip")
    assert {p["track_id"] for p in result["predictions"]} == {1, 2}
    assert [d["track_id"] for d in result["dropped_tracks"]] == [7]


def test_timestamps_use_video_time_not_wall_clock(test_settings, sample_video):
    result, _ = _analyze(test_settings, sample_video, "two_people")
    # 6초 영상이면 마지막 관측 시각이 약 6초 (처리 속도와 무관)
    last = max(p["last_seen_sec"] for p in result["predictions"])
    assert 5.5 <= last <= 6.0


# ------------------------------------------------------------------ API 계약
@pytest.fixture()
def client(test_settings, monkeypatch):
    import routers.predictions as predictions_module
    from main import app
    from services.runtime import runtime

    monkeypatch.setattr(predictions_module, "settings", test_settings)
    runtime.bundle = runtime.tracker = None
    runtime.load(test_settings, tracker_factory=lambda b: FakePoseTracker(b, "two_people"))
    assert runtime.ready, runtime.load_error
    with TestClient(app) as c:
        yield c
    runtime.bundle = runtime.tracker = None


def test_health_reports_model(client):
    body = client.get("/health").json()
    assert body["model_ready"] is True
    assert body["model"]["feature_count"] == 1140


def test_predict_video_ok_and_ids_restart_per_request(client, sample_video):
    for _ in range(2):
        with sample_video.open("rb") as f:
            res = client.post("/predict/video", files={"file": ("a.mp4", f, "video/mp4")})
        assert res.status_code == 200, res.text
        assert sorted(p["track_id"] for p in res.json()["predictions"]) == [1, 2]


def test_corrupt_video_is_422(client, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a real video" * 100)
    with bad.open("rb") as f:
        res = client.post("/predict/video", files={"file": ("broken.mp4", f, "video/mp4")})
    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "VIDEO_UNREADABLE"


def test_wrong_extension_is_400(client, tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("hello")
    with txt.open("rb") as f:
        res = client.post("/predict/video", files={"file": ("a.txt", f, "text/plain")})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "UNSUPPORTED_FORMAT"


def test_too_long_video_is_422(client, tmp_path, test_settings, monkeypatch):
    import routers.predictions as predictions_module
    from dataclasses import replace

    monkeypatch.setattr(predictions_module, "settings", replace(test_settings, max_video_seconds=3.0))
    video = write_video(tmp_path / "long.mp4", seconds=5)
    with video.open("rb") as f:
        res = client.post("/predict/video", files={"file": ("long.mp4", f, "video/mp4")})
    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "VIDEO_TOO_LONG"


def test_model_not_ready_is_503(client, sample_video):
    from services.runtime import runtime

    runtime.tracker = None
    runtime.load_error = "모델 파일 없음"
    with sample_video.open("rb") as f:
        res = client.post("/predict/video", files={"file": ("a.mp4", f, "video/mp4")})
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_READY"


def test_too_short_video_is_422(client, tmp_path):
    # day07 설정 window 2.0초 × fill 0.8 = 1.6초 미만이면 Window를 만들 수 없음
    video = write_video(tmp_path / "short.mp4", seconds=1)
    with video.open("rb") as f:
        res = client.post("/predict/video", files={"file": ("short.mp4", f, "video/mp4")})
    assert res.status_code == 422
    assert res.json()["detail"]["code"] == "VIDEO_TOO_SHORT"


def test_guide_url_alias_works(client, sample_video):
    # 9~10일차 개발 가이드 권장 주소
    with sample_video.open("rb") as f:
        res = client.post("/api/v1/predict/", files={"file": ("a.mp4", f, "video/mp4")})
    assert res.status_code == 200
    assert sorted(p["track_id"] for p in res.json()["predictions"]) == [1, 2]


def test_window_seconds_override(test_settings):
    from dataclasses import replace

    from services.model_loader import load_bundle

    bundle = load_bundle(replace(test_settings, window_seconds_override=2.5))
    assert bundle.runtime_config["window_seconds"] == 2.5
    assert load_bundle(test_settings).runtime_config["window_seconds"] == 2.0  # 덮어쓰지 않으면 설정 파일 값


# ------------------------------------------------------------------ 팀 모델용 MediaPipe heavy 관절 재추출
def _fake_refiner(landmarks):
    from types import SimpleNamespace

    from services.keypoint_refiner import HeavyKeypointRefiner

    refiner = HeavyKeypointRefiner.__new__(HeavyKeypointRefiner)
    refiner._mp = SimpleNamespace(Image=lambda image_format, data: data, ImageFormat=SimpleNamespace(SRGB=1))
    refiner._landmarker = SimpleNamespace(detect=lambda image: SimpleNamespace(pose_landmarks=landmarks))
    refiner.pad = 0.15
    refiner.refined = refiner.fallback = 0
    return refiner


def test_heavy_refiner_maps_crop_coordinates_to_frame():
    from types import SimpleNamespace

    import numpy as np

    from services.keypoint_refiner import MP_TO_COCO

    # 33개 landmark 중 i번은 crop 기준 (i/100, 0.5) 위치
    person = [SimpleNamespace(x=i / 100, y=0.5) for i in range(33)]
    refiner = _fake_refiner([person])
    frame = np.zeros((480, 640, 3), np.uint8)
    box = np.array([100, 100, 300, 400], np.float32)  # 15% 여백 → crop (70,55)-(330,445)
    out = refiner.refine(frame, box, np.zeros((17, 2), np.float32))
    assert out.shape == (17, 2)
    nose_idx = MP_TO_COCO[0]
    assert abs(out[0, 0] - (70 + nose_idx / 100 * 260)) < 1e-3
    assert abs(out[0, 1] - (55 + 0.5 * 390)) < 1e-3
    left_hip = MP_TO_COCO[11]  # MediaPipe 23 → COCO left_hip
    assert left_hip == 23 and abs(out[11, 0] - (70 + 0.23 * 260)) < 1e-3
    assert (refiner.refined, refiner.fallback) == (1, 0)


def test_heavy_refiner_falls_back_to_yolo_when_no_person():
    import numpy as np

    refiner = _fake_refiner([])
    yolo = np.full((17, 2), 7.0, np.float32)
    out = refiner.refine(np.zeros((100, 100, 3), np.uint8), np.array([10, 10, 50, 90]), yolo)
    assert out is yolo
    assert (refiner.refined, refiner.fallback) == (0, 1)


def test_action_set_mismatch_warns(test_settings):
    from dataclasses import replace

    from services.model_loader import load_bundle

    bundle = load_bundle(replace(test_settings, action_set="team"))
    assert any("ACTION_SET='team'" in w for w in bundle.warnings)
    assert load_bundle(test_settings).warnings == []


def test_model_in_subfolder_reads_its_own_metadata(test_settings, tmp_path):
    """팀 모델처럼 정보 파일 이름이 같아도 모델 파일 옆의 정보를 읽는다."""
    import json
    import shutil
    from dataclasses import replace

    from services.model_loader import load_bundle

    models = tmp_path / "models"
    shutil.copytree(test_settings.model_dir, models)
    (models / "team").mkdir()
    for name in ("rf_action_baseline.joblib", "rf_feature_columns.json", "rf_training_info.json"):
        shutil.copy(models / name, models / "team" / name.replace("baseline", "team"))
    info = json.loads((models / "rf_training_info.json").read_text())
    (models / "rf_training_info.json").write_text(json.dumps({**info, "classes": ["x"]}))  # 바깥 파일은 일부러 틀리게
    bundle = load_bundle(replace(test_settings, model_dir=models, model_file="team/rf_action_team.joblib"))
    assert bundle.feature_count == 1140 and len(bundle.classes) == 6
