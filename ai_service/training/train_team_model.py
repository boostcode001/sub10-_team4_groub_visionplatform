"""팀 재활 6동작 모델 학습 스크립트 (팀이 직접 촬영한 RGB 영상 → MediaPipe heavy 관절 CSV).

| 동작 | 영문 Label | 데이터 |
|---|---|---|
| A001 앉아서 다리 앞으로 뻗기 | seated_knee_extension | 카카오 videos(S01~S03) + 11/17 추가 촬영(정면) |
| A002 팔 위아래로 뻗기 | shoulder_reach_up_down | 카카오 videos(S01~S03) + 11/17 추가 촬영(정면) |
| A003 폼롤러 힙힌지 | foam_roller_hip_hinge | 11/17 추가 촬영(정면) |
| A006 옆으로 누워 다리 들기 | side_lying_leg_raise | 카카오 videos(S01~S03) |
| A009 무릎 모으며 엉덩이 들기 | adductor_bridge | 11/17 추가 촬영(정면) |
| A010 버드독 | bird_dog | 11/17 추가 촬영 |

관절 CSV 파일 이름 규칙 (둘 다 지원):
- S01_A006_T03.csv   → 사람 S01, 동작 A006 (팀 keypoints, time_sec 없음 → frame_id / 30)
- T009_kss_2.csv     → 사람 kss, 동작 A009 (11/17 추가 촬영, extract_heavy.py 출력, time_sec 있음)

Feature 규격은 6일차와 같다 (30 Step × 38 = 1,140). 학습 샘플은 v2와 같이 Window 증강으로 만든다.

실행:
    python training/train_team_model.py --project ~/ai_vision/subject10_pose_action \
        --keypoints <팀 keypoints 폴더> <추가 촬영 관절 CSV 폴더> --out training/model_team6
출력: rf_action_team.joblib, rf_feature_columns.json, rf_training_info.json, cv_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LABELS = {
    "A001": "seated_knee_extension",
    "A002": "shoulder_reach_up_down",
    "A003": "foam_roller_hip_hinge",
    "A006": "side_lying_leg_raise",
    "A009": "adductor_bridge",
    "A010": "bird_dog",
}
TRAIN_WINDOWS = (2.0, 3.0, 4.0, 5.0)
TRAIN_STRIDE = 0.25
EVAL_WINDOW = 3.0          # 팀 모드 서비스 WINDOW_SECONDS
SEQUENCE_STEPS = 30
MIN_FRAMES_PER_WINDOW = 10
TRIM_SECONDS = 1.0         # 원본 영상 앞뒤의 T자세·녹화 버튼 구간을 조금 잘라 Label 잡음을 줄임
DEFAULT_FPS = 30.0
RF_PARAMS = dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
                 class_weight="balanced")  # 동작마다 영상 수가 달라서 균형 가중치 사용

TEAM_NAME = re.compile(r"^(S\d+)_(A\d{3})_T\d+$")      # S01_A006_T03
EXTRA_NAME = re.compile(r"^T(\d{3})_([a-z]+\d*)_\d+$")  # T009_kss_2 → A009, kss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="subject10_pose_action 경로 (src/ 사용)")
    p.add_argument("--keypoints", nargs="+", required=True, help="관절 CSV 폴더들 (하위 폴더 포함)")
    p.add_argument("--out", default="training/model_team6")
    p.add_argument("--skip-cv", action="store_true")
    return p.parse_args()


def parse_name(stem: str):
    m = TEAM_NAME.match(stem)
    if m:
        return m.group(1), m.group(2)
    m = EXTRA_NAME.match(stem)
    if m:
        return m.group(2), f"A{m.group(1)}"
    return None


def load_clips(folders: list[Path]) -> dict:
    from src.sequence_utils import build_frame_feature_matrix

    clips = {}
    for folder in folders:
        for path in sorted(folder.rglob("*.csv")):
            parsed = parse_name(path.stem)
            if not parsed or parsed[1] not in LABELS:
                print(f"[SKIP] 이름 규칙 밖: {path.name}")
                continue
            subject, action_id = parsed
            df = pd.read_csv(path)
            if "time_sec" not in df.columns:
                df["time_sec"] = df["frame_id"] / DEFAULT_FPS
            # 앞뒤 TRIM_SECONDS 제거 (짧은 클립은 자르지 않음)
            t0, t1 = df["time_sec"].min(), df["time_sec"].max()
            if t1 - t0 > 4 * TRIM_SECONDS + EVAL_WINDOW:
                df = df[(df["time_sec"] >= t0 + TRIM_SECONDS) & (df["time_sec"] <= t1 - TRIM_SECONDS)]
            try:
                matrix, frame_ids = build_frame_feature_matrix(df)
            except Exception as exc:  # 유효 프레임이 너무 적은 클립은 건너뜀
                print(f"[SKIP] {path.stem}: {exc}")
                continue
            if len(frame_ids) < MIN_FRAMES_PER_WINDOW:
                print(f"[SKIP] {path.stem}: 유효 프레임 {len(frame_ids)}")
                continue
            times = df.set_index("frame_id").loc[frame_ids, "time_sec"].to_numpy(dtype=float)
            clips[path.stem] = {"subject": subject, "label": LABELS[action_id], "matrix": np.asarray(matrix),
                                "frame_ids": np.asarray(frame_ids), "times": times}
    return clips


def window_samples(clip: dict, window: float, stride: float) -> list[np.ndarray]:
    from src.sequence_utils import flatten_sequence, resample_sequence

    times = clip["times"]
    if times[-1] - times[0] <= window:
        return [flatten_sequence(resample_sequence(clip["matrix"], clip["frame_ids"], target_steps=SEQUENCE_STEPS))]
    samples, start = [], times[0]
    while start + window <= times[-1] + 1e-6:
        mask = (times >= start) & (times <= start + window)
        if mask.sum() >= MIN_FRAMES_PER_WINDOW:
            seq = resample_sequence(clip["matrix"][mask], clip["frame_ids"][mask], target_steps=SEQUENCE_STEPS)
            samples.append(flatten_sequence(seq))
        start += stride
    return samples


def build_xy(clips: dict, subjects: set[str]):
    X, y = [], []
    for clip in clips.values():
        if clip["subject"] in subjects:
            for window in TRAIN_WINDOWS:
                s = window_samples(clip, window, TRAIN_STRIDE)
                X += s
                y += [clip["label"]] * len(s)
    return np.asarray(X, dtype=np.float32), y


def leave_one_subject_out(clips: dict) -> dict:
    """한 명을 빼고 학습 → 그 사람 영상으로 시험 (처음 보는 사람에게 통하는지)."""
    subjects = sorted({c["subject"] for c in clips.values()})
    classes = sorted(LABELS.values())
    report = {"window_seconds": EVAL_WINDOW, "folds": [], "misclassified": []}
    confusion = {t: Counter() for t in classes}
    clip_hits, win_hits, confs = [], [], []
    for hold in subjects:
        X, y = build_xy(clips, set(subjects) - {hold})
        model = RandomForestClassifier(**RF_PARAMS).fit(X, y)
        fold = []
        for cid, clip in clips.items():
            if clip["subject"] != hold:
                continue
            proba = model.predict_proba(np.asarray(window_samples(clip, EVAL_WINDOW, 0.25)))
            pred = model.classes_[proba.argmax(1)]
            win_hits += list(pred == clip["label"])
            confs += list(proba.max(1))
            labels, counts = np.unique(pred, return_counts=True)
            top = str(labels[counts.argmax()])
            confusion[clip["label"]][top] += 1
            fold.append(top == clip["label"])
            if top != clip["label"]:
                report["misclassified"].append({"clip_id": cid, "true": clip["label"], "pred": top,
                                                "vote_ratio": round(float(counts.max() / counts.sum()), 3)})
        clip_hits += fold
        report["folds"].append({"test_subject": hold, "clips": len(fold), "clip_accuracy": round(float(np.mean(fold)), 4)})
        print(f"[CV] test={hold:<5} clips={len(fold):2d} clip_acc={np.mean(fold):.3f}", flush=True)
    conf = np.asarray(confs)
    report.update(clip_accuracy=round(float(np.mean(clip_hits)), 4), clips=len(clip_hits),
                  window_accuracy=round(float(np.mean(win_hits)), 4),
                  mean_confidence=round(float(conf.mean()), 4),
                  ratio_confidence_ge_055=round(float((conf >= 0.55).mean()), 4),
                  confusion={t: dict(c) for t, c in confusion.items()})
    print(f"[CV] clip={report['clip_accuracy']:.3f} ({sum(clip_hits)}/{len(clip_hits)}) "
          f"window={report['window_accuracy']:.3f} conf={report['mean_confidence']:.2f}")
    return report


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    sys.path.insert(0, str(project))
    from src.sequence_utils import sequence_feature_names

    clips = load_clips([Path(k).expanduser().resolve() for k in args.keypoints])
    per_class = Counter(c["label"] for c in clips.values())
    subjects = sorted({c["subject"] for c in clips.values()})
    print(f"[DATA] 클립 {len(clips)}개, 사람 {subjects}")
    for label in sorted(LABELS.values()):
        print(f"       {label:<24} {per_class[label]}개")
    missing = set(LABELS.values()) - set(per_class)
    if missing:
        print(f"[ERROR] 데이터가 없는 동작: {sorted(missing)}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = None if args.skip_cv else leave_one_subject_out(clips)

    X, y = build_xy(clips, set(subjects))
    model = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    columns = sequence_feature_names(SEQUENCE_STEPS)
    assert model.n_features_in_ == len(columns) == SEQUENCE_STEPS * 38, "Feature 규격 불일치"

    joblib.dump(model, out / "rf_action_team.joblib")
    (out / "rf_feature_columns.json").write_text(json.dumps(columns, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "model_version": "team6_v1_window_augmented",
        "pose_model": "mediapipe pose_landmarker_heavy (33 → COCO17 mapped)",
        "classes": [str(c) for c in model.classes_],
        "action_ids": {v: k for k, v in LABELS.items()},
        "feature_count": len(columns),
        "sequence_steps": SEQUENCE_STEPS,
        "min_valid_frames": MIN_FRAMES_PER_WINDOW,
        "train_windows_seconds": list(TRAIN_WINDOWS),
        "train_stride_seconds": TRAIN_STRIDE,
        "trim_seconds": TRIM_SECONDS,
        "train_subjects": subjects,
        "train_clips": len(clips),
        "clips_per_class": dict(per_class),
        "train_samples": len(y),
        "samples_per_class": dict(Counter(y)),
        "n_estimators": RF_PARAMS["n_estimators"],
        "class_weight": RF_PARAMS["class_weight"],
        "random_state": RF_PARAMS["random_state"],
        "cv_clip_accuracy": report["clip_accuracy"] if report else None,
    }
    (out / "rf_training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if report:
        (out / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out} (샘플 {len(y)}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
