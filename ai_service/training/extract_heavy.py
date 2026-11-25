"""팀 RGB 영상 → MediaPipe heavy(VIDEO 모드) → COCO17 관절 CSV (팀 keypoints CSV와 같은 형식).

사용: python extract_heavy.py <model.task> <out_dir> <video1> [video2 ...]
- 30fps 영상을 2프레임마다 1번 처리(=15fps, 서비스 TARGET_PROCESS_FPS와 동일)
- frame_id는 원본 프레임 번호, time_sec = frame_id / fps
"""
import sys, time
from pathlib import Path
import cv2, pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
MAX_SIDE = 960


def make_landmarker(model_path):
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
    return vision.PoseLandmarker.create_from_options(opts)


def extract(model_path, video, out_dir):
    video = Path(video); out = Path(out_dir) / f"{video.stem}.csv"
    if out.exists():
        print("skip", out.name); return
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / 15))
    lm = make_landmarker(model_path)  # 영상마다 새로 만들어 timestamp가 0부터 시작
    rows, fid, t0 = [], -1, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid % step:
            continue
        h, w = frame.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(fid * 1000 / fps))
        row = {"clip_id": video.stem, "frame_id": fid, "time_sec": fid / fps, "pose_detected": 0}
        if res.pose_landmarks:
            row["pose_detected"] = 1
            pts = res.pose_landmarks[0]
            for name, i in zip(NAMES, MP_TO_COCO):
                p = pts[i]
                row[f"{name}_x_px"] = p.x * w; row[f"{name}_y_px"] = p.y * h
                row[f"{name}_x_norm"] = p.x; row[f"{name}_y_norm"] = p.y
                row[f"{name}_conf"] = float(p.visibility if p.visibility is not None else 1.0)
        rows.append(row)
    lm.close(); cap.release()
    cols = ["clip_id", "frame_id", "time_sec", "pose_detected"] + [
        f"{n}_{k}" for n in NAMES for k in ("x_px", "y_px", "x_norm", "y_norm", "conf")]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(out, index=False)
    det = sum(r["pose_detected"] for r in rows)
    print(f"{video.stem}: {len(rows)}프레임, 검출 {det}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    model, out_dir, *videos = sys.argv[1:]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for v in videos:
        try:
            extract(model, v, out_dir)
        except Exception as e:  # 한 영상이 깨져도 나머지는 계속
            print("FAIL", v, e, flush=True)
