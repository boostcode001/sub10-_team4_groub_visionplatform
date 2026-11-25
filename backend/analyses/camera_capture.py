"""허용된 IP Webcam에서 정해진 시간(기본 8초)만 Frame을 읽어 MP4 한 개로 저장하고 연결을 끊는다.

- 누가 읽나: Celery Worker (Django 웹 요청은 기다리지 않음)
- 언제 끝나나: capture_seconds 가 지나면 종료 (연결 시간은 포함하지 않음)
- 저장 FPS: 스트림이 알려 주는 FPS(MJPEG는 0인 경우가 많음)를 믿지 않고, 실제로 받은 Frame 수 ÷ 수집 시간으로 계산한다.
  FastAPI가 frame_idx / fps 로 시간을 계산하므로 FPS가 틀리면 Sliding Window 길이가 틀어진다.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 30


class CaptureError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _resize(frame, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def capture_clip(
    url: str,
    seconds: float,
    output_path: Path,
    open_timeout_sec: float = 10.0,
    max_width: int = 960,
    min_seconds: float = 2.0,
) -> dict:
    """반환: {"frames": n, "fps": 실제 FPS, "duration_sec": 실제 수집 시간}"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.mp4")

    capture = cv2.VideoCapture()
    capture.setExceptionMode(False)
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_sec * 1000),
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
    opened = capture.open(url, cv2.CAP_FFMPEG, params) or capture.open(url)
    if not opened:
        raise CaptureError("CAMERA_UNAVAILABLE", "카메라에 연결할 수 없습니다. 휴대폰 IP Webcam 앱과 같은 네트워크인지 확인하세요.")

    writer = None
    frames = 0
    failures = 0
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CaptureError("CAMERA_NO_FRAME", "카메라에 연결됐지만 영상을 받지 못했습니다.")

        frame = _resize(frame, max_width)
        height, width = frame.shape[:2]
        provisional_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 15.0
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), provisional_fps, (width, height))

        started = time.monotonic()  # 첫 Frame 이후부터 수집 시간을 잰다 (연결 시간 제외)
        while True:
            if frame is not None:
                writer.write(_resize(frame, max_width))
                frames += 1
            if time.monotonic() - started >= seconds:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                frame = None
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    break  # 스트림 끊김
                time.sleep(0.05)
            else:
                failures = 0
        duration = time.monotonic() - started
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if duration < min_seconds or frames < 10:
        tmp_path.unlink(missing_ok=True)
        raise CaptureError(
            "CAMERA_STREAM_INTERRUPTED",
            f"카메라 연결이 끊겨 {duration:.1f}초({frames} Frame)만 수집했습니다.",
        )

    actual_fps = frames / duration
    if abs(actual_fps - provisional_fps) / actual_fps > 0.05:
        _rewrite_with_fps(tmp_path, output_path, actual_fps)
        tmp_path.unlink(missing_ok=True)
    else:
        os.replace(tmp_path, output_path)

    logger.info("captured %s frames in %.2fs (%.2f fps) -> %s", frames, duration, actual_fps, output_path)
    return {"frames": frames, "fps": round(actual_fps, 3), "duration_sec": round(duration, 3)}


def _rewrite_with_fps(src: Path, dst: Path, fps: float) -> None:
    """Frame 내용은 그대로 두고 FPS 메타데이터만 실제 값으로 다시 기록."""
    reader = cv2.VideoCapture(str(src))
    ok, frame = reader.read()
    if not ok:
        reader.release()
        raise CaptureError("CAPTURE_WRITE_FAILED", "수집한 영상을 다시 읽지 못했습니다.")
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    while ok:
        writer.write(frame)
        ok, frame = reader.read()
    reader.release()
    writer.release()
