# FastAPI AI 추론 서버 이미지
# 학습 때 쓴 Python·scikit-learn·ultralytics 버전과 맞춰야 joblib 모델이 같은 결과를 낸다.
# 버전은 .env 의 PYTHON_VERSION / SKLEARN_VERSION / ULTRALYTICS_VERSION 으로 지정한다.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG SKLEARN_VERSION=1.9.1
ARG ULTRALYTICS_VERSION=8.4.211
# 팀 모델(MediaPipe heavy 관절)용. 6동작 모델만 쓸 때는 0으로 빌드해도 됨
ARG INSTALL_MEDIAPIPE=1
ARG MEDIAPIPE_VERSION=1.2.3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# OpenCV 실행용(libgl1·libglib2.0-0), MediaPipe 네이티브 라이브러리용(libegl1·libgles2·libportaudio2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libportaudio2 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU 전용 torch: CUDA 빌드보다 이미지가 수 GB 작다
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt \
    "scikit-learn==${SKLEARN_VERSION}" \
    "ultralytics==${ULTRALYTICS_VERSION}" \
    && if [ "$INSTALL_MEDIAPIPE" = "1" ]; then pip install "mediapipe==${MEDIAPIPE_VERSION}"; fi \
    && python -c "import cv2, sklearn, ultralytics; print('cv2', cv2.__version__, 'sklearn', sklearn.__version__)" \
    && if [ "$INSTALL_MEDIAPIPE" = "1" ]; then python -c "from mediapipe.tasks.python.core import mediapipe_c_bindings as b; b.load_raw_library(); print('mediapipe native OK')"; fi

# 코드 (src/ 는 기존 subject10_pose_action/src 복사본). 모델·설정은 compose에서 볼륨으로 마운트
COPY main.py ./
COPY routers ./routers
COPY services ./services
COPY src ./src
COPY scripts ./scripts

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
