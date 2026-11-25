"""Django 설정. 비밀값·주소는 모두 환경변수(.env)로 받는다."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = env_bool("DJANGO_DEBUG", False)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")

# 로그인 기능 없음: admin·sessions·messages·accounts 제외.
# auth·contenttypes는 DRF가 내부에서 import 하므로 남긴다 (테이블만 생기고 사용하지 않음).
INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",  # DRF Browsable API 화면의 CSS
    "rest_framework",
    "analyses",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # gunicorn 환경에서 static 제공
    "django.middleware.common.CommonMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,  # DRF Browsable API 템플릿
        "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --------------------------------------------------------------------------- DB
# 기본: PostgreSQL (Compose의 postgres 서비스). 테스트에서는 DB_ENGINE=sqlite 로 전환 가능.
if os.getenv("DB_ENGINE", "postgres") == "sqlite":
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "vision_action"),
            "USER": os.getenv("POSTGRES_USER", "vision"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "postgres"),  # Compose 서비스 이름
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": 60,
            # 연결을 재사용하기 전에 살아 있는지 확인한다.
            # 없으면 postgres 컨테이너 재시작 직후 첫 요청이 'SSL connection has been closed' 로 500이 된다 (T10에서 발견).
            "CONN_HEALTH_CHECKS": True,
        }
    }

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

# --------------------------------------------------------------------------- Static / Media
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"
        if not DEBUG
        else "django.contrib.staticfiles.storage.StaticFilesStorage"
    },
}
# 업로드·카메라 수집 영상. Compose에서 django와 celery가 같은 볼륨을 공유한다.
MEDIA_URL = "media/"
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "media"))
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # 이보다 큰 업로드는 임시파일로 받음

# --------------------------------------------------------------------------- DRF
REST_FRAMEWORK = {
    # 로그인 없음: 누구나 요청·조회 가능. 인증이 없으므로 CSRF 검사도 하지 않는다.
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "UNAUTHENTICATED_USER": None,
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
}

# --------------------------------------------------------------------------- Celery / Redis
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ACKS_LATE = True  # Worker가 처리 중 죽으면 작업을 다시 큐로
CELERY_WORKER_PREFETCH_MULTIPLIER = 1  # 긴 영상 작업을 한 Worker가 몰아 받지 않도록
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "600"))
CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "540"))
CELERY_TIMEZONE = TIME_ZONE
# STARTED 상태로 이 시간(초)을 넘기면 멈춘 작업으로 보고 실패 처리한다 (analyses/job_state.py)
JOB_STALE_SECONDS = int(os.getenv("JOB_STALE_SECONDS", str(CELERY_TASK_TIME_LIMIT + 120)))

# --------------------------------------------------------------------------- AI 서비스
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://fastapi:8000")  # localhost 아님: 서비스 이름
AI_CONNECT_TIMEOUT = float(os.getenv("AI_CONNECT_TIMEOUT", "5"))
AI_READ_TIMEOUT = float(os.getenv("AI_READ_TIMEOUT", "300"))

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# --------------------------------------------------------------------------- IP Webcam
# 사용자가 URL을 직접 입력하지 않는다. 허용된 카메라 목록에서 key 만 고른다 (SSRF 방지).
# 형식: key|표시이름|URL , 여러 개는 쉼표로 구분
#   IP_WEBCAMS=phone1|교실 휴대폰 1|http://192.168.0.XX:8080/video
def _parse_cameras(raw: str) -> dict[str, dict[str, str]]:
    cameras: dict[str, dict[str, str]] = {}
    for item in raw.split(","):
        parts = [p.strip() for p in item.split("|")]
        if len(parts) == 3 and all(parts):
            key, label, url = parts
            cameras[key] = {"label": label, "url": url}
    return cameras


IP_WEBCAMS = _parse_cameras(os.getenv("IP_WEBCAMS", ""))
CAPTURE_SECONDS = float(os.getenv("CAPTURE_SECONDS", "8"))
CAPTURE_OPEN_TIMEOUT_SEC = float(os.getenv("CAPTURE_OPEN_TIMEOUT_SEC", "10"))
CAPTURE_MAX_WIDTH = int(os.getenv("CAPTURE_MAX_WIDTH", "960"))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
}
