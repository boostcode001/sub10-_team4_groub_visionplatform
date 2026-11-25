# Django + Celery Worker 공용 이미지 (같은 코드, 실행 명령만 다름)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/media /app/staticfiles \
    && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["sh", "./entrypoint.sh"]
