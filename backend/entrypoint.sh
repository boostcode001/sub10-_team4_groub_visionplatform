#!/bin/sh
# django 컨테이너 시작 순서: DB 마이그레이션 → static 수집 → gunicorn
set -e
python manage.py migrate --noinput
python manage.py collectstatic --noinput
exec gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120
