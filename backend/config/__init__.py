# Django가 시작될 때 Celery 앱도 로드해 @shared_task 가 이 앱에 연결되도록 한다.
from .celery import app as celery_app

__all__ = ("celery_app",)
