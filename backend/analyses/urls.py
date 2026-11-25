from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()
router.register("analyses", views.AnalysisJobViewSet, basename="analysis")

# 같은 API를 두 주소로 연다.
#   /api/analyses/     과제 문서
#   /api/v1/analyses/  9~10일차 단계별 가이드
urlpatterns = [
    path("api/cameras/", views.camera_list, name="camera-list"),
    path("api/v1/cameras/", views.camera_list),
    path("api/", include(router.urls)),
    path("api/v1/", include(router.urls)),
]
