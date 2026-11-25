from django.urls import include, path

urlpatterns = [
    path("", include("analyses.urls")),  # /api/analyses/, /api/cameras/ (+ /api/v1/ 별칭)
]
