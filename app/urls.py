from django.urls import path, include
from rest_framework.routers import DefaultRouter

# Create router and register viewsets
router = DefaultRouter()
router.register(r'factchecks', FactcheckViewSet)
router.register(r'user-reports', UserReportViewSet)
router.register(r'active-users', ActiveUserViewSet)

api_urlpatterns = [
    # Router URLs
    path('api/v1/', include(router.urls)),
    
    # Custom API endpoints
    path('api/v1/fact-check/', FactCheckAPIView.as_view(), name='fact-check'),
    path('api/v1/multi-language-fact-check/', MultiLanguageFactCheckAPIView.as_view(), name='multi-language-fact-check'),
    
    # Authentication (if using DRF auth)
    path('api-auth/', include('rest_framework.urls')),
]

