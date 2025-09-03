# myapp/urls.py
from django.urls import path
from . import views

from .views import (result_detail_hausa,result_detail_arabic,result_detail_french,result_detail_igbo,result_detail_swahili,result_detail_yoruba,fact1,index,hausa,igbo, yoruba,french,habout,hausa_result,yabout,
                    iabout,fabout,ifactcheck,sabout,sfactcheck,swahili,fact,
                    swahili_result,yfactcheck,ffactcheck,about, result,hausa_factcheck,
                    hausa_result,preview,arabic,arabic_result,arabout,arafactcheck,
                    reliable_sources,yoruba_result,igbo_result,french_result,internet_safety,
                    # FactcheckAPI,FactCheckWithTavilyAPIView,
                    # FactCheckAPIView,factcheck_webpage,combined_view,
                    combine
                    )
from rest_framework import routers

router = routers.DefaultRouter()

from django.urls import include


urlpatterns = [
    
    path('', index, name='index'),
    path('router', include(router.urls)),
    path('about/', about, name='about'),
    # path('report/', report, name='report'),
    path('fact/',fact, name="fact"),
    path('fact1/',fact1, name="fact1"),
 
    path('preview/', preview, name='preview'),
    path('result/', result, name='result'),
    # path('process_input/', views.process_user_input, name='process_input'),
    # path('factcheck-webpage/', factcheck_webpage, name='factcheck_webpage'),
    # path('factcheck-assistant/', combined_view, name='factcheck_assistant'),
    path('combine/', combine, name='combine'),
    path('all-factchecks/', views.all_factchecks, name='all_factchecks'),
    path('export-factchecks/', views.export_factchecks_csv, name='export_factchecks_csv'),


   
    path('reliable_sources/<str:user_query>/', reliable_sources, name='reliable_sources'),

    path('hausa', hausa, name='hausa'),
    path('hausa/factcheck',hausa_factcheck, name="hausa_factcheck"),
    path('hausa/about', habout, name='habout'),
    path('hausa/result/', hausa_result, name='hausa_result'),
    path('hausa/result/<slug:slug>/', result_detail_hausa, name='result_detail_hausa'),
   
    path('yoruba', yoruba, name='yoruba'),
    path('yoruba/factcheck', yfactcheck, name='yfactcheck'),
    path('yoruba/about', yabout, name='yabout'),
    path('yoruba/result', yoruba_result, name='yoruba_result'),
    path('yoruba/result/<slug:slug>/', result_detail_yoruba, name='result_detail_yoruba'),

    path('igbo', igbo, name='igbo'),
    path('igbo/factcheck', ifactcheck, name='ifactcheck'),
    path('igbo/about', iabout, name='iabout'),
    path('igbo/result', igbo_result, name='igbo_result'),
    path('igbo/result/<slug:slug>/', result_detail_igbo, name='result_detail_igbo'),


    path('swahili', swahili, name='swahili'),
    path('swahili/factcheck', sfactcheck, name='sfactcheck'),
    path('swahili/about', sabout, name='sabout'),
    path('swahili/result', swahili_result, name='swahili_result'),
    path('swahili/result/<slug:slug>/', result_detail_swahili, name='result_detail_swahili'),

    path('french', french, name='french'),
    path('french/factcheck', ffactcheck, name='ffactcheck'),
    path('french/about', fabout, name='fabout'),
    path('french/result', french_result, name='french_result'),
    path('french/result/<slug:slug>/', result_detail_french, name='result_detail_french'),

    path('arabic', arabic, name='arabic'),
    path('arabic/factcheck', arafactcheck, name='arabicfactcheck'),
    path('arabic/about', arabout, name='arabicabout'),
    path('arabic/result', arabic_result, name='arabic_result'),
    path('arabic/result/<slug:slug>/', result_detail_arabic, name='result_detail_arabic'),
    
    path('safety/', internet_safety, name='internet_safety'),    # path('api/factcheck/', FactcheckAPI.as_view(), name='factcheck_api'),
    # path('api/factchecker/', FactCheckAPIView.as_view(), name='factchecker_api'),
    # path('factcheckAPI/', FactCheckWithTavilyAPIView.as_view(), name='fact-check'),
]

urlpatterns += router.urls