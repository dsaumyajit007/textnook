from django.conf.urls import patterns, include, url
from django.conf import settings

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
   
    url(r'^admin/', include(admin.site.urls)),
    url(r'^auth/',include('auth.urls',namespace='auth')),
    
)