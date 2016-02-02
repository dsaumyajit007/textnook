from django.conf.urls import url

from . import views

app_name = 'learn'

urlpatterns = [
	url(r'^',views.upload_document,name='index'),
	url(r'^upload/$',views.upload_document,name='upload_document'),
	
 ]