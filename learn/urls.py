from django.conf.urls import url

from . import views

app_name = 'machine'

urlpatterns = [
	url(r'^$',views.index,name='index'),
	url(r'^upload/$',views.upload_document,name='upload_document'),
	
 ]