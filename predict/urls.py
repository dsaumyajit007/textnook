from django.conf.urls import url

from . import views

app_name = 'predict'

urlpatterns = [
	#url(r'^',views.view_predictions,name='index'),
	url(r'^upload/$',views.upload_document,name='upload_document'),
	url(r'^add_predictions/$',views.add_predictions,name='add_predictions'),


	
 ]