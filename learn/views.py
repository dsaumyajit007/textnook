from django.http import HttpResponse
from django.shortcuts import get_object_or_404,render,redirect
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.contrib.auth.models import User
from learn.models import Document,Book
import json,csv
import pandas
import time

def handle_uploaded_file(request,file):
	if not request.user.is_authenticated or not request.user.is_active:
		return redirect('/auth/')
	
	document = pandas.read_csv(file)
	

	if 'Title' not in document or 'Category ID' not in document or 'Subcategory ID' not in document:
		return 'error'
	book_names = document['Title']
	category_names = document['Category ID']
	subcategory_names = document['Subcategory ID']
	

	for book_name,category_name,subcategory_name in zip(book_names,category_names,subcategory_names):
		book = Book(book_name = book_name,category_name = category_name, subcategory_name = subcategory_name)
		book.save()

	return "success"

def index(request):
	if not request.user.is_authenticated or not request.user.is_active:
		return redirect('/auth/')
	return render(request,'learn/index.html',None)


def upload_document(request):
	if not ( request.user.is_authenticated() or request.user.is_active == True):
		return redirect("/auth/")
	
	if request.method == 'POST':
		uploaded_documents=request.FILES.getlist('uploaded_document')
		for uploaded_document in uploaded_documents :
			new_document=Document(document=uploaded_document)
			new_document.save()
			if handle_uploaded_file(request,new_document.document.path) == 'error' :
				return render(request,'learn/uploaddocument.html',{'error':'Upload was unsuccessful. Please try again'})
			else :
				print "successful"
				return HttpResponse()
		
		
	else:
		return render(request,'learn/uploaddocument.html',None)
