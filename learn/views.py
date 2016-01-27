from django.http import HttpResponse
from django.shortcuts import get_object_or_404,render,redirect
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.contrib.auth.models import User
from learn.models import Document,Book
import json,csv
import pandas
import time
import pandas
import re
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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


def train_dataset(request):
	if not request.user.is_authenticated() or not request.user.is_active:
		return redirect("/authenticate/")
	snowball_stemmer = LancasterStemmer()
	#snowball_stemmer = SnowballStemmer('english')
	data_set = pandas.read_csv('/home/saumyajit/Desktop/code/textnook/media/documents/Books.csv', encoding =  'utf-8')
	data_set = data_set[data_set['Category ID'] != 'Others']
	data_set['Title'] = data_set['Title'].str.lower()

	remove_list = ['edition','ed','edn', 'vol' , 'vol.' , '-' ,'i']

	for index, row in data_set.iterrows():
		row['Title']=' '.join([i for i in row['Title'].split() if i not in remove_list])

	data_set['Title'] = data_set['Title'].apply(lambda x :re.sub(r'\w*\d\w*', '', x).strip())
	data_set['Title'] = data_set['Title'].apply(lambda x :re.sub(r'\([^)]*\)', ' ', x))
	data_set['Title'] = data_set['Title'].apply(lambda x :re.sub('[^A-Za-z0-9]+', ' ', x))

#data_set['Category ID'] = data_set['Category ID']+"|"+data_set['Subcategory ID']


	for index, row in data_set.iterrows():
		row['Title']=" ".join([snowball_stemmer.stem(i) for i in  row['Title'].split()])


	target = data_set['Category ID'].unique()
	X_train, X_test, y_train, y_test = train_test_split(
		data_set['Title'], data_set['Category ID'], test_size=0.33, random_state=90)

	

	text_clf = Pipeline([('vect', CountVectorizer()),
	                     ('tfidf', TfidfTransformer()),
	                     ('clf', MultinomialNB()),
	])
	text_clf = text_clf.fit(X_train, y_train)
	
	predicted = text_clf.predict(X_test)
	print np.mean(predicted == y_test)



	return HttpResponse()

