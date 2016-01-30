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
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.externals import joblib
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))



def train_dataset_from_user_upload():
    #Data input
    data_set = pandas.DataFrame(list(Book.objects.all().values('book_name', 'category_name', 'subcategory_name')))


    #Data Preprocessing
    data_set = data_set.dropna(axis=0,how='any')
    column=list(data_set)
    data_set = data_set[data_set['category_name'] != 'Others']
    data_set['book_name'] = data_set['book_name'].str.lower()

    #Manual removal List
    remove_list = ['edition','ed','edn', 'vol' , 'vol.' , '-' ,'i']

    for index, row in data_set.iterrows():
        row['book_name']=' '.join([i for i in row['book_name'].split() if i not in remove_list])

    data_set['book_name'] = data_set['book_name'].apply(lambda x :re.sub(r'\w*\d\w*', '', x).strip())
    data_set['book_name'] = data_set['book_name'].apply(lambda x :re.sub(r'\([^)]*\)', ' ', x))
    data_set['book_name'] = data_set['book_name'].apply(lambda x :re.sub('[^A-Za-z0-9]+', ' ', x))
    #data_set['category_name'] = data_set['category_name']+"|"+data_set['subcategory_name']


    #Stemming the book titles
    stemmer = LancasterStemmer()
    for index, row in data_set.iterrows():
        row['book_name']=" ".join([stemmer.stem(i) for i in  row['book_name'].split()])

    #Setting up target list
    target = data_set['category_name'].unique()
    sub_target = data_set['subcategory_name'].unique()


    #Splitting the data into train and test
    train_data,test_data=train_test_split(data_set,test_size=0.2)

    #Pipeline for training
    random_clf = Pipeline([('vect', CountVectorizer( ngram_range=(1, 4),analyzer="word")),
                         ('tfidf', TfidfTransformer()),
                         ('clf', OneVsRestClassifier(LinearSVC(random_state=142))),
    ])
    main_random_clf = random_clf.fit(train_data['book_name'], train_data['category_name'])


    #Predicting the category_name
    predicted = main_random_clf.predict(test_data['book_name'])
    print np.mean(predicted == test_data['category_name'])  
    print(metrics.classification_report(test_data['category_name'], predicted,target_names=target))
    metrics.confusion_matrix(test_data['category_name'], predicted)

    #saving model
    joblib.dump(main_random_clf, os.path.join(BASE_DIR+"/learners/",'category_predict.pkl')) 

    #Predicting the Subcategory_name
    sub_random_clf = random_clf.fit(train_data['book_name'], train_data[column[2]])

    predicted = sub_random_clf.predict(test_data['book_name'])
    print np.mean(predicted == test_data[column[2]])
    print(metrics.classification_report(test_data[column[2]], predicted,target_names=sub_target))
    metrics.confusion_matrix(test_data[column[2]], predicted)

    #saving model
    joblib.dump(sub_random_clf, os.path.join(BASE_DIR+"/learners/",'subcategory_predict.pkl'))
    




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
	train_dataset_from_user_upload()

def index(request):
	print "was here"
	if not request.user.is_authenticated or not request.user.is_active:
		return redirect('/authentication/')
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
				return render(request,'learn/uploaddocument.html',{'success' : "Data set was trained successfully"})
		
		
	else:
		return render(request,'learn/uploaddocument.html',None)




