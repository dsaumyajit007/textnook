from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404,render,redirect
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.contrib.auth.models import User
from learn.models import Book
from predict.models import Document,Prediction
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
# Create your views here.

def train_dataset_from_predictions():
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
    print "dumped"




def predict_category_subcategory(book_name):
	data_set1 = pandas.Series(book_name.encode('ascii'))

    #Data Preprocessing
	data_set1 = data_set1.dropna(axis=0,how='any')
	data_set1 = data_set1.str.lower()

    #Manual removal List
	remove_list = ['edition','ed','edn', 'vol' , 'vol.' , '-' ,'i']


	data_set1[0] =' '.join([i for i in data_set1[0].split() if i not in remove_list])

	data_set1 = data_set1.apply(lambda x :re.sub(r'\w*\d\w*', '', x).strip())
	data_set1 = data_set1.apply(lambda x :re.sub(r'\([^)]*\)', ' ', x))
	data_set1 = data_set1.apply(lambda x :re.sub('[^A-Za-z0-9]+', ' ', x))
    #data_set['Category ID'] = data_set['Category ID']+"|"+data_set['Subcategory ID']


    #Stemming the book titles
	stemmer = LancasterStemmer()
	data_set1[0]=" ".join([stemmer.stem(i) for i in  data_set1[0].split()])

	clf = joblib.load(os.path.join(BASE_DIR+"/learners/",'category_predict.pkl'))
	ans = clf.predict(data_set1)
	sub_clf = joblib.load(os.path.join(BASE_DIR+"/learners/",'subcategory_predict.pkl'))
	sub_ans = sub_clf.predict(data_set1)
	return [ans[0],sub_ans[0]]

def handle_uploaded_file(request,file):
	if not request.user.is_authenticated() or not request.user.is_active : 
		return redirect('/authentication/')

	document = pandas.read_csv(file)



	if 'Title' not in document :
		return 'error'
	book_names = document['Title']
	c = 0
	document['Category'] = document['Title']

	document['Subcategory'] = document['Title']

	for book_name in book_names : 
		print book_name
		[predicted_category,predicted_subcategory]=predict_category_subcategory(book_name)
		document['Category'][c] = predicted_category
		document['Subcategory'][c] = predicted_subcategory
		c = c+1;
		print "doing"

		prediction = Prediction(book_name = book_name,category_name = predicted_category,subcategory_name = predicted_subcategory)
		prediction.save()
		document.to_csv('out.csv',sep=',')

 



def upload_document(request):
	if not ( request.user.is_authenticated() or request.user.is_active == True):
		return redirect("/authentication/")
	
	if request.method == 'POST':
		uploaded_documents=request.FILES.getlist('uploaded_document')
		print uploaded_documents
		for uploaded_document in uploaded_documents :
			new_document=Document(document=uploaded_document)
			new_document.save();
			print "Saved"
			if handle_uploaded_file(request,new_document.document.path) == 'error' :
				return render(request,'predict/uploaddocument.html',{'error':'Upload was unsuccessful. Please try again'})
			else :
				return render(request,'predict/uploaddocument.html',{'success':'File successfully saved'})
		
	else:
		return render(request,'predict/uploaddocument.html',None)

def view_predictions(request) :
	if not ( request.user.is_authenticated() or request.user.is_active == True):
		return redirect("/authentication/")
	predictions = Prediction.objects.all();
	#print predictions
	return render(request,'predict/predictions.html',{'predictions':predictions})


def add_predictions(request) :
	print "i was here"
	if not ( request.user.is_authenticated() or request.user.is_active == True):
		return redirect("/auth/")

	if request.method == 'POST' :
		print "here"
		predictions_list = json.loads(request.POST.get('predictions'))
		print predictions_list
		for prediction in predictions_list :
			#print prediction
			p = Prediction.objects.get(id = prediction)
			#print p.category_name
			print p
			p.is_added = "1"
			p.save()
			book = Book(book_name = p.book_name , category_name = p.category_name, subcategory_name = p.subcategory_name)
			book.save()
			
		train_dataset_from_predictions()
		return HttpResponse("done")



