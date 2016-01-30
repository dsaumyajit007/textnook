from __future__ import unicode_literals

from django.db import models
import pandas
import re
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib




import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Create your models here.
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

class Prediction(models.Model):
    """
    Description: storing all predictions
    """
    id=models.AutoField(primary_key=True)
    book_name=models.CharField(max_length=512)
    category_name=models.CharField(max_length=128,blank=True)
    subcategory_name=models.CharField(max_length=128,blank=True)
    	
    def save(self, *args, **kwargs):
   		print "was called"
   		[predicted_category,predicted_subcategory]=predict_category_subcategory(self.book_name)
		print predicted_category
		print predicted_subcategory
		self.category_name = predicted_category
		self.subcategory_name = predicted_subcategory
		super(Prediction,self).save(*args, **kwargs)
        

    def __str__(self):
        return self.book_name + " : "+self.category_name+" : "+self.subcategory_name 
    

