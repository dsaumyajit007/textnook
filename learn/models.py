from django.db import models
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
# Create your models here.


def train_dataset_from_admin_upload():
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
    


def add_book_from_document(file):
    document = pandas.read_csv(file)
    

    if 'Title' not in document or 'Category ID' not in document or 'Subcategory ID' not in document:
        raise ValueError('The file does not contain either of "Title" , "Category ID" or "Subcategory ID"')
    book_names = document['Title']
    category_names = document['Category ID']
    subcategory_names = document['Subcategory ID']
    
    books=Book.objects.all()
    for book_name,category_name,subcategory_name in zip(book_names,category_names,subcategory_names):
        
        book = Book(book_name = book_name,category_name = category_name, subcategory_name = subcategory_name)
        book.save()

    

class Document(models.Model):
    document_help_text='<strong>THE FILE UPLOADED MUST CONTAIN THESE THREE COLUMNS:<br/>1. Title<br/>2. Category ID<br/>3. Subcategory ID<br/>Plese note that the column names are case and space sensitive</strong>'
    document = models.FileField(upload_to='documents',help_text=document_help_text)
    def save(self, *args, **kwargs):
        super(Document,self).save(*args, **kwargs)
        add_book_from_document(self.document.path)
        train_dataset_from_admin_upload()

    def __str__(self):
        return self.document.name




class Book(models.Model):
    """
    book model including an id, book title, category, and subcategory
    Description: Model Description
    """
    id=models.AutoField(primary_key=True)
    book_name=models.CharField(max_length=512)
    category_name=models.CharField(max_length=128)
    subcategory_name=models.CharField(max_length=128)
    def __str__(self):
        return self.book_name + " : "+self.category_name+" : "+self.subcategory_name 