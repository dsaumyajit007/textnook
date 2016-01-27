from django.db import models
import pandas
# Create your models here.




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
    