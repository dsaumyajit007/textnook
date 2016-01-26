from django.db import models

# Create your models here.




class Document(models.Model):
    document = models.FileField(upload_to='documents')
    






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
    