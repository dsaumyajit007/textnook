from django.db import models

# Create your models here.

class Category(models.Model):
    """
    cate
    Description: Model Description
    """
    id = models.AutoField(primary_key=True)
    category_name=models.CharField(max_length=10)

    def __str__(self):
		return self.category_name

# class subcategory(models.Model):
#     """
#     Description: Model Description
#     """
#     id=models.AutoField(primary_key=True)

    
class Book(models.Model):
    """
    book model including an id, book title, category, and subcategory
    Description: Model Description
    """
    id=models.AutoField(primary_key=True)
    book_name=models.CharField(max_length=512)
    category=models.OneToOneField(Category)

    def __str__(self):
    	return self.book_name + " : "+self.category.category_name
    