# -*- coding: utf-8 -*-
# Generated by Django 1.9.1 on 2016-01-27 17:41
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='prediction',
            name='category_name',
            field=models.CharField(blank=True, max_length=128),
        ),
        migrations.AlterField(
            model_name='prediction',
            name='subcategory_name',
            field=models.CharField(blank=True, max_length=128),
        ),
    ]