# -*- coding: utf-8 -*-
# Generated by Django 1.9.1 on 2016-01-30 13:38
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0003_document'),
    ]

    operations = [
        migrations.AddField(
            model_name='prediction',
            name='is_added',
            field=models.CharField(blank=True, max_length=10),
        ),
    ]
