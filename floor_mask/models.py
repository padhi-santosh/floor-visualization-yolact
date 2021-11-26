from django.db import models

# Create your models here.
class image_mask(models.Model):
	pic = models.ImageField(upload_to='images')