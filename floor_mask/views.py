from django.shortcuts import render
from .models import image_mask
from django.http import HttpResponseRedirect
from django.conf import settings
from .yolact_cpu.eval import img_out
from PIL import Image
from io import BytesIO
import base64


# Create your views here.

def home(request):
	
	images=image_mask.objects.all()
	url=images[len(images)-1].pic.url
	path = "/home/sky/Desktop/yolact/django_yolact"+url
	out = img_out(path)
	img = Image.fromarray(out, 'RGB')
	
	data = BytesIO()
	img.save(data, "JPEG") # pick your format
	data64 = base64.b64encode(data.getvalue())
	
	return render(request,'home.html',{'upload_img':url, "mask_img":'data:img/jpeg;base64,'+data64.decode('utf-8') })



def uploadImage(request):
	print('image handling')
	img = request.FILES['image']
	image = image_mask(pic=img)
	image.save()
	return HttpResponseRedirect('/')