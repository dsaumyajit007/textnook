from django.http import HttpResponse
from django.shortcuts import get_object_or_404,render,redirect
from django.contrib.auth import authenticate,login
from django.contrib.auth import logout
from django.contrib.auth.models import User
import json
# Create your views here.

def index(request):

	if request.user.is_authenticated() and request.user.is_active == True:
		return redirect('/learn')
	else:
		return redirect('/auth/signin')



def signout(request):
	#print "logging out" this also clears all session data
	logout(request) 
	return redirect('/auth/')


def signin(request):
	if request.user.is_authenticated() == True and request.user.is_active == True:
		return redirect('/learn')
	if request.method == 'POST':
		print "post"
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(username=username,password=password)
		if user is None:
			return render(request,'auth/signin.html',{'error':'Invalid username or password'})
		
		login(request,user)
		return redirect('/learn/') #render the learn page
	else:
		print "no post"
		return render(request,'auth/signin.html',None)

def signup(request):
	if request.user.is_authenticated() == True and request.user.is_active == True:
		return redirect('/learn')
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		password2 = request.POST['password2']
		firstName = request.POST['firstName']
		lastName = request.POST['lastName']
		if password2 != password:
			return render(request,'auth/signup.html',{'error':'Password doesn\'t match'}) 

		emailAddr = request.POST['email']

		try:
			u = User._default_manager.get(username__iexact=username) #fix for case sensitive username
			return render(request,'auth/signup.html',{'error':'username already already exists!'})
		except User.DoesNotExist:
			user = User.objects.create_user(username=username,email=emailAddr,first_name=firstName,last_name=lastName)
			user.set_password(password)
			user.is_active = False #user will be active only when he completes registration
			user.save()
			user.backend = 'django.contrib.auth.backends.ModelBackend' #user backend error fix
			authenticate(username=username,password=password)
			login(request,user)
			return redirect('/auth')
	else:
		return render(request,'auth/signup.html',None)

