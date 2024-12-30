from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from .models import UploadedImage 

classes = ["benign", "malignant", "normal"]

# Function to preprocess the uploaded image
def preprocess_uploaded_image(uploaded_image):
    resized_image = cv2.resize(uploaded_image, (64, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to classify the image
def classify_image(uploaded_image):
    preprocessed_input = preprocess_uploaded_image(uploaded_image)
    input_feature = extract_hog_features([preprocessed_input])
    predicted_class_idx = svm_model.predict(input_feature)[0]
    if predicted_class_idx < len(classes):
        predicted_class = classes[predicted_class_idx]
        return predicted_class
    else:
        return "Not related to breast cancer"

# Load the SVM model
svm_model_path = os.path.join(settings.BASE_DIR, 'svm_model.pkl')
svm_model = joblib.load(svm_model_path)

# Feature Extraction (HOG)
def extract_hog_features(images):
    hog_features = []
    hog = cv2.HOGDescriptor(
        (64, 128),  # Adjust the window size to match your image size
        (16, 16),   # Adjust the block size as needed
        (8, 8),     # Adjust the block stride as needed
        (8, 8),     # Adjust the cell size as needed
        9           # Number of bins for the histograms
    )
    for image in images:
        hog_feature = hog.compute(image)
        hog_features.append(hog_feature)
    return np.array(hog_features)

# Your existing views
def home(request):
    return render(request, 'classifier/home.html')

# from django.contrib.staticfiles.templatetags.staticfiles import static

from django.conf.urls.static import static
def classify(request):
    if request.method == 'POST' and request.FILES['uploaded_image']:
        uploaded_image = request.FILES['uploaded_image'].read()
        nparr = np.frombuffer(uploaded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the uploaded image to the static folder
        static_image_path = os.path.join(settings.STATICFILES_DIRS[0], 'classifier', 'uploaded_image.png')
        cv2.imwrite(static_image_path, img)

        result = classify_image(img)
        return render(request, 'classifier/classify.html', {'result': result, 'static_image_path': '../static/classifier/uploaded_image.png'})
    else:
        return HttpResponse("Invalid Request")

