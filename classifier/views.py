from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import cv2
import numpy as np
import joblib
from .models import UploadedImage
from PIL import Image
import time
import glob
import os
import os.path
import fnmatch
from pathlib import Path
import pandas as pd                                     # Data analysis and manipultion tool
import numpy as np                                      # Fundamental package for linear algebra and multidimensional arrays
import matplotlib.pyplot as plt
import cv2                                              # Library for image processing
import datetime as dt
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split    # For splitting the data into train and validation set
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils import shuffle
from IPython.core.display import display, HTML
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import PIL.ImageOps

# Heart disease prediction model and functions
from keras.models import load_model

# Load the model
cnn_heart_model = load_model("model_acc_95.h5")

classes_heart = ["NORMAL", "Myocardial Infarction Patient"]
image_size_heart = 100

def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j,i] = 255
        elif img.ndim == 3:
            img[j,i,0]= 255
            img[j,i,1]= 255
            img[j,i,2]= 255
        return img

def bg_remov(image):
    result = salt(image, 10)
    median = cv2.medianBlur(result,5)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    
    return blackAndWhiteImage

# Here is function wrap it up
def bg_remover(image, path = None):
  
  if(path != None):
    img = np.asarray(im_crop(path))

  # The Image will be of type PIL.Image.Image , so we will convert it to np.asarray:

  # convert to graky
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # threshold input image as mask
  mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

  # negate mask
  mask = 255 - mask

  # apply morphology to remove isolated extraneous noise
  # use borderconstant of black since foreground touches the edges
  kernel = np.ones((3,3), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # anti-alias the mask -- blur then stretch
  # blur alpha channel
  mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

  # linear stretch so that 127.5 goes to 0, but 255 stays 255
  mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

  # put mask into alpha channel
  result = img.copy()
  result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
  result[:, :, 3] = mask

  # save resulting masked image

  return result

#from PIL import Image
#import matplotlib.pyplot as plt

def im_crop(image,left=71.5, top= 287.5, right=2102, bottom= 1228):
  """ This function is used to crop the image and get just the ECG signals.
      input: 
        image : the image (jpg,png,...Etc)
        left: location in left of image
        top: location in top of image
        right: location in right of image
        bottom: location in bottom of image

        ######
        # choices from paper: left=71.5, top= 287.5, right=2102, bottom= 1228
        ######
      output: 
        img_out: the cropped ECG image.
  """
  img = Image.open(image) # for example : MI_df["File"][0]
  img_out = img.crop((left, top, right, bottom))
  
  


  return img_out


# Test 
# img_cc = im_crop(MI_df["File"][7])

#from PIL import Image
#import matplotlib.pyplot as plt

import PIL.ImageOps

def img_seg_12leads(image, width= 315, height= 315):
  """ This function is used to crop the image and get 12 leads of  the ECG signals.
      input: 
        image : the image cropped of ECG Signal
        width = 315
        height = 315

        ######
        # choices from paper: left=71.5, top= 287.5, right=2102, bottom= 1228
        ######
      output: 
        12 img_out: 12 leads ECG
  """

  # With ECG 12leads order 
  I_img   = image.crop((120.5, 0.5, width + 120.5 , 0.5 + height)).convert('L') # Converting Images to Grayscale 
  I_Neg   = PIL.ImageOps.invert(I_img)
  II_img  = image.crop((120.5, 315.5, width + 120.5 , 315.5+ height)).convert('L')
  II_Neg   = PIL.ImageOps.invert(II_img)
  III_img = image.crop((120.5, 630.5, width + 120.5 , 630.5+ height)).convert('L')
  III_Neg   = PIL.ImageOps.invert(III_img)
  aVL_img = image.crop((672.5, 315.5, width + 672.5 , 315.5+ height)).convert('L')
  aVL_Neg   = PIL.ImageOps.invert(aVL_img)
  aVR_img = image.crop((672.5, 0.5, width + 672.5 , 0.5 + height)).convert('L')
  aVR_Neg   = PIL.ImageOps.invert(aVR_img)
  aVF_img = image.crop((672.5, 630.5, width + 672.5 , 630.5+ height)).convert('L')
  aVF_Neg   = PIL.ImageOps.invert(aVF_img)
  V1_img  = image.crop((1133.5, 0.5, width + 1133.5 , 0.5+ height)).convert('L')
  V2_img  = image.crop((1133.5, 315.5, width + 1133.5 , 315.5+ height)).convert('L')
  V3_img  = image.crop((1133.5, 630.5, width + 1133.5 , 630.5+ height)).convert('L')
  V4_img  = image.crop((1639.5, 0.5, width + 1639.5 , 0.5 + height)).convert('L')
  V5_img  = image.crop((1639.5, 0.5, width + 1639.5 , 0.5+ height)).convert('L')
  V6_img  = image.crop((1639.5, 630.5, width + 1639.5 , 630.5+ height)).convert('L')

  return [I_img, I_Neg,II_img, II_Neg,III_img, III_Neg, aVR_img, aVR_Neg,aVL_img, aVL_Neg, aVF_img ,aVF_Neg,V1_img,V2_img,V3_img,V4_img,V5_img,V6_img,]

# Function to classify the image for heart disease prediction
def classify_image_heart(uploaded_image):
    picture = np.asarray(im_crop(uploaded_image))
    result_pic = bg_remover(picture, uploaded_image)
    pic_res = Image.fromarray(result_pic)
    all_formats = img_seg_12leads(pic_res)

    input_x = []
    for img_format in all_formats:
        input_img_array = np.array(img_format)
        # input_img_array = bg_remov(input_img_array)
        input_new_img_array = cv2.resize(input_img_array, (image_size_heart, image_size_heart))
        input_x.append([input_new_img_array])

    output_x = cnn_heart_model.predict(np.array(input_x).reshape(-1, image_size_heart, image_size_heart, 1))

    final_output = "Myocardial Infarction Patient" if np.count_nonzero(output_x == [0]) >= 9 else "NORMAL"
    
    return final_output


def home(request):
    return render(request, 'classifier/home.html')

def classify(request):
    if request.method == 'POST' and request.FILES['uploaded_image']:
        uploaded_image = request.FILES['uploaded_image'].read()
        nparr = np.frombuffer(uploaded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the uploaded image to the static folder
        static_image_path = os.path.join(settings.STATICFILES_DIRS[0], 'classifier', 'uploaded_image.png')
        cv2.imwrite(static_image_path, img)

        result = classify_image_heart(static_image_path)
        
        print(result)

        if(result == 'Myocardial Infarction Patient'):
          return render(request, 'classifier/classify1.html', {'result': result, 'static_image_path': '../static/classifier/uploaded_image.png'})
        else:
          return render(request, 'classifier/classify.html', {'result': result, 'static_image_path': '../static/classifier/uploaded_image.png'})

    else:
        return HttpResponse("Invalid Request")
