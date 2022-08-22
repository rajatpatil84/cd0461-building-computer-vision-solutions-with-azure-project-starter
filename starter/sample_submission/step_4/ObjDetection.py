#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# In[2]:


import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# In[3]:


def show_image_in_cell(face_url):
    response = requests.get(face_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(20,10))
    plt.imshow(img)
    plt.show()


# In[4]:


from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid


# In[5]:


TRAINING_ENDPOINT = "https://udacitycustomvision.cognitiveservices.azure.com/"
training_key = "00503dc21b6b417e9bdaaccd87eae1d0"
training_resource_id = '/subscriptions/aacec976-caf3-499d-81b3-829197f07e1a/resourceGroups/aind-204379/providers/Microsoft.CognitiveServices/accounts/udacitycustomvision'


# In[6]:


PREDICTION_ENDPOINT = 'https://udacitycustomvision-prediction.cognitiveservices.azure.com/'
prediction_key = "ef56a21c4f8a47b08a8b566153b3f53e"
prediction_resource_id = "/subscriptions/aacec976-caf3-499d-81b3-829197f07e1a/resourceGroups/aind-204379/providers/Microsoft.CognitiveServices/accounts/udacitycustomvision"


# In[19]:


training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)


# In[7]:


prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)


# In[8]:


predictor.api_version


# In[9]:


get_ipython().system('pwd')


# In[10]:


local_image_path = '/home/workspace'
get_ipython().system('ls $local_image_path')


# In[14]:


project_id = '99c32f88-edc1-4328-ad13-f5f9957a4a45'
publish_iteration_name = "Iteration1"


# In[15]:


file_names = ['lighter_test_set_2of5.jpg', 'lighter_test_set_3of5.jpg', 'lighter_test_set_4of5.jpg', 'lighter_test_set_5of5.jpg']


# In[16]:


for file_name in file_names:
    with open(os.path.join (local_image_path, file_name), "rb") as image_contents:
        results = predictor.detect_image(project_id, publish_iteration_name, image_contents.read())

        img_view_ready = Image.open(image_contents)
        plt.figure()
        plt.imshow(img_view_ready)
        img_view_ready.close()
        # Display the results.
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))
    print(f"*************{file_name}********************")


# In[ ]:




