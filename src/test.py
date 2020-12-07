#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from datetime import datetime
from dateutil.tz import tzutc
from ast import literal_eval

import re
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import joblib
import pickle
from sklearn import preprocessing
from scipy import spatial
from tensorflow import keras
from sklearn.decomposition import PCA
import sys
import warnings
warnings.filterwarnings("ignore")

# In[2]:


# add test cases here and in videos/test csv
patient = 26207
patient_info = pd.read_csv('/data/patients_test.csv')
videos = pd.read_csv('/data/videos_test.csv').set_index('Unnamed: 0')
videos.head()


# In[3]:


patient_info = patient_info[['patient_id', 'age', 'sex', 'has_bh_specialist', 'total_activities_done', 'unique_days_with_activity']]
patient_info = pd.get_dummies(patient_info, columns = ['sex', 'has_bh_specialist'])

big = patient_info.merge(videos, on = "patient_id")

video_stats = big.groupby(['video_id']).mean()
video_features = videos.groupby('video_id').mean()
video_features['avg_age'] = video_stats['age']
video_features['gender'] = video_stats['sex_Male']


# In[4]:


# Normalize, PCA
cols = list(video_features.columns)
x = video_features.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(x)
x_scaled = min_max_scaler.transform(x)
video_features = pd.DataFrame(x_scaled)
dims = len(video_features.columns)

pca = joblib.load('/models/video_pca.pkl')
reduced_movie_features = pca.transform(video_features)
reduced_movie_features = pd.DataFrame(reduced_movie_features)
reduced_movie_features = reduced_movie_features.set_index(video_stats.index.values)


# In[5]:


patient_features = big.groupby(['patient_id']).mean()
patient_features = patient_features [['age', 'sex_Female', 'sex_Male', 'has_bh_specialist_False', 'has_bh_specialist_True',
                                      'length', 'video_created_time', 'video_views', 'primary_category_ADHD',
                                      'primary_category_Anxiety', 'primary_category_Cognitive Behavioral Therapy',
                                      'primary_category_Depression', 'primary_category_Managing Pain',
                                      'primary_category_Mindfulness', 'primary_category_New & Expecting Mothers',
                                      'primary_category_PTSD', 'primary_category_Sleep', 'primary_category_Stress',
                                      'primary_category_Substance Use', 'primary_category_Yoga']]
patient_features = patient_features.dropna()
patient_index = patient_features.index.values
patient_features_unscaled = patient_features.copy()
cols = list(patient_features.columns)
x = patient_features.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
patient_features = pd.DataFrame(x_scaled)

user_pca = joblib.load('/models/user_pca.pkl')
reduced_patient_features = user_pca.transform(patient_features)
reduced_patient_features = pd.DataFrame(reduced_patient_features)
reduced_patient_features = reduced_patient_features.set_index(patient_index)
patient_features = patient_features.set_index(patient_index)


# In[8]:


from scipy import spatial
vids = video_stats.index.values
model = keras.models.load_model('/models/model.h5')
def get_closest_user(user, k, pca):
  """For a given user, returns the k nearest neighbors in the new PCA feature space.
  params:
  user - id of the user in question (int)
  k - number of nearest neighbors
  pca - PCA object for transform."""
  patient_pca = pca.transform(patient_features)
  patients = patient_features.index.values
  patient_pca = pd.DataFrame(patient_pca)
  patient_pca = patient_pca.set_index(patients)

  patient_index = patient_pca[patient_pca.index.values == user]

  patient_similarity = [spatial.distance.cosine(list(patient_index), list(x)[1:]) for x in patient_pca.itertuples()]

  closest_indices = np.argpartition(patient_similarity, k+1).tolist()[1:k+1]
  return patients[closest_indices]

def get_closest_movie(movie, user, k, pca):
  """For a given movie, return the k nearest movies in the new PCA feature space.
  This movie cannot be seen before by the user. (Business logic)
  params:
  movie = vector of average movie
  user = user id 
  k = number of nearest neighbors
  pca = pca object"""

  
  video_pca = pca.transform(video_features)
  patients = video_features.index.values
  video_pca = pd.DataFrame(video_pca)
  video_pca = video_pca.set_index(vids)

  
  transformed_movie = pca.transform(movie.reshape(-1, 1))[0]

  video_similarity = [spatial.distance.cosine(transformed_movie, list(x)[1:]) for x in video_pca.itertuples()]

  closest_indices = np.argpartition(video_similarity, k+1).tolist()[1:k+1]

  video_similarity = np.array(video_similarity)
  return vids[closest_indices], video_similarity[closest_indices]
  
def nn_predict(user):
  """Predicts next movie based on user ID."""
  ## First take a look at the user's features.
  patient_info[patient_info['patient_id'] == user]
  ## We wish to transform these features using our PCA reduction
  reduced_patient_features = user_pca.transform(patient_features)
  reduced_patient_features = pd.DataFrame(reduced_patient_features)
  reduced_patient_features = reduced_patient_features.set_index(patient_index)
  user_features = reduced_patient_features[reduced_patient_features.index.values == 26207]

  ## This reduced feature space goes into our neural network
  predictions = model.predict(user_features)[0]
  # finding the predicted movie(s)
  top_movies = predictions.argsort()[-10:][::-1]
  ## Convert index back to movie
  return top_movies


# In[9]:


recommendations = nn_predict(patient)
vids_orig = pd.read_csv('/data/video_watched_events_CONFIDENTIAL.csv')
print('Based on your previous watch history, we recommend:')
print()
for rec in recommendations:
  print(vids_orig.loc[rec, :].notes + ': ' + vids_orig.loc[rec, :].url)

