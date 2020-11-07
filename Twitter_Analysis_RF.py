#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/Twitter_Analysis/train.csv")
test = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/Twitter_Analysis/test.csv")


# In[3]:


train


# In[4]:


test


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


#Downloading commonly used data sets
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# In[8]:


# APPEND TEST DATA WITH TRAIN DATA

merged = train.append(test)
merged


# ## PREPROCESSING

# In[9]:


# SEPERATING TWEETS FROM THE APPENDED DATAFRAME (MERGED)
# 'tweets' CONTAIN BOTH TRAIN AND TEST TWEETS

tweets = merged.tweet
tweets


# ## TOKENIZATION

# In[10]:


# TOKENIZING 'tweets' DATA AND THEN JOINING THEM TOGETHER

from nltk import TweetTokenizer
from nltk.tokenize import word_tokenize
tk = TweetTokenizer()
tweets = tweets.apply(lambda x: tk.tokenize(x)).apply(lambda x: ' '.join(x))


# In[11]:


tweets


# ## REMOVING PUNCTUATIONS

# In[12]:


tweets = tweets.str.replace('[^a-zA-Z]+', ' ')
tweets


# In[13]:


from nltk.tokenize import word_tokenize


# In[14]:


tweets


# ## REMOVING SHORT WORDS

# In[15]:


# REMOVING SMALL WORD FROM TWEETS (LENGTH <= 3)

tweets = tweets.apply(lambda x: ' '.join([w for w in word_tokenize(x) if len(w) > 3]))
tweets


# ## STEMMING

# In[16]:


from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
tweets = tweets.apply(lambda x: [stemmer.stem(i.lower()) for i in tk.tokenize(x)]).apply(lambda x: ' '.join(x))


# In[17]:


tweets


# ## REMOVING STOP WORDS

# In[18]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
tweets = tweets.apply(lambda x: [i for i in word_tokenize(x) if i not in stop]).apply(lambda x: ' '.join(x))


# In[19]:


tweets


# In[20]:


# REPLACING EXISTING TWEET COLUMN IN MERGED DATA WITH THE PROCESSED 'tweets'

merged.tweet = tweets
merged


# In[21]:


#Seperating train data and test data from merged data using the .iloc with row limit as the shape of the train data (which we loaded in the beginning)
train_data = merged.iloc[:train.shape[0]]


# In[22]:


test_data = merged.iloc[train.shape[0]:]


# In[23]:


train_data


# In[24]:


test_data


# In[25]:


train_data


# In[26]:


# SEPERATING TWEETS FROM BOTH train_data and test_data

train_tweet = train_data.tweet
test_tweet = test_data.tweet


# In[27]:


train_tweet


# In[28]:


test_tweet


# ## TFIDF VECTORIZATION

# In[29]:


# VECTORIZING THE TWEETS USING TFIDF VECTORIZER

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words = stop)


# In[30]:


train_data_vec = vec.fit_transform(train_tweet)


# In[31]:


test_data_vec = vec.transform(test_tweet)


# In[32]:


train_data_vec


# In[33]:


test_data_vec


# In[34]:


# TAKING LABEL FROM train_data

y = train_data.label
y = y.values
y


# In[35]:


# COUNTING LABEL VALUES 0 AND 1 TO CHECK THE IMBALANCE OF DATA

pd.Series(y).value_counts()


# ## SPLITTING TRAIN AND TEST DATA

# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data_vec, y, test_size = 0.2, stratify = y, random_state = 42)


# In[37]:


# TO OVERCOME THE IMBALANCED DATA

from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_res, y_res = smote.fit_resample(x_train, y_train)


# ## RANDOM FOREST CLASSIFICATION

# In[38]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(x_res, y_res)
y_pred2 = model2.predict(x_test)


# In[39]:


y_pred2


# In[40]:


y_test


# In[41]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred2)


# ## RF (SUBMISSION)

# In[42]:


test_data_vec


# In[43]:


y_sub2 = model2.predict(test_data_vec)


# In[44]:


print(y_sub2)


# In[45]:


my_submission = pd.DataFrame({'id': test.id, 'label': y_sub2})


# In[46]:


my_submission.to_csv('submission_RF3.csv', index = False)

