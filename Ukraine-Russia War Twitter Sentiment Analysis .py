#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

data=pd.read_csv(r"C:\Users\Muskan Khan\Downloads\Ukrain vs. Russia.csv")
print(data.head())


# In[2]:


data1=data.reset_index()
print(data1.head())


# In[3]:


print(data.columns)


# In[4]:


data=data[["username","tweet","language"]]


# In[5]:


data.isnull().sum()


# In[6]:


data["language"].value_counts()


# In[7]:


nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
stopword=set(stopwords.words('english'))

def clean(text):
    text =str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','', text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text=[stemmer.stem(word)for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"]=data["tweet"].apply(clean)


# In[8]:




text=" ".join(i for i in data.tweet)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color="white").generate(text)


# In[13]:


nltk.download('vader_lexicon')
sentiments=SentimentIntensityAnalyzer()
data["Positive"]=[sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"]=[sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"]=[sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data=data[["tweet","Positive","Negative","Neutral"]]
print(data.head())


# In[23]:


positive=' '.join([i for i in data['tweet'][data['Positive']>data["Negative"]]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color="white", colormap='Greys').generate(positive)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


negative=' '.join([i for i in data['tweet'][data['Negative']>data["Positive"]]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color="white", colormap='Blues').generate(positive)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




