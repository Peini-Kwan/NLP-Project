#!/usr/bin/env python
# coding: utf-8

# ## Streamlit for NLP

# - To perform basic and useful NLP task with Streamlit and Gensim

# In[2]:


import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

sns.set_style('whitegrid')


# In[3]:


import re
import unicodedata

import contractions
import gensim
import gensim.downloader as api
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvi
import base64


# In[4]:


from cleantext import clean
from gensim import corpora
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.keyedvectors import KeyedVectors
# from gensim.summarization import keywords
from gensim.test.utils import common_texts
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from textblob import TextBlob
from wordcloud import WordCloud


# In[5]:


import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[6]:


from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


# In[7]:


def convert_to_df(sentiment):
    sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
    return sentiment_df


# In[8]:


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result 


# In[10]:


def main():
    st.title("Amazon The North Face Sentiment Analysis & Topic Modelling NLP App")
    st.subheader("Streamlit Projects")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1,col2 = st.columns(2)
        
        
#         if submit_button:

#              with col1:
#                 st.info("Results")
#                 sentiment = TextBlob(raw_text).sentiment
#                 st.write(sentiment)

#                 # Emoji
#                 if sentiment.polarity > 0:
#                     st.markdown("Sentiment:: Positive :smiley: ")
#                 elif sentiment.polarity < 0:
#                     st.markdown("Sentiment:: Negative :angry: ")
#                 else:
#                     st.markdown("Sentiment:: Neutral 😐 ")

#                 # Dataframe
#                 result_df = convert_to_df(sentiment)
#                 st.dataframe(result_df)

#                 # Visualization
#                 c = alt.Chart(result_df).mark_bar().encode(
#                     x='metric',
#                     y='value',
#                     color='metric')
#                 st.altair_chart(c,use_container_width=True)

        if submit_button:
        
            with col1:
                st.info("Results")
                sentiment = TextBlob(raw_text).sentiment
                st.write(sentiment)

                # Emoji
                if sentiment.polarity > 0:
                    st.markdown("Sentiment: Positive 😃")
                elif sentiment.polarity < 0:
                    st.markdown("Sentiment: Negative 😠")
                else:
                    st.markdown("Sentiment: Neutral 😐")

                # Dataframe
                result_df = convert_to_df(sentiment)
                st.dataframe(result_df)

                # Visualization
                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric',
                    y='value',
                    color='metric')
                st.altair_chart(c, use_container_width=True)



            with col2:
                st.info("Token Sentiment")

                token_sentiments = analyze_token_sentiment(raw_text)
                st.write(token_sentiments)

    else:
        st.subheader("About")


if __name__ == '__main__':
    main()

