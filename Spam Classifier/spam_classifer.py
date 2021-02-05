import numpy as np
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
import re
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
model=load_model('Spam_classifier.h5',compile=False)

st.title('SPAM MAIL CLASSIFIER')
st.image('download.jpg',width=500)
voc_size=10000
sent_length = 20

st.header('Give the Input as a Mail to check whether it is a spam or not')
input=st.text_input('Write the text here')
input_data=list(input)

def input_preprocessing(x):
    corpus = []

    review = re.sub('[^a-zA-z]', ' ', str(x))
    review = review.lower()
    review = review.split()
    review = [wnl.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

    onehot_repr = [one_hot(words, voc_size) for words in corpus]


    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    X=np.array(embedded_docs)
    return X


x=input_preprocessing(input_data)
prediction = model.predict_proba(x)
if st.button('predict'):
    st.write('The Percentage of having this mail is spam is',prediction)
    #if prediction==1:
        #st.write('This mail is Spam')

            #st.write('Probability of having Cancer to the patient is '+str(output))
            #st.write('Please Consult to the doctor as early as possible')






