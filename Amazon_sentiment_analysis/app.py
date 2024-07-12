import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as ps

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score



cv = joblib.load('countVectorizer.pkl')
best_random_forest_model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Sentiment Analysis")

input_text = st.text_area("Enter the text")

if st.button('Predict'):

    # Initialize PorterStemmer
    ps_stemmer = ps()

    # Preprocess function
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []  # removing special characters
        for i in text:
            if i.isalnum():
                y.append(i)

        # Remove stopwords and punctuation, and perform stemming
        y = [ps_stemmer.stem(word) for word in y if word not in stopwords.words('english') and word not in string.punctuation]

        return " ".join(y)

    transformed_text = transform_text(input_text)

    # Vectorize
    vector_input = cv.transform([transformed_text]).toarray()

    # Scale input
    vector_input_scaled = scaler.transform(vector_input)

    # Predict
    prediction = best_random_forest_model.predict(vector_input_scaled)[0]

    # Display result
    if prediction == 1:
        st.header("Positive Sentiment")
    else:
        st.header("Negative Sentiment")
