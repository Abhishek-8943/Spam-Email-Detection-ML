import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the model and vectorizer
model = pickle.load(open(spam_model.pkl, rb))
vectorizer = pickle.load(open(vectorizer.pkl, rb))

# Text cleaning function
def clean_text(text)
    text = text.lower()
    text = re.sub(r'd+', '', text)
    text = re.sub(r'[^ws]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Streamlit app UI
st.title(📧 Spam Email Detector)
email_input = st.text_area(Enter your emailmessage text)

if st.button(Predict)
    cleaned = clean_text(email_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    st.write(### Result)
    if prediction == 1
        st.error(🚫 This is Spam!)
    else
        st.success(✅ This is Not Spam!)
