import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean email text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'[^\w\s]', '', text)      # remove punctuation
    text = re.sub(r'\s+', ' ', text)         # remove extra spaces
    return text.strip()

# Streamlit UI
st.title("📧 Spam Email Detector")

email_input = st.text_area("Enter your email message text")

if st.button("Predict"):
    cleaned = clean_text(email_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    st.write("### Result")
    if prediction[0] == 1:
        st.error("🚫 This is Spam!")
    else:
        st.success("✅ This is Not Spam!")
