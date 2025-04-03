import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load saved model, vectorizer, and label encoder
@st.cache_resource
def load_model():
    with open("artifacts/model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("artifacts/vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("artifacts/label_encoder.pkl", "rb") as le_file:
        label_encoder = pickle.load(le_file)
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Streamlit App UI
st.title("📝 Sentiment Analysis NLP App")
st.markdown("### Enter a review below to predict sentiment.")

# User Input
user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess user input
        processed_text = preprocess_text(user_input)

        # Convert text to numerical format
        text_vectorized = vectorizer.transform([processed_text])

        # Predict sentiment
        prediction = model.predict(text_vectorized)

        # Decode label
        sentiment_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction Result:")
        if sentiment_label.lower() == "positive":
            st.success("😊 This review is **Positive**!")
        else:
            st.error("😠 This review is **Negative**.")
    else:
        st.warning("⚠️ Please enter a review before analyzing!")

# Footer
st.markdown("---")
st.markdown("🎯 NLP Sentiment Analysis using Streamlit")
