import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import string
import spacy
from nltk.corpus import stopwords

# Load the model and vectorizer
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Download NLTK stopwords
nltk.download("stopwords")
stopwords_list = stopwords.words("english")

# Define utility functions
def clean_text(text):
    return text.strip().lower()

def remove_punctuation(text):
    return "".join([char for char in text if char not in string.punctuation])

def tokenization(text):
    return re.split(r"\s+", text)

def remove_stopwords(tokens):
    return " ".join(token for token in tokens if token not in stopwords_list)

def lemmatizer(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if token.text not in set(stopwords_list))

# Streamlit app interface
st.set_page_config(page_title="Instagram Threads Sentiment Analysis", page_icon="📊", layout="wide")
st.title("📊 Instagram Threads Sentiment Analysis")

st.markdown(
    """
    Analyze the sentiment of your text input (positive, negative, or neutral) with this intuitive and interactive app.
    
    **Instructions:**
    1. Enter your text in the input box below.
    2. Click the **Predict Sentiment** button to get the sentiment analysis result.
    """
)

# Sidebar for additional features
with st.sidebar:
    st.header("About this App")
    st.write(
        "This app uses a pre-trained Support Vector Machine (SVM) model and TF-IDF vectorization to analyze the sentiment of text data."
    )
    st.write(
        """
        **Key Features:**
        - Text preprocessing with lemmatization and stopword removal.
        - Sentiment classification into Positive, Neutral, and Negative categories.
        """
    )

# User input
st.subheader("📝 Enter your text for analysis:")
user_input = st.text_area("Type your message here...", "", height=150)

# Process input
if user_input:
    with st.spinner("Processing your text..."):
        processed_input = clean_text(user_input)
        processed_input = remove_punctuation(processed_input)
        processed_input = tokenization(processed_input)
        processed_input = remove_stopwords(processed_input)
        processed_input = lemmatizer(processed_input)
else:
    processed_input = None

# Predict button
if st.button("🔍 Predict Sentiment"):
    if processed_input:
        text_vectorized = vectorizer.transform([processed_input])
        prediction = model.predict(text_vectorized)[0]

        st.header("Prediction Result:")
        if prediction == -1:
            st.error("### 😠 Negative Sentiment")
            st.write("The sentiment of the text is **negative**.")
        elif prediction == 0:
            st.warning("### 😐 Neutral Sentiment")
            st.write("The sentiment of the text is **neutral**.")
        elif prediction == 1:
            st.success("### 😊 Positive Sentiment")
            st.write("The sentiment of the text is **positive**.")
    else:
        st.error("⚠️ Please enter valid text for analysis.")
