import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("Customer Review Sentiment Analysis")

# Text input
review = st.text_area("Enter a Review")

# Predict button
if st.button("Predict Sentiment"):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    probability = model.predict_proba(review_vector)

    confidence = max(probability[0]) * 100

    st.write("Sentiment:", prediction[0])

    st.write("Confidence:", round(confidence, 2), "%")