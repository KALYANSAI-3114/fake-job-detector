# app.py
import streamlit as st
import pickle
from utils import clean_text

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("🧠 Fake Job Posting Detector")
st.write("Enter the job posting details below:")

title = st.text_input("Job Title")
company = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")

if st.button("Check If It's Fake"):
    input_text = title + " " + company + " " + description + " " + requirements
    cleaned = clean_text(input_text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0][prediction]

    if prediction == 1:
        st.error(f"🚨 This job post is **FAKE**! (Confidence: {prob*100:.2f}%)")
    else:
        st.success(f"✅ This job post seems **REAL**. (Confidence: {prob*100:.2f}%)")
