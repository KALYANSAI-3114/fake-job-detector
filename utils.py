# utils.py
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(df):
    df = df[['title', 'company_profile', 'description', 'requirements', 'fraudulent']].copy()
    df.fillna('', inplace=True)
    df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements']
    df['text'] = df['text'].apply(clean_text)
    return df[['text', 'fraudulent']]
