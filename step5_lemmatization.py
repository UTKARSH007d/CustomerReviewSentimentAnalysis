import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("data/IMDB Dataset.csv")


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = text.split()

    words = [word for word in words
             if word not in stop_words]

    return " ".join(words)


def lemmatize_text(text):

    lemmatizer = WordNetLemmatizer()

    words = text.split()

    words = [lemmatizer.lemmatize(word)
             for word in words]

    return " ".join(words)


review = df['review'][0]

cleaned = clean_text(review)

without_stopwords = remove_stopwords(cleaned)

lemmatized = lemmatize_text(without_stopwords)

print("After Lemmatization:\n")

print(lemmatized)
