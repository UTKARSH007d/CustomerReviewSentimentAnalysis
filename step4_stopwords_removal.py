import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("data/IMDB Dataset.csv")


def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    words = text.split() #tokenization

    words = [word for word in words
             if word not in stop_words]

    return " ".join(words)


review = df['review'][0]

cleaned = clean_text(review)

without_stopwords = remove_stopwords(cleaned)

print("Original Review:\n")
print(review)

print("\n--------------------\n")

print("After Cleaning:\n")
print(cleaned)

print("\n--------------------\n")

print("After Stopword Removal:\n")
print(without_stopwords)