import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/IMDB Dataset.csv")

df = df.head(5000)


def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    stop_words = set(stopwords.words('english'))

    words = text.split()

    words = [word for word in words
             if word not in stop_words]

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word)
             for word in words]

    return " ".join(words)


df['review'] = df['review'].apply(preprocess)

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['review'])

print("TF-IDF Matrix Shape:")
print(X.shape)