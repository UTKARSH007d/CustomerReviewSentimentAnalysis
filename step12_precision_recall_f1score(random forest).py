import pandas as pd
import re
import joblib


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Logistic Regression...")

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:")
print(accuracy)
print("Precision:")
print(precision_score(y_test, predictions, pos_label='positive'))

print("Recall:")
print(recall_score(y_test, predictions, pos_label='positive'))

print("F1 Score:")
print(f1_score(y_test, predictions, pos_label='positive'))
print("\nTraining Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:")
print(nb_accuracy)
print("Naive Bayes Precision:")
print(precision_score(y_test, nb_predictions, pos_label='positive'))

print("Naive Bayes Recall:")
print(recall_score(y_test, nb_predictions, pos_label='positive'))

print("Naive Bayes F1 Score:")
print(f1_score(y_test, nb_predictions, pos_label='positive'))

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Random Forest Accuracy:")
print(rf_accuracy)
print("Random Forest Precision:")
print(precision_score(y_test, rf_predictions, pos_label='positive'))

print("Random Forest Recall:")
print(recall_score(y_test, rf_predictions, pos_label='positive'))

print("Random Forest F1 Score:")
print(f1_score(y_test, rf_predictions, pos_label='positive'))
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model Saved Successfully")