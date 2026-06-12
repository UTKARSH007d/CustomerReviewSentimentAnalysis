# Customer Review Sentiment Analysis

## Project Overview

This project performs sentiment analysis on customer reviews using both traditional Machine Learning and Transformer-based Deep Learning models.

The system classifies reviews as:

- Positive
- Negative

A Streamlit dashboard is provided for real-time sentiment prediction.

---

## Dataset

Dataset Used: IMDb Movie Reviews Dataset

- Total Reviews Used: 5000
- Training Samples: 4000
- Testing Samples: 1000

---

## Data Preprocessing

The following preprocessing steps were applied:

1. Text Cleaning
2. Stopword Removal
3. Lemmatization
4. TF-IDF Vectorization

---

## Models Implemented

### Traditional Machine Learning

- Logistic Regression
- Naive Bayes
- Random Forest

### Transformer Models

- DistilBERT
- BERT

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 0.866 | 0.840 | 0.883 | 0.861 |
| DistilBERT | 0.845 | 0.827 | 0.847 | 0.837 |
| BERT | 0.839 | 0.824 | 0.836 | 0.830 |
| Naive Bayes | 0.831 | 0.828 | 0.809 | 0.818 |
| Random Forest | 0.828 | 0.824 | 0.806 | 0.815 |

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Transformers (Hugging Face)
- PyTorch

---

## Streamlit Dashboard

The application allows users to:

- Enter a review
- Predict sentiment
- View confidence score

---

## Author

Utkarsh Gupta

AI/ML Internship Project
