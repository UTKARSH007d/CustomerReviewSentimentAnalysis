import pandas as pd
import re

df = pd.read_csv("data/IMDB Dataset.csv")

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text

print("Original Review:")
print(df['review'][0])

print("\n")

print("Cleaned Review:")
print(clean_text(df['review'][0]))