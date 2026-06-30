import pandas as pd
import numpy as np
import os
import nltk
import pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load datasets
print("Loading datasets...")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
true_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'news_dataset', 'True.csv'))
fake_df = pd.read_csv(os.path.join(BASE_DIR, 'dataset', 'news_dataset', 'Fake.csv'))

# Add labels
true_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake

# Combine
df = pd.concat([true_df, fake_df], ignore_index=True)

# Combine title + text
df['content'] = df['title'] + ' ' + df['text']

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("Cleaning text...")
df['content'] = df['content'].apply(clean_text)

# Split data
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
print("Vectorizing...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Done! Model saved as model.pkl and vectorizer.pkl")