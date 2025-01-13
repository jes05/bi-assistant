import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.DataFrame(data)

# Features and labels
X = df["query"]
y = df["column"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the pipeline
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'column_classifier.pkl')

print("Model saved as column_classifier.pkl")
