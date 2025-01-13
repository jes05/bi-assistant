import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pickle

# Path to the file mapping CSV
from get_common_variables import output_path
file_mapping_path = output_path  
file_mapping_df = pd.read_csv(file_mapping_path)

# Sample Training Data (You should expand this for better accuracy)
# Assuming you augment this dataset manually or via user interactions
training_data = [
    {"query": "Show me stock data", "filename": "all_stocks_5yr.csv"},
    {"query": "List all cards", "filename": "cards_data.csv"},
    {"query": "Get user details", "filename": "cve.csv"},
    {"query": "Fetch incident logs", "filename": "incident_event_log.csv"},
    {"query": "Show product vulnerabilities", "filename": "products.csv"}
]
training_df = pd.DataFrame(training_data)

# Train the Bag-of-Words Classifier
def train_classifier():
    X = training_df["query"]
    y = training_df["filename"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with TfidfVectorizer and Logistic Regression
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(X_train, y_train)

    # Save the model for future use
    with open("query_classifier.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Classifier trained successfully!")

# Load the trained classifier
def load_classifier():
    with open("query_classifier.pkl", "rb") as f:
        return pickle.load(f)

# Get the file path using the trained model
def get_filepath_from_model(user_query):
    classifier = load_classifier()
    predicted_filename = classifier.predict([user_query])[0]
    file_row = file_mapping_df[file_mapping_df["filename"] == predicted_filename]
    if not file_row.empty:
        return file_row["filepath"].values[0]
    return None

# Load CSV file into DataFrame
def load_csv_to_dataframe(filepath):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"No CSV file found at path: {filepath}")

# Process user query
def process_user_query(user_query):
    filepath = get_filepath_from_model(user_query)
    if not filepath:
        return {"error": "No matching file found for the query."}

    try:
        df = load_csv_to_dataframe(filepath)
        print(f"Loaded file: {filepath}")

        # Example: Filtering intent
        if "enabled" in user_query.lower():
            if "enabled" in df.columns:
                result = df[df["enabled"] == 1]
            else:
                return {"error": "'enabled' column not found in the file."}
        else:
            result = df

        return result.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

