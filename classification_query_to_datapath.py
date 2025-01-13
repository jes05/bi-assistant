import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pickle
from get_common_variables import user_query_data_path, output_path
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('D://Mtech//Semester4//bi-assistant-config.ini')

file_mapping_df = pd.read_csv(output_path)

# Load the training data from CSV
def load_training_data():
    if os.path.exists(user_query_data_path):
        return pd.read_csv(user_query_data_path)
    else:
        raise FileNotFoundError(f"No training data found at: {user_query_data_path}")

# Append new query to the training data CSV
def append_new_query(query, filename):
    # Load the existing training data
    training_df = load_training_data()
    
    # Check if the query already exists
    if query not in training_df['query'].values:
        # Append new query to the DataFrame
        new_data = pd.DataFrame({"query": [query], "filename": [filename]})
        training_df = pd.concat([training_df, new_data], ignore_index=True)
        
        # Save the updated DataFrame back to the CSV
        training_df.to_csv(user_query_data_path, index=False)
        print(f"New query '{query}' added to the training data.")
    else:
        print(f"Query '{query}' already exists in the training data.")

# Train the Bag-of-Words Classifier
def train_classifier():
    # Load training data
    training_df = load_training_data()

    # Ensure the training data has 'query' and 'filename' columns
    if 'query' not in training_df.columns or 'filename' not in training_df.columns:
        raise ValueError("Training data must contain 'query' and 'filename' columns.")

    # Prepare the features and labels
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

    # Get the filename prediction from the classifier
    predicted_filename = get_filepath_from_model(user_query)

    # Check and append the query if new
    append_new_query(user_query, predicted_filename)

    try:     
        if 'settings' not in config.sections():
            config.add_section('settings')  # Add 'settings' section if not exists
        config['settings']['temp_df_path'] = temp_df_path
        with open('D://Mtech//Semester4//bi-assistant-config.ini', 'w') as configfile:
            config.write(configfile)  
        return filepath
    except Exception as e:
        return {"error": str(e)}
