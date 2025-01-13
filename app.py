import nltk
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import time
import pandasql as ps
from classification_query_to_datapath import process_user_query
import joblib  # Assuming the classifier is saved as a .pkl file

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Directory to save temporary CSV files for download
DOWNLOAD_FOLDER = "temp_downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Load the pre-trained classification model (assuming it's a pickle file)
# The model classifies query terms into column names
model = joblib.load('models/column_classifier.pkl')

# Function to process user query using NLTK and generate transformation logic
def generate_transformation(query, available_columns):
    """
    Generate transformation logic (filtering, sorting, group by) based on the user query.
    
    Parameters:
        query (str): The user's natural language query.
        available_columns (list): List of available columns in the dataset.
    
    Returns:
        dict: The conditions or transformations (e.g., filtering, sorting, group by).
    """
    tokens = nltk.word_tokenize(query.lower())
    transformations = {
        "filters": [],
        "sort_by": None,
        "group_by": None,
        "aggregation": 'mean'  # Default aggregation
    }
    
    # Classify columns from the query using the model
    classified_columns = classify_columns_from_query(query, available_columns)
    
    # Extract filter conditions using the classified columns
    for column in classified_columns:
        if "older" in tokens and "than" in tokens:
            age_index = tokens.index("than") + 1
            age = int(tokens[age_index])
            transformations["filters"].append((column, ">", age))
        elif "from" in tokens and "city" in tokens:
            city_index = tokens.index("city") + 1
            city = query.split("city")[-1].strip().capitalize()
            transformations["filters"].append((column, "=", city))
    
    # Extract sorting condition (if any)
    if "sorted by" in tokens:
        sort_index = tokens.index("sorted by") + 2
        sort_column = tokens[sort_index]
        if sort_column.capitalize() in available_columns:
            transformations["sort_by"] = sort_column.capitalize()

    # Extract group by condition (if any)
    if "group by" in tokens:
        group_by_index = tokens.index("group by") + 2
        group_by_column = tokens[group_by_index].capitalize()
        if group_by_column in available_columns:
            transformations["group_by"] = group_by_column
    
    # Handle optional aggregation (sum, count, etc.)
    if "sum" in tokens:
        transformations["aggregation"] = 'sum'
    elif "count" in tokens:
        transformations["aggregation"] = 'count'
    
    return transformations

def classify_columns_from_query(query, available_columns):
    """
    Use the classification model to identify relevant columns based on the user query.
    
    Parameters:
        query (str): The user's query.
        available_columns (list): List of available columns in the dataset.
    
    Returns:
        list: List of columns identified as relevant to the query.
    """
    tokens = nltk.word_tokenize(query.lower())
    relevant_columns = []
    
    # Use the model to predict relevant columns
    for token in tokens:
        if token in available_columns:  # Simple direct match with available columns
            relevant_columns.append(token)
        else:
            # Use the classifier to predict the column(s) related to the token
            predicted_column = model.predict([token])[0]  # Assuming model.predict() returns column names
            if predicted_column in available_columns:
                relevant_columns.append(predicted_column)
    
    return list(set(relevant_columns))  # Remove duplicates

def apply_transformation(df, transformations):
    """
    Apply transformations (filters, sorting, group by) to the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        transformations (dict): The transformation details (filtering, sorting).
    
    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Apply filters
    for column, operator, value in transformations["filters"]:
        if operator == ">":
            df = df[df[column] > value]
        elif operator == "=":
            df = df[df[column] == value]
    
    # Apply sorting
    if transformations["sort_by"]:
        df = df.sort_values(by=transformations["sort_by"])
    
    # Apply group by and aggregation
    if transformations["group_by"]:
        # Check if there's an aggregation and apply it
        if transformations["aggregation"] == 'mean':
            df = df.groupby(transformations["group_by"]).mean().reset_index()
        elif transformations["aggregation"] == 'sum':
            df = df.groupby(transformations["group_by"]).sum().reset_index()
        elif transformations["aggregation"] == 'count':
            df = df.groupby(transformations["group_by"]).count().reset_index()
    
    return df

def get_dataset_response(query):
    """
    Process user queries and return filtered data.
    
    Parameters:
        query (str): The user's query.
    
    Returns:
        str: File path to the filtered data or error message.
    """
    # Get the dataset using the process_user_query function
    filepath = process_user_query(query)  # Get the file path from the model
    
    if not filepath:
        return "No matching file found for the query."
    
    # Load the dataset
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"
    
    if df is None or df.empty:
        return "No data available for the given query."
    
    # Get available columns from the dataset
    available_columns = df.columns.tolist()

    # Generate the transformation logic
    transformations = generate_transformation(query, available_columns)
    print(f"Generated Transformations: {transformations}")

    # Apply the transformation to the DataFrame
    transformed_df = apply_transformation(df, transformations)
    
    if transformed_df.empty:
        return "No data matches the query conditions."

    # Generate a unique filename based on the current timestamp
    file_name = f"filtered_data_{int(time.time())}.csv"
    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    transformed_df.to_csv(file_path, index=False)
    print(transformed_df)
    return file_path

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    
    # Process the user query and get the filtered dataset
    dataset_response = get_dataset_response(user_message)
    
    if isinstance(dataset_response, str):  # If there is an error
        return jsonify({"response": dataset_response})
    
    # If successful, provide a link to download the CSV
    return jsonify({
        "response": "Your filtered data is ready. Click to download.",
        "download_link": f"/download/{os.path.basename(dataset_response)}"
    })

@app.route("/download/<filename>")
def download(filename):
    """
    Serve the filtered CSV file for download.
    
    Parameters:
        filename (str): The name of the CSV file to download.
    
    Returns:
        File: The CSV file to be downloaded.
    """
    file_path = os.path.join(DOWNLOAD_FOLDER, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    return jsonify({"error": "File not found!"})

@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
