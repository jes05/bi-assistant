from rapidfuzz import process
import nltk
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import time
from classification_query_to_datapath import process_user_query

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Directory to save temporary CSV files for download
DOWNLOAD_FOLDER = "temp_downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Function to clean up query, remove unwanted values, and filter out small words (<=2 chars)
def clean_query(query):
    """
    Clean the user query by removing unwanted words and words with size <= 2 (except "id").
    
    Parameters:
        query (str): The user's original query.
    
    Returns:
        list: A list of cleaned tokens.
    """
    unwanted_words = ["the", "is", "in", "on", "for", "and", "to", "of", "me"]  # Add more unwanted words as needed
    tokens = nltk.word_tokenize(query.lower())
    cleaned_tokens = [
        token for token in tokens
        if len(token) > 2 or token == "id"  # Keep "id" even if it's <= 2 in length
        and token not in unwanted_words  # Remove unwanted words
    ]
    print(cleaned_tokens)
    return cleaned_tokens

# Function to process user query using NLTK and generate transformation logic
def generate_transformation(query, available_columns):
    """
    Generate transformation logic (filtering, sorting, group by) based on the user query.
    
    Parameters:
        query (str): The user's natural language query.
        available_columns (list): List of available columns in the dataset.
    
    Returns:
        dict: The conditions or transformations (e.g., filtering, sorting).
    """
    tokens = clean_query(query)  # Clean the query
    transformations = {
        "filters": [],
        "sort_by": None,
        "group_by": None,
        "aggregation": 'mean'  # Default aggregation
    }
    
    # Classify columns from the query using fuzzy logic
    classified_columns = classify_columns_from_query(query, available_columns)
    
    # Process common filter conditions (greater than, less than, equal to, etc.)
    for column in classified_columns:
        # Check for age-like conditions (greater than, less than, etc.)
        if "greater" in tokens or "more" in tokens:
            if "than" in tokens:
                index = tokens.index("than") + 1
                value = tokens[index]
                if value.isdigit():
                    transformations["filters"].append((column, ">", int(value)))
        elif "less" in tokens or "fewer" in tokens:
            if "than" in tokens:
                index = tokens.index("than") + 1
                value = tokens[index]
                if value.isdigit():
                    transformations["filters"].append((column, "<", int(value)))
        elif "equal" in tokens or "equals" in tokens:
            if "to" in tokens:
                index = tokens.index("to") + 1
                value = query.split("to")[-1].strip().capitalize()
                transformations["filters"].append((column, "=", value))
        
        # Handle range-based filtering
        if "between" in tokens:
            try:
                range_start_index = tokens.index("between") + 1
                range_end_index = tokens.index("and") + 1
                start_value = tokens[range_start_index]
                end_value = tokens[range_end_index]
                
                if start_value.isdigit() and end_value.isdigit():
                    transformations["filters"].append((column, "between", (int(start_value), int(end_value))))
            except ValueError:
                pass
        
        # Handle generic filtering: "from [column] [value]"
        if "from" in tokens:
            # Find the column name after "from" and capture the value that follows
            from_index = tokens.index("from") + 1
            column_name = tokens[from_index]
            if column_name.capitalize() in available_columns:
                # Look for the value after the column name
                value = query.split(column_name)[-1].strip().capitalize()
                if value:
                    transformations["filters"].append((column_name.capitalize(), "=", value))
    
        # Extract sorting condition (if any)
        if "sorted by" in tokens:
            sort_column = column
            if sort_column.capitalize() in available_columns:
                transformations["sort_by"] = sort_column.capitalize()

        # Extract group by condition (if any)
        if "group by" in tokens or "count" in tokens or "sum" in tokens or "sum" in tokens:
            group_by_column = column
            if group_by_column in available_columns:
                transformations["group_by"] = group_by_column
                if "sum" in tokens:
                    transformations["aggregation"] = 'sum'
                elif "count" in tokens:
                    transformations["aggregation"] = 'count'
                elif "mean" in tokens:
                    transformations["aggregation"] = 'mean'
    
    return transformations

def classify_columns_from_query(query, available_columns):
    """
    Use fuzzy matching to identify relevant columns based on the user query.
    
    Parameters:
        query (str): The user's query.
        available_columns (list): List of available columns in the dataset.
    
    Returns:
        list: List of columns identified as relevant to the query.
    """
    tokens = clean_query(query)
    relevant_columns = []

    # Use fuzzy logic to find the best matching columns for each token
    for token in tokens:
        # Get the best matches for the token from the available columns
        matches = process.extract(
            token, available_columns, limit=1  # 'limit' specifies how many matches to return
        )
        
        # Ensure matches is a list of tuples (match, score)
        if matches and isinstance(matches, list):
            for match_info in matches:
                print("Test:",match_info, "Length: ", len(match_info))
                if isinstance(match_info, tuple) and len(match_info) == 3:
                    match, score, token_pos = match_info
                    print("Match: ", match, "Score: ", score)
                    if int(score) >= 89.0:
                        print("Match Found:",match," Score: ", score) 
                        relevant_columns.append(match)
                else:
                    print(f"Warning: Unexpected match structure {match_info} for token '{token}'")
        else:
            print(f"Warning: No valid matches found for token '{token}' or matches is not a list.")
    print("relevant_columns: ", match)
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
    # Check if the user wants the column names
    if "columns" in query.lower() or "column names" in query.lower():
        # Get the dataset using the process_user_query function
        filepath = process_user_query(query) 
        print("testing filepath") # Get the file path from the model
        print(filepath)
        if not filepath:
            return "No matching file found for the query."
        
        # Load the dataset
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error loading dataset: {str(e)}"
        
        if df is None or df.empty:
            return "No data available for the given query."
        
        # Return the available columns
        return jsonify({"columns": df.columns.tolist()})

    # Proceed with original logic if columns are not requested
    filepath = process_user_query(query) 
    print("testing filepath") # Get the file path from the model
    print(filepath)
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
    
    # If successful, extract and show the transformations and data
    transformed_df = pd.read_csv(dataset_response)  # Read the transformed dataset
    
    # Generate the response message
    response_message = f"Thank you for your patience. Here is the output:\n\n"
    
    # Show the first few rows of the transformed DataFrame as a preview
    preview = transformed_df.head().to_html(index=False)
    
    response_message += f"Transformed Data (Preview):\n{preview}\n\n"
    
    response_message += f"Your filtered data is ready. Click to download the full dataset."
    print("Response Message: ", response_message)
    return jsonify({
        "response": response_message,
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
