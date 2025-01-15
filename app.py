from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import nltk
import torch
from flask import Flask, render_template, request, jsonify, send_file
import os
import time
from classification_query_to_datapath import process_user_query
from rapidfuzz import process

# Initialize GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Download necessary NLTK data if not already available
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Directory to save temporary CSV files for download
DOWNLOAD_FOLDER = "temp_downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Function to clean up query, remove unwanted values, and filter out small words (<=2 chars)
def clean_query(query):
    unwanted_words = ["the", "is", "in", "on", "for", "and", "to", "of", "me"]  
    tokens = nltk.word_tokenize(query.lower())
    cleaned_tokens = [
        token for token in tokens
        if len(token) > 2 or token == "id"  
        and token not in unwanted_words
    ]
    return cleaned_tokens

# Function to generate GPT-2 response with business intelligence assistant context
def generate_gpt2_response(prompt, max_length=100):
    # Check for simple greetings first
    greetings = ["hey", "hello", "hi", "good morning", "good evening", "howdy", "hi there"]
    if any(greeting in prompt.lower() for greeting in greetings):
        return "Hello! How can I assist you with your data queries today?"

    # If no greeting, proceed with the usual BI assistant context
    bi_assistant_prompt = "You are a business intelligence assistant helping users with data queries. Please respond with a relevant transformation or insight based on the query. Query: "
    input_ids = tokenizer.encode(bi_assistant_prompt + prompt, return_tensors="pt")
    
    # Set pad_token_id to eos_token_id (50256 for GPT-2) to avoid warnings
    pad_token_id = tokenizer.eos_token_id
    
    # Generate the attention mask: 1 for real tokens and 0 for padding tokens
    attention_mask = (input_ids != pad_token_id).type(torch.long)
    
    # Generate response using GPT-2
    output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, attention_mask=attention_mask, pad_token_id=pad_token_id)
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


# Function to check if the query is a greeting
def is_greeting(query):
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'howdy', 'greetings']
    return any(greeting in query.lower() for greeting in greetings)

# Function to process user query using NLTK and generate transformation logic
def generate_transformation(query, available_columns):
    tokens = clean_query(query)
    transformations = {
        "filters": [],
        "sort_by": None,
        "group_by": None,
        "aggregation": 'mean'
    }
    
    classified_columns = classify_columns_from_query(query, available_columns)
    
    for column in classified_columns:
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
        
        if "from" in tokens:
            from_index = tokens.index("from") + 1
            column_name = tokens[from_index]
            if column_name.capitalize() in available_columns:
                value = query.split(column_name)[-1].strip().capitalize()
                if value:
                    transformations["filters"].append((column_name.capitalize(), "=", value))
    
        if "sorted by" in tokens:
            sort_column = column
            if sort_column.capitalize() in available_columns:
                transformations["sort_by"] = sort_column.capitalize()

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
    tokens = clean_query(query)
    relevant_columns = []
    for token in tokens:
        matches = process.extract(token, available_columns, limit=1)
        if matches and isinstance(matches, list):
            for match_info in matches:
                if isinstance(match_info, tuple) and len(match_info) == 3:
                    match, score, token_pos = match_info
                    if int(score) >= 89.0:
                        relevant_columns.append(match)
    return list(set(relevant_columns))

def apply_transformation(df, transformations):
    for column, operator, value in transformations["filters"]:
        if operator == ">":
            df = df[df[column] > value]
        elif operator == "=":
            df = df[df[column] == value]
    
    if transformations["sort_by"]:
        df = df.sort_values(by=transformations["sort_by"])
    
    if transformations["group_by"]:
        if transformations["aggregation"] == 'mean':
            df = df.groupby(transformations["group_by"]).mean().reset_index()
        elif transformations["aggregation"] == 'sum':
            df = df.groupby(transformations["group_by"]).sum().reset_index()
        elif transformations["aggregation"] == 'count':
            df = df.groupby(transformations["group_by"]).count().reset_index()
    
    return df

def get_dataset_response(query):
    if "columns" in query.lower() or "column names" in query.lower():
        filepath = process_user_query(query)
        if not filepath:
            return "No matching file found for the query."
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return f"Error loading dataset: {str(e)}"
        
        if df is None or df.empty:
            return "No data available for the given query."
        
        return jsonify({"columns": df.columns.tolist()})

    filepath = process_user_query(query)
    if not filepath:
        return "No matching file found for the query."
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error loading dataset: {str(e)}"
    
    if df is None or df.empty:
        return "No data available for the given query."
    
    available_columns = df.columns.tolist()
    transformations = generate_transformation(query, available_columns)
    transformed_df = apply_transformation(df, transformations)
    
    if transformed_df.empty:
        return "No data matches the query conditions."

    file_name = f"filtered_data_{int(time.time())}.csv"
    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    transformed_df.to_csv(file_path, index=False)
    return file_path

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    
    # Check if the query is a greeting
    if is_greeting(user_message):
        # Generate a friendly GPT-2 response for the greeting
        gpt2_response = generate_gpt2_response(user_message)
        return jsonify({"response": gpt2_response})
    
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
    return jsonify({
        "response": response_message,
        "download_link": f"/download/{os.path.basename(dataset_response)}"
    })

@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(DOWNLOAD_FOLDER, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    return jsonify({"error": "File not found!"})

@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
