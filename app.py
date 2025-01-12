import flask
from flask import Flask, render_template, request, jsonify
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('D://Mtech//Semester4//bi-assistant-config.ini')

# Initialize Flask app
app = Flask(__name__)

# Load processed data
DATA_FILE = config['FILEPATH']['output_path']
main_path = config['FILEPATH']['main_path']
data = pd.read_csv(DATA_FILE)

# Load Hugging Face model and tokenizer
#MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Free and open-source model
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL_NAME = "gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

def generate_response(prompt, max_length=200, temperature=0.7):
    """
    Generate a response from the model with focused behavior.
    
    Parameters:
        prompt (str): The input prompt for the model.
        max_length (int): Maximum length of the generated response.
        temperature (float): Controls the randomness of the output.
    
    Returns:
        str: Generated response text.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id as pad_token_id
        attention_mask=inputs.attention_mask,
        temperature=temperature,  # Adjusting temperature for more focused responses
        top_p=0.9,  # Adjusting to limit randomness further
        top_k=50, 
        do_sample=True,
        no_repeat_ngram_size=2  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_dataset_response(query):
    """
    Process user queries related to the dataset.
    
    Parameters:
        query (str): User's question.
    
    Returns:
        str: Response based on the dataset.
    """
    query = query.lower()

    # Check if the query is asking for dataset names
    if "list datasets" in query:
        return f"Available datasets: {', '.join(data['filename'].unique())}"

    # Check if the query is asking for dataset categories
    if "categories" in query:
        categories = data['category'].unique()
        return f"Dataset categories: {', '.join(categories)}"
    
    # Check if the query asks for details about a specific dataset
    if "details for" in query:
        dataset_name = query.replace("details for", "").strip()
        dataset_info = data[data['filename'].str.contains(dataset_name, case=False, na=False)]
        if not dataset_info.empty:
            details = dataset_info.to_dict(orient="records")[0]
            return f"Details for {dataset_name}: {details}"
        return f"No dataset found with name '{dataset_name}'"
    
    # Check for user queries asking about specific data points (i.e., data values in rows)
    if "data from" in query:
        dataset_name = query.replace("data from", "").strip()
        dataset_info = data[data['filename'].str.contains(dataset_name, case=False, na=False)]
        if not dataset_info.empty:
            # Returning top 5 rows of the dataset as an example
            return f"Here are the first 5 rows from {dataset_name}: {dataset_info.head().to_dict(orient='records')}"
        return f"No data found for '{dataset_name}'"

    return None

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    
    # Check for greetings
    greeting_keywords = ["hi", "hello", "hey", "howdy", "greetings"]
    if any(greeting in user_message.lower() for greeting in greeting_keywords):
        return jsonify({"response": "Hey, how can I help you?"})
    
    # Try to fetch dataset-specific response
    dataset_response = get_dataset_response(user_message)
    if dataset_response:
        return jsonify({"response": dataset_response})
    
    # Use Hugging Face model for general conversational responses with controlled temperature
    response = generate_response(user_message, temperature=0.7)
    return jsonify({"response": response})

@app.route("/")
def home():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)
