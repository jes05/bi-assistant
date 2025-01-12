from flask import Flask, render_template, request, jsonify
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('D://Mtech//Semester4//config.ini')

# Initialize Flask app
app = Flask(__name__)

# Load processed data
DATA_FILE = config['FILEPATH']['output_path']
data = pd.read_csv(DATA_FILE)

# Load Hugging Face model and tokenizer
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Free and open-source model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_response(prompt, max_length=200):
    """
    Generate a response from the model.
    
    Parameters:
        prompt (str): The input prompt for the model.
        max_length (int): Maximum length of the generated response.
    
    Returns:
        str: Generated response text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
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
    if "list datasets" in query.lower():
        return f"Available datasets: {', '.join(data['filename'].unique())}"
    
    if "categories" in query.lower():
        categories = data['category'].unique()
        return f"Dataset categories: {', '.join(categories)}"
    
    if "details for" in query.lower():
        # Extract dataset name from query
        dataset_name = query.lower().replace("details for", "").strip()
        dataset_info = data[data['filename'].str.contains(dataset_name, case=False, na=False)]
        if not dataset_info.empty:
            details = dataset_info.to_dict(orient="records")[0]
            return f"Details for {dataset_name}: {details}"
        return f"No dataset found with name '{dataset_name}'"
    
    return None

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    
    # Try to fetch dataset-specific response
    dataset_response = get_dataset_response(user_message)
    if dataset_response:
        return jsonify({"response": dataset_response})
    
    # Use Hugging Face model for general conversational responses
    response = generate_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
