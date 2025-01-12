from flask import Flask, render_template, request, jsonify
import pandas as pd
from llama_cpp import Llama
import configparser

config = configparser.ConfigParser()
config.read('D://Mtech//Semester4//config.ini')
# Initialize Flask app
app = Flask(__name__)

# Load processed data
DATA_FILE = config['FILEPATH']['output_path']
data = pd.read_csv(DATA_FILE)

# Load Llama model
MODEL_PATH = "path/to/your/llama/model.bin"  # Update with your Llama model path
llm = Llama(model_path=MODEL_PATH)

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
    
    # Use Llama model for general conversational responses
    response = llm(user_message)
    return jsonify({"response": response["choices"][0]["text"].strip()})

if __name__ == "__main__":
    app.run(debug=True)
