import os
import pandas as pd
from fuzzywuzzy import process
from get_common_variables import output_path


file_mapping_path = output_path  
file_mapping_df = pd.read_csv(file_mapping_path)

def get_filepath_from_mapping(user_query):
    """
    Identify the target file path based on the user's query using fuzzy matching.
    """
    # Use fuzzy matching to find the best match for the filename
    filenames = file_mapping_df["filename"].tolist()
    best_match, score = process.extractOne(user_query.lower(), filenames)
    
    if score > 70:  # Only consider matches with a high confidence score
        file_row = file_mapping_df[file_mapping_df["filename"] == best_match]
        return file_row["filepath"].values[0]
    
    return None

def load_csv_to_dataframe(filepath):
    """
    Load the specified CSV file into a DataFrame.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        raise FileNotFoundError(f"No CSV file found at path: {filepath}")

def process_user_query(user_query):
    """
    Process the user's query by dynamically loading the relevant CSV 
    and performing operations based on the query intent.
    """
    # Match query to a file path
    filepath = get_filepath_from_mapping(user_query)
    if not filepath:
        return {"error": "No matching file found for the query."}

    try:
        # Load the relevant CSV file
        df = load_csv_to_dataframe(filepath)
        print(f"Loaded file: {filepath}")

        # Example: Process filtering intent
        if "enabled" in user_query.lower():
            if "enabled" in df.columns:
                result = df[df["enabled"] == 1]  # Filter rows where enabled is 1
            else:
                return {"error": "'enabled' column not found in the file."}
        else:
            # Default: Return all rows
            result = df

        # Convert result to JSON for returning to the user
        return result.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}

# Example user queries
if __name__ == "__main__":
    # Query 1
    query1 = "List all card details from cards_data.csv."
    response1 = process_user_query(query1)
    print("Response 1:", response1)

    # Query 2
    query2 = "List all card details from cards-data."
    response2 = process_user_query(query2)
    print("Response 2:", response2)




