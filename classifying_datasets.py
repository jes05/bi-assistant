import os
import pandas as pd
import numpy as np

# Define paths
source_path = "D://Mtech//Semester4//Sample-Datasets"
output_path = "D://Mtech//Semester4//bi-assistant//source-classified-sets//file-mapping.csv"

# Check if the output file already exists
if os.path.exists(output_path):
    processed_df = pd.read_csv(output_path)
    processed_files = set(processed_df['filename'].tolist())
else:
    processed_df = pd.DataFrame()
    processed_files = set()

# Get the list of all files in the source directory
dir_list = os.listdir(source_path)
print("Files and directories in '", source_path, "' :")
print(dir_list)

# Filter out already processed files
new_files = [file for file in dir_list if file not in processed_files]
print("New files to process:", new_files)

# Create a list to store feature summaries for new files
feature_summary = []

# Define dataset categories based on column name patterns
DATASET_CATEGORIES = {
    "Financial": ["price", "cost", "revenue", "transaction", "salary", "income"],
    "User Data": ["user", "name", "email", "gender", "age"],
    "Product Data": ["product", "category", "brand", "sku"],
    "Event Logs": ["event", "log", "timestamp", "incident"],
    "Vendor Data": ["vendor", "supplier"],
    "Cloud Data": ["cloud", "vm", "instance"],
}

def categorize_dataset(columns):
    """
    Categorize dataset based on column names.
    
    Parameters:
        columns (list): List of column names in the dataset.
    
    Returns:
        str: Detected category for the dataset.
    """
    categories_detected = []
    for category, keywords in DATASET_CATEGORIES.items():
        for col in columns:
            if any(keyword in col.lower() for keyword in keywords):
                categories_detected.append(category)
                break
    return ", ".join(categories_detected) if categories_detected else "Uncategorized"

def identify_features(df):
    """
    Automatically identify feature types and other properties in a dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Dictionary summarizing feature categories and properties.
    """
    features = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "binary": [],
        "high_cardinality": [],
        "low_variance": [],
        "missing_values": {}
    }

    # Iterate through each column to classify features
    for col in df.columns:
        # Check for missing values
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            features["missing_values"][col] = missing_count

        # Check column types
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].nunique()

            if unique_vals == 2:
                features["binary"].append(col)
            else:
                # Check variance for numerical columns
                if df[col].var() < 1e-5:
                    features["low_variance"].append(col)
                else:
                    features["numerical"].append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            unique_vals = df[col].nunique()

            if unique_vals > 0.5 * len(df):  # High cardinality threshold
                features["high_cardinality"].append(col)
            else:
                features["categorical"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            features["datetime"].append(col)
        else:
            features["text"].append(col)

    return features

# Process each new file and extract features
for filename in new_files:
    dataset_path = os.path.join(source_path, filename)
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"Processing: {filename}")

        # Identify features
        features_info = identify_features(df)

        # Categorize dataset
        detected_category = categorize_dataset(df.columns)

        # Add features to the summary
        feature_summary.append({
            "filename": filename,
            "category": detected_category,
            "numerical": ", ".join(features_info["numerical"]),
            "categorical": ", ".join(features_info["categorical"]),
            "datetime": ", ".join(features_info["datetime"]),
            "text": ", ".join(features_info["text"]),
            "binary": ", ".join(features_info["binary"]),
            "high_cardinality": ", ".join(features_info["high_cardinality"]),
            "low_variance": ", ".join(features_info["low_variance"]),
            "missing_values": "; ".join([f"{col}: {cnt}" for col, cnt in features_info["missing_values"].items()])
        })
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        feature_summary.append({
            "filename": filename,
            "category": "Error",
            "error": str(e)
        })

# Combine new data with existing data and save to the output file
if feature_summary:
    new_summary_df = pd.DataFrame(feature_summary)
    updated_df = pd.concat([processed_df, new_summary_df], ignore_index=True)
    updated_df.to_csv(output_path, index=False)
    print(f"Feature summary updated and saved to {output_path}")
else:
    print("No new files to process.")
