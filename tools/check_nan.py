import os
import pandas as pd
import numpy as np

def check_nan_inf_in_file(file_path):
    """
    Check for NaN and Inf values in a file.
    Supports CSV and Excel files.
    """
    try:
        if file_path.endswith('.txt'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print(f"Skipping unsupported file: {file_path}")
            return

        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"File: {file_path} contains {nan_count} NaN values.")
        else:
            print(f"File: {file_path} does not contain any NaN values.")

        # Check for Inf values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"File: {file_path} contains {inf_count} Inf values.")
        else:
            print(f"File: {file_path} does not contain any Inf values.")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def check_nan_inf_in_directory(directory):
    """
    Recursively check all files in a directory and its subdirectories for NaN and Inf values.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            check_nan_inf_in_file(file_path)

if __name__ == "__main__":
    # Specify the directory to check
    directory_to_check = r"Q:/FruitS/ML/pomegranate/1/00_labeled_txt"

    # Perform the NaN/Inf check
    check_nan_inf_in_directory(directory_to_check)