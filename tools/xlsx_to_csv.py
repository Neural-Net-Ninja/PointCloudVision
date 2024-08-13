import os
import pandas as pd


def convert_xlsx_to_csv(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all .xlsx files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(input_dir, filename)
            # Load the Excel file
            data = pd.read_excel(file_path)
            
            # Replace German-style commas with dots
            data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
            
            # Define the output CSV file path
            csv_filename = filename.replace('.xlsx', '_converted.csv')
            csv_file_path = os.path.join(output_dir, csv_filename)
            
            # Save the modified DataFrame as a CSV file
            data.to_csv(csv_file_path, index=False, sep=',')
            print(f"Converted {filename} to {csv_filename}")


# Example usage
input_directory = r'K:\Fruit_detection\10687819'
output_directory = r'K:\Fruit_detection\PCT'
convert_xlsx_to_csv(input_directory, output_directory)
