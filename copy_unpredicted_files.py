import os
import shutil

def get_file_names_without_extension(folder):
    return {os.path.splitext(file)[0] for file in os.listdir(folder)}

def copy_unpredicted_files(source_folder, predicted_folder, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Get the base names (without extensions) of files in the source and predicted folders
    source_files = get_file_names_without_extension(source_folder)
    predicted_files = get_file_names_without_extension(predicted_folder)
    
    # Determine the base names of files that have not been predicted yet
    unpredicted_files = source_files - predicted_files
    
    # Copy unpredicted files to the destination folder
    for file_name in os.listdir(source_folder):
        base_name = os.path.splitext(file_name)[0]
        if base_name in unpredicted_files:
            src_file = os.path.join(source_folder, file_name)
            dest_file = os.path.join(destination_folder, file_name)
            shutil.copy(src_file, dest_file)
            print(f"Copied {file_name} to {destination_folder}")


# Define the paths to your folders
folder_a = r'K:\Spie\Paket_3\ML_paket_3\01_preprocessed\02_predict\L487'
folder_b = r'K:\Spie\Paket_3\ML_paket_3\02_output\L487'
folder_c = r'K:\Spie\Paket_3\ML_paket_3\01_preprocessed\02_predict\Un_L487'

# Copy the unpredicted files
copy_unpredicted_files(folder_a, folder_b, folder_c)
