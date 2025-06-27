import os
import shutil
import math
from multiprocessing import Pool
from functools import partial

def copy_file(args):
    src_file, dst_file = args
    try:
        shutil.copy2(src_file, dst_file)  # Copy file with metadata
        print(f"Copied {os.path.basename(src_file)} to {os.path.basename(os.path.dirname(dst_file))}")
    except Exception as e:
        print(f"Error copying {os.path.basename(src_file)}: {e}")

def split_files_into_folders(source_folder, max_files_per_folder=100):
    # Get list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    total_files = len(files)
    
    if total_files == 0:
        print("No files found in the source folder.")
        return
    
    # Calculate number of folders needed
    num_folders = math.ceil(total_files / max_files_per_folder)
    
    # Get base folder name from source folder
    base_folder_name = os.path.basename(os.path.normpath(source_folder))
    
    # Define the parent folder for splits (one level up from source folder)
    parent_folder = os.path.dirname(source_folder)
    splits_folder = os.path.join(parent_folder, f"splits_{base_folder_name}")
    
    # Create the splits_<source_folder_name> directory if it doesn't exist
    os.makedirs(splits_folder, exist_ok=True)
    
    # Prepare copy tasks
    copy_tasks = []
    for i in range(num_folders):
        # Create folder name with padded number (e.g., L_C_559_01)
        folder_name = f"{base_folder_name}_{str(i+1).zfill(2)}"
        folder_path = os.path.join(splits_folder, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Calculate start and end index for files to copy
        start_idx = i * max_files_per_folder
        end_idx = min((i + 1) * max_files_per_folder, total_files)
        
        # Add copy tasks for this folder
        for file_idx in range(start_idx, end_idx):
            src_file = os.path.join(source_folder, files[file_idx])
            dst_file = os.path.join(folder_path, files[file_idx])
            copy_tasks.append((src_file, dst_file))
    
    # Parallelize file copying using multiprocessing
    with Pool() as pool:
        pool.map(copy_file, copy_tasks)
    
    print(f"Completed! Files copied into {num_folders} folders in {splits_folder}.")

if __name__ == "__main__":
    # Hardcoded source folder path (copy-paste your path here)
    source_folder = r'K:\Spie\00_training_data_16_04_2025\Prediction_data\pct_format\Paket_3\L_C_559'  # Example path, replace with your own
    split_files_into_folders(source_folder, max_files_per_folder=100)