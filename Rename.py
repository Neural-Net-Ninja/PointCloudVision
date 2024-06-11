import os
import glob

# Specify the directory
directory = r"Q:\50Hertz\Paket_1\ML\00\02_output\L971"

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, '*'))

# Iterate over each file
for file_path in files:
    # Extract the file name from the path
    file_name = os.path.basename(file_path)
    # Initialize a variable to check if the file name was modified
    modified = False

    # Check if "_preprocessed" is in the file name
    if '_preprocessed' in file_name:
        # Get the new name by removing '_preprocessed' from the file name
        file_name = file_name.replace('_preprocessed', '')
        modified = True

    # Check if "density_reduced" is in the file name
    if "density_reduced" in file_name:
        # Construct a new file name by replacing "density_reduced" with an empty string
        file_name = file_name.replace("density_reduced", "")
        modified = True

    # Rename the file if it was modified
    if modified:
        new_file_path = os.path.join(directory, file_name)
        os.rename(file_path, new_file_path)