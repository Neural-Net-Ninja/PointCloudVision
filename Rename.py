import os
import glob

# Specify the directory
directory = r"Q:\50Hertz\Paket_1\ML\00\02_output\L951"

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, '*_preprocessed*'))

# Iterate over each file
for file in files:
    # Get the new name by removing '_preprocessed' from the file name
    new_name = file.replace('_preprocessed', '')
    
    # Rename the file
    os.rename(file, new_name)
    

import os
import glob

# Define the directory path
dir_path = r"Q:\50Hertz\Paket_1\ML\00\02_output\L951"

# Get a list of all files in the directory
files = glob.glob(os.path.join(dir_path, '*'))

# Iterate over the list of files
for file in files:
    # Check if "density_reduced" is in the filename
    if "density_reduced" in file:
        # Construct a new filename by replacing "density_reduced" with an empty string
        new_filename = file.replace("density_reduced", "")
        # Rename the file
        os.rename(file, new_filename)