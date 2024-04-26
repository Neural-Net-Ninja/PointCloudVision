import os
import glob

# Specify the directory
directory = 'Q:/50Hertz/Grossbeeren-Thyrow/prediction/webviz_spie'

# Get a list of all files in the directory
files = glob.glob(os.path.join(directory, '*_preprocessed*'))

# Iterate over each file
for file in files:
    # Get the new name by removing '_preprocessed' from the file name
    new_name = file.replace('_preprocessed', '')
    
    # Rename the file
    os.rename(file, new_name)