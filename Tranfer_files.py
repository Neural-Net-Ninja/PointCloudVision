import os
import shutil

# Define your directories
dir_A = 'Q:/50Hertz/Grossbeeren-Thyrow/ML_00/02_output'
dir_B = 'Q:/50Hertz/Grossbeeren-Thyrow/Grossbeeren-Thyrow'
dir_C = 'Q:/50Hertz/Grossbeeren-Thyrow/ML_00/Full_density_label'

# Get list of file base names in directory A (without file extensions)
files_in_A = [os.path.splitext(filename)[0] for filename in os.listdir(dir_A)]

# Iterate over files in directory B
for filename in os.listdir(dir_B):
    # If a file base name in B is in the list from A, copy it to C
    if os.path.splitext(filename)[0] in files_in_A:
        shutil.copy(os.path.join(dir_B, filename), dir_C)