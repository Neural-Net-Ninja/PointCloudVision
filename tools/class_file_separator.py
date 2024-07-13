import os
import shutil

# Set the paths to the folders
folder_a = "/path/to/folder/A"
folder_b = "/path/to/folder/B"
folder_c = "/path/to/folder/C"

# Get a list of all files in folder A
files_a = os.listdir(folder_a)

# Check each file in folder A against folder B and copy to folder C if it exists
for file_a in files_a:
    if os.path.isfile(os.path.join(folder_b, file_a)):
        shutil.copy(os.path.join(folder_a, file_a), folder_c)