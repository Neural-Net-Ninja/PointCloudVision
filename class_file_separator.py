import os
import shutil

# Set the paths to the folders
pcr_separated = "/path/to/folder/A"
all_files = "/path/to/folder/B"
class_seprated_files = "/path/to/folder/C"

# Get a list of all files in folder A
pcr_separated = os.listdir(pcr_separated)

# Check each file in folder A against folder B and copy to folder C if it exists
for file in pcr_separated:
    if os.path.isfile(os.path.join(all_files, file)):
        shutil.copy(os.path.join(pcr_separated, file), class_seprated_files)