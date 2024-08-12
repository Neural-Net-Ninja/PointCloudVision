import os
import shutil

def copy_directory_structure(src_dir, dst_dir):
    # Iterate over all directories and files in source directory
    for dirpath, dirnames, filenames in os.walk(src_dir):
        # Compute destination path
        dst_path = dirpath.replace(src_dir, dst_dir, 1)

        # If destination path does not exist, create it
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)


# Define your directories
src_dir = r'K:\Spie\Paket_4\ML_paket_4'
dst_dir = r'K:\Spie\Paket_5\ML_paket_5'

# Copy directory structure
copy_directory_structure(src_dir, dst_dir)