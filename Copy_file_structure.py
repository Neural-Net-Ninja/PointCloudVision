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
src_dir = r'Q:\50Hertz\Paket_2'
dst_dir = r'Q:\50Hertz\Paket_2\PCT_Converted_Data'

# Copy directory structure
copy_directory_structure(src_dir, dst_dir)