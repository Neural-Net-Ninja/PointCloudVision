import os

def count_specific_files(directory, extensions=('.laz', '.txt', '.xpc')):  # Lowercase extensions
    """
    Counts all files within a directory (and its subdirectories) that end with specific extensions.

    Parameters:
    - directory (str): The root directory to start counting from.
    - extensions (tuple): A tuple of file extensions to count, in lowercase.

    Returns:
    - int: The total count of files matching the specified extensions.
    """
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):  # Convert filename to lowercase before checking
                count += 1
    return count

# Example usage
directory = r'Q:\50Hertz\Paket_2'
file_count = count_specific_files(directory)
print(f"Total number of .Laz, .txt, and .Xpc files: {file_count}")