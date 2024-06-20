import os

def count_specific_files_with_details(directory, extensions=('.laz', '.txt', '.xpc')):
    """
    Counts all files within a directory (and its subdirectories) that end with specific extensions
    and provides details about the count and names of files in each directory.

    Parameters:
    - directory (str): The root directory to start counting from.
    - extensions (tuple): A tuple of file extensions to count, in lowercase.

    Returns:
    - dict: A dictionary where each key is a directory path, and the value is another dictionary
            with keys 'count' for the number of files and 'files' for the list of filenames.
    """
    details = {}
    for root, dirs, files in os.walk(directory):
        print(f"Checking directory: {root}")  # Diagnostic print statement
        details[root] = {'count': 0, 'files': []}
        for file in files:
            if file.lower().endswith(extensions):
                details[root]['count'] += 1
                details[root]['files'].append(file)
    return details

directory = r'Q:\50Hertz\Paket_2'
details = count_specific_files_with_details(directory)

for dir_path, info in details.items():
    print(f"Directory: {dir_path}")
    print(f"Total number of .Laz, .txt, and .Xpc files: {info['count']}")
    print("Files:", ", ".join(info['files']))
    print("----------")