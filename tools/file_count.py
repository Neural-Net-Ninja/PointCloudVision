import os

def count_specific_files(directory, extensions=('.laz')):  # '.txt', '.xpc'
    """
    Counts all files within a directory (and its subdirectories) that end with specific extensions,
    and prints the count for each folder with only the last folder name.
    The main folder name is printed centered at the top.

    Parameters:
    - directory (str): The root directory to start counting from.
    - extensions (tuple): A tuple of file extensions to count, in lowercase.

    Returns:
    - None
    """
    folder_counts = {}  # Dictionary to hold counts per folder
    total_count = 0  # Variable to hold the total count

    main_folder_name = os.path.basename(directory)
    print(f"{main_folder_name.center(50)}")  # Center the main folder name

    for root, dirs, files in os.walk(directory):
        count = sum(1 for file in files if file.lower().endswith(extensions))
        if count > 0:
            last_folder_name = os.path.basename(root)  # Get the last part of the folder path
            folder_counts[last_folder_name] = count
            total_count += count

    # Print the count for each folder
    for folder, count in folder_counts.items():
        print(f"{folder}: {count} files")

    # Print the total count
    print(f"Total number of .laz, .txt, and .xpc files: {total_count}")

# Example usage
directory = r'K:\Spie\Paket_3'
count_specific_files(directory)