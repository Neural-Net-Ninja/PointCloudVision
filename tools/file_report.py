import os
from collections import defaultdict

def generate_tree(directory, prefix=''):
    """
    Generates a tree structure of the given directory.

    :param directory: The directory path.
    :type directory: str
    :param prefix: The prefix for the tree structure.
    :type prefix: str
    :return: Tree structure as a string.
    :rtype: str
    """
    tree = []
    contents = sorted(os.listdir(directory))
    pointers = ['├── '] * (len(contents) - 1) + ['└── ']
    
    for pointer, content in zip(pointers, contents):
        path = os.path.join(directory, content)
        tree.append(prefix + pointer + content)
        if os.path.isdir(path):
            extension = '│   ' if pointer == '├── ' else '    '
            tree.extend(generate_tree(path, prefix + extension))
    
    return tree

def count_files_by_extension(directory):
    """
    Counts files by their extensions in the given directory.

    :param directory: The directory path.
    :type directory: str
    :return: Dictionary with file extensions as keys and counts as values.
    :rtype: dict
    """
    extension_count = defaultdict(int)
    
    for root, _, files in os.walk(directory):
        for file in files:
            extension = os.path.splitext(file)[1]
            extension_count[extension] += 1
    
    return extension_count

def generate_report(directory):
    """
    Generates a report of the directory contents.

    :param directory: The directory path.
    :type directory: str
    :return: Report as a string.
    :rtype: str
    """
    tree = generate_tree(directory)
    extension_count = count_files_by_extension(directory)
    
    report = []
    report.append("Directory Tree:")
    report.extend(tree)
    report.append("\nFile Counts by Extension:")
    for ext, count in extension_count.items():
        report.append(f"{ext}: {count}")
    
    return "\n".join(report)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    if os.path.isdir(directory):
        report = generate_report(directory)
        print(report)
    else:
        print("Invalid directory path.")