import os


def count_specific_files(directory, extensions=('.txt', '.xpc', '.laz', '.las', '.csv')):

    folder_counts = {}
    total_counts = {ext: 0 for ext in extensions}

    main_folder_name = os.path.basename(directory)
    print(f"\n{main_folder_name.center(50)}\n")

    for root, dirs, files in os.walk(directory):
        last_folder_name = os.path.basename(root)
        folder_counts[last_folder_name] = {ext: 0 for ext in extensions}

        for file in files:
            for ext in extensions:
                if file.lower().endswith(ext):
                    folder_counts[last_folder_name][ext] += 1
                    total_counts[ext] += 1

    for folder, counts in folder_counts.items():
        if any(count > 0 for count in counts.values()):
            print(f"{folder}:")
            for ext in extensions:
                count = counts[ext]
                if count > 0:
                    print(f"  {ext}: {count} files")

    print("\nTotal counts:")
    for ext in extensions:
        print(f"{ext}: {total_counts[ext]} files")
    print(f"Total number of files: {sum(total_counts.values())}")

directory = r'K:\Spie\00_training_data_16_04_2025\Paket_2\Paket_2'
count_specific_files(directory)