import os
import shutil


def get_file_names_without_extension(folder):
    return {os.path.splitext(file)[0] for file in os.listdir(folder)}

def copy_unpredicted_files(source_folder, predicted_folder, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the base names (without extensions) of files in the source and predicted folders
    source_files = get_file_names_without_extension(source_folder)
    predicted_files = get_file_names_without_extension(predicted_folder)

    # Determine the base names of files that have not been predicted yet
    unpredicted_files = source_files - predicted_files

    # Total number of files to copy
    total_files = len(unpredicted_files)
    copied_files = 0

    # Copy unpredicted files to the destination folder
    for file_name in os.listdir(source_folder):
        base_name = os.path.splitext(file_name)[0]
        if base_name in unpredicted_files:
            src_file = os.path.join(source_folder, file_name)
            dest_file = os.path.join(destination_folder, file_name)
            shutil.copy(src_file, dest_file)
            copied_files += 1
            # Calculate and print the percentage of completion
            percentage_complete = (copied_files / total_files) * 100
            print(f"Copying files: {percentage_complete:.2f}% complete", end='\r')

    print("\nCopying complete.")


# Define the paths to your folders
all_files = r'K:\Spie\Spie_files\Spie_ML\01_preprocessed\paket_6_before_cut\L315'
predicted_files = r'K:\Spie\Spie_files\Spie_ML\02_output\paket_6\L315'
yet_to_be_predicted = r'K:\Spie\Spie_files\Spie_ML\01_preprocessed\delete'

# Count the number of files in the all_files folder
all_files_count = len([name for name in os.listdir(all_files) if os.path.isfile(os.path.join(all_files, name))])
print(f"Number of files in all_files folder: {all_files_count}")

# Count the number of files in the predicted_files folder
predicted_files_count = len([name for name in os.listdir(predicted_files) if os.path.isfile(os.path.join(predicted_files, name))])
print(f"Number of files in predicted_files folder: {predicted_files_count}")

yet_to_be_predicted_count = all_files_count - predicted_files_count
print(f"Number of files to be predicted: {yet_to_be_predicted_count}")

# Copy the unpredicted files
copy_unpredicted_files(all_files, predicted_files, yet_to_be_predicted)