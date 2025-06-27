import os

# Define the folder path
folder = r'C:\Machine_learning\Run_training\02_predict\Paket_2\L_C_501'

# Get base names of files in the folder
for file in os.listdir(folder):
    if os.path.isfile(os.path.join(folder, file)) and not file.startswith('.'):
        base_name = os.path.splitext(file)[0]
        print(base_name)