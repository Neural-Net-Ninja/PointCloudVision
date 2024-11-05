import os
import shutil

ground_truth_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\gt'
predicted_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\predict'
output_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\predict_new'

ground_truth_files = {f for f in os.listdir(ground_truth_folder) if f.endswith('.shp')}
predicted_files = {f for f in os.listdir(predicted_folder) if f.endswith('.shp')}

common_files = ground_truth_files.intersection(predicted_files)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in common_files:
    source_path = os.path.join(predicted_folder, file_name)
    destination_path = os.path.join(output_folder, file_name)
    shutil.copy(source_path, destination_path)