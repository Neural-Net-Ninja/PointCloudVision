import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def analyze_shapefiles(ground_truth_gdf, predicted_gdf):
    ground_truth_gdf['geometry'] = ground_truth_gdf.buffer(0.001)
    predicted_gdf['geometry'] = predicted_gdf.buffer(0.001)

    false_positives = []
    false_negatives = []
    report_data = []

    for idx, predicted_shape in predicted_gdf.iterrows():
        intersects = ground_truth_gdf.intersects(predicted_shape['geometry'])
        if intersects.any():
            false_negatives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'False Negative'})
        else:
            false_positives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'False Positive'})

    fp_gdf = gpd.GeoDataFrame(false_positives, columns=predicted_gdf.columns)
    fn_gdf = gpd.GeoDataFrame(false_negatives, columns=predicted_gdf.columns)
    report_df = pd.DataFrame(report_data)

    return fp_gdf, fn_gdf, report_df


ground_truth_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\gt'
predicted_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\predict_new'
output_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\output'

ground_truth_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.shp')]
predicted_files = [f for f in os.listdir(predicted_folder) if f.endswith('.shp')]

common_files = set(ground_truth_files).intersection(predicted_files)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

overall_fp_count = 0
overall_fn_count = 0
overall_report_data = []

for file_name in common_files:
    ground_truth_path = os.path.join(ground_truth_folder, file_name)
    predicted_path = os.path.join(predicted_folder, file_name)
    ground_truth_gdf = gpd.read_file(ground_truth_path)
    predicted_gdf = gpd.read_file(predicted_path)
    fp, fn, report = analyze_shapefiles(ground_truth_gdf, predicted_gdf)
    overall_fp_count += len(fp)
    overall_fn_count += len(fn)
    overall_report_data.extend(report.to_dict('records'))
    print(f"Results for {file_name}:")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")

total_objects = overall_fp_count + overall_fn_count
metrics = {
    'Total Objects': total_objects,
    'False Positives': overall_fp_count,
    'False Negatives': overall_fn_count
}

print("Overall Results:")
print(metrics)