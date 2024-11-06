import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

def visualize_shapefile(file_path):
    gdf = gpd.read_file(file_path)
    gdf.plot()
    plt.title("Visualization of the Shapefile")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def analyze_shapefiles(ground_truth_gdf, predicted_gdf):
    ground_truth_gdf['geometry'] = ground_truth_gdf.buffer(0.001)
    predicted_gdf['geometry'] = predicted_gdf.buffer(0.001)

    true_positives = []
    false_positives = []
    report_data = []

    for idx, predicted_shape in predicted_gdf.iterrows():
        intersects = ground_truth_gdf.intersects(predicted_shape['geometry'])
        if intersects.any():
            true_positives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'True Positive'})
        else:
            false_positives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'False Positive'})

    tp_gdf = gpd.GeoDataFrame(true_positives, columns=predicted_gdf.columns)
    fp_gdf = gpd.GeoDataFrame(false_positives, columns=predicted_gdf.columns)
    report_df = pd.DataFrame(report_data)

    return tp_gdf, fp_gdf, report_df

ground_truth_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\gt'
predicted_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\predict_new'

ground_truth_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.shp')]
predicted_files = [f for f in os.listdir(predicted_folder) if f.endswith('.shp')]

common_files = set(ground_truth_files).intersection(predicted_files)

for file_name in common_files:
    ground_truth_path = os.path.join(ground_truth_folder, file_name)
    predicted_path = os.path.join(predicted_folder, file_name)
    ground_truth_gdf = gpd.read_file(ground_truth_path)
    predicted_gdf = gpd.read_file(predicted_path)
    tp, fp, report = analyze_shapefiles(ground_truth_gdf, predicted_gdf)
    print(f"Results for {file_name}:")
    print(f"True Positives: {len(tp)}")
    print(f"False Positives: {len(fp)}")
    print(report)