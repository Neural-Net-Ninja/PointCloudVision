import os
import geopandas as gpd

ground_truth_folder = 'path/to/ground_truth_folder'
predicted_folder = 'path/to/predicted_folder'

ground_truth_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.shp')]
predicted_files = [f for f in os.listdir(predicted_folder) if f.endswith('.shp')]

common_files = set(ground_truth_files).intersection(predicted_files)

def analyze_shapefiles(ground_truth_gdf, predicted_gdf):
    ground_truth_gdf['geometry'] = ground_truth_gdf.buffer(0.001)
    predicted_gdf['geometry'] = predicted_gdf.buffer(0.001)
    joined_gdf = gpd.sjoin(predicted_gdf, ground_truth_gdf, how='left', op='intersects')
    joined_gdf['TP'] = joined_gdf['index_right'].notnull()
    joined_gdf['FP'] = joined_gdf['index_right'].isnull()
    tp = joined_gdf[joined_gdf['TP']]
    fp = joined_gdf[joined_gdf['FP']]
    return tp, fp

for file_name in common_files:
    ground_truth_path = os.path.join(ground_truth_folder, file_name)
    predicted_path = os.path.join(predicted_folder, file_name)
    ground_truth_gdf = gpd.read_file(ground_truth_path)
    predicted_gdf = gpd.read_file(predicted_path)
    tp, fp = analyze_shapefiles(ground_truth_gdf, predicted_gdf)
    print(f"Results for {file_name}:")
    print(f"True Positives: {len(tp)}")
    print(f"False Positives: {len(fp)}")
    print()
    tp.to_file(f"results/tp_{file_name}")
    fp.to_file(f"results/fp_{file_name}")