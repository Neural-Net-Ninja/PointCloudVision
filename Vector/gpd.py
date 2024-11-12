import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def get_gt_files(filename, class_name="Pfeile", gt_shape_dir=""):
    path = "/".join(filename.split("/")[:-1])
    filename = filename.split("/")[-1]
    # filename = filename.split(" ", 1)[1]  # remove "Run"
    side = "L" if "4532" in filename else "R"
    run_id = filename.split("S", 1)[0]
    part_id = filename.split("_")[-2]
    part_id = part_id.split("part")[1] + "_" if "part" in part_id else ""

    # return fp and fn
    fp_name = f"Run{run_id}_{side}_{part_id}FP_{class_name}_1.shp"
    if not os.path.isfile(gt_shape_dir + "/" + fp_name):
        fp_name = ""

    fn_name = f"Run{run_id}_{side}_{part_id}FN_{class_name}_1.shp"
    if not os.path.isfile(gt_shape_dir + "/" + fp_name):
        fn_name = ""

    if fp_name != "" or fn_name != "":
        print(fp_name, fn_name)
    return fp_name, fn_name


def analyze_shapefiles(ground_truth_fp_gdf, ground_truth_fn_gdf, predicted_gdf):
    ground_truth_fp_gdf['geometry'] = ground_truth_fp_gdf.buffer(0.001)
    ground_truth_fn_gdf['geometry'] = ground_truth_fn_gdf.buffer(0.001)
    predicted_gdf['geometry'] = predicted_gdf.buffer(0.001)

    false_positives = []
    false_negatives = []
    report_data = []

    for idx, predicted_shape in predicted_gdf.iterrows():
        intersects_fp = ground_truth_fp_gdf.intersects(predicted_shape['geometry'])
        intersects_fn = ground_truth_fn_gdf.intersects(predicted_shape['geometry'])
        if intersects_fn.any():
            false_negatives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'False Negative'})
        elif intersects_fp.any():
            false_positives.append(predicted_shape)
            report_data.append({'Object Number': idx, 'Status': 'False Positive'})

    fp_gdf = gpd.GeoDataFrame(false_positives, columns=predicted_gdf.columns)
    fn_gdf = gpd.GeoDataFrame(false_negatives, columns=predicted_gdf.columns)
    report_df = pd.DataFrame(report_data)

    return fp_gdf, fn_gdf, report_df


ground_truth_folder = r'D:\v2'
predicted_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\predict_new'
output_folder = r'D:\Vector_analysis_Twin4road\shape_evaluation\output'

predicted_files = [f for f in os.listdir(predicted_folder) if f.endswith('.shp')]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

overall_fp_count = 0
overall_fn_count = 0
overall_report_data = []

for file_name in predicted_files:
    predicted_path = os.path.join(predicted_folder, file_name)
    fp_name, fn_name = get_gt_files(file_name, gt_shape_dir=ground_truth_folder)
    if fp_name and fn_name:
        ground_truth_fp_path = os.path.join(ground_truth_folder, fp_name)
        ground_truth_fn_path = os.path.join(ground_truth_folder, fn_name)
        ground_truth_fp_gdf = gpd.read_file(ground_truth_fp_path)
        ground_truth_fn_gdf = gpd.read_file(ground_truth_fn_path)
        predicted_gdf = gpd.read_file(predicted_path)
        fp, fn, report = analyze_shapefiles(ground_truth_fp_gdf, ground_truth_fn_gdf, predicted_gdf)
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