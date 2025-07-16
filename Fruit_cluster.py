import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

def is_point_cloud_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = {'X', 'Y', 'Z', 'Intensity', 'Temperature', 'FruitID'}
        return required_columns.issubset(df.columns)
    except:
        return False

def find_csv_files(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and is_point_cloud_csv(os.path.join(root, file)):
                csv_files.append(os.path.join(root, file))
    return csv_files

def cluster_fruit_data(df, n_clusters=4):
    # Filter out FruitID 0
    fruit_data = df[df['FruitID'] != 0].copy()

    # Perform clustering on Temperature
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    fruit_data['ClusterID'] = kmeans.fit_predict(fruit_data[['Temperature']])

    # Assign cluster IDs back to original dataframe
    df['ClusterID'] = -1  # Default for non-fruit points
    df.loc[fruit_data.index, 'ClusterID'] = fruit_data['ClusterID']

    return df, kmeans

def create_cluster_plot(df, output_path, filename):
    plt.figure(figsize=(10, 6))
    fruits = df[df['FruitID'] != 0]

    for cluster_id in range(4):
        cluster_data = fruits[fruits['ClusterID'] == cluster_id]
        plt.scatter(cluster_data['FruitID'], cluster_data['Temperature'],
                    label=f'Cluster {cluster_id}', alpha=0.6)

    plt.xlabel('Fruit ID')
    plt.ylabel('Temperature')
    plt.title(f'Temperature Clusters for {filename}')
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{filename}.png"))
    plt.close()

def process_point_cloud_files(root_dir):
    # Create output directories
    output_base = os.path.join(root_dir, 'processed')
    plot_dir = os.path.join(output_base, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Find all valid CSV files
    csv_files = find_csv_files(root_dir)

    for file_path in csv_files:
        # Read CSV
        df = pd.read_csv(file_path)

        # Perform clustering
        df_clustered, _ = cluster_fruit_data(df)

        # Create output path
        rel_path = os.path.relpath(file_path, root_dir)
        output_dir = os.path.join(output_base, os.path.dirname(rel_path))
        os.makedirs(output_dir, exist_ok=True)

        # Save clustered data
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_cluster.csv")
        df_clustered.to_csv(output_file, index=False)

        # Create and save plot
        create_cluster_plot(df_clustered, plot_dir, os.path.splitext(os.path.basename(file_path))[0])

def main():
    root_directory = "path/to/your/data"  # Replace with actual path
    process_point_cloud_files(root_directory)

if __name__ == "__main__":
    main()