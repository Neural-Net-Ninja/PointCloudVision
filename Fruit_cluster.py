import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path


def is_point_cloud_csv(file_path):
    """Check if CSV file contains required point cloud columns."""
    try:
        df = pd.read_csv(file_path)
        required_columns = {'X', 'Y', 'Z', 'Intensity', 'Temperature', 'FruitID'}
        return required_columns.issubset(df.columns)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def find_csv_files(root_dir):
    """Recursively find all valid point cloud CSV files."""
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if is_point_cloud_csv(file_path):
                    csv_files.append(file_path)
    return csv_files


def cluster_fruit_data(df, n_clusters=4):
    """Perform clustering on fruit temperature data."""
    # Filter out FruitID 0 (non-fruit points)
    fruit_data = df[df['FruitID'] != 0].copy()

    if len(fruit_data) == 0:
        print("No fruit data found (all FruitID = 0)")
        df['ClusterID'] = -1
        return df, None

    # Perform clustering on Temperature attribute for all fruit points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    fruit_data['ClusterID'] = kmeans.fit_predict(fruit_data[['Temperature']])

    # Assign cluster IDs back to original dataframe
    df['ClusterID'] = -1  # Default for non-fruit points (FruitID = 0)
    df.loc[fruit_data.index, 'ClusterID'] = fruit_data['ClusterID']

    return df, kmeans


def create_cluster_plot(df, output_path, filename):
    """Create and save clustering visualization."""
    try:
        plt.figure(figsize=(12, 8))
        fruits = df[df['FruitID'] != 0]

        if len(fruits) == 0:
            print(f"No fruit data to plot for {filename}")
            return

        # Get unique clusters
        unique_clusters = sorted(fruits['ClusterID'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = fruits[fruits['ClusterID'] == cluster_id]
            plt.scatter(cluster_data['FruitID'], cluster_data['Temperature'], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=20)

        plt.xlabel('Fruit ID')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Temperature Clusters for {filename}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved: {os.path.join(output_path, f'{filename}.png')}")

    except Exception as e:
        print(f"Error creating plot for {filename}: {e}")


def process_point_cloud_files(root_dir):
    """Process all point cloud files in the directory structure."""
    # Find all valid CSV files
    csv_files = find_csv_files(root_dir)

    if not csv_files:
        print("No valid point cloud CSV files found!")
        return

    print(f"Found {len(csv_files)} valid point cloud files")

    for file_path in csv_files:
        try:
            print(f"\nProcessing: {file_path}")

            # Read CSV
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} points")

            # Perform clustering
            df_clustered, kmeans_model = cluster_fruit_data(df)

            # Create output paths with new folder structure
            rel_path = os.path.relpath(file_path, root_dir)
            path_parts = Path(rel_path).parts
            filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]

            # Create new folder structure with "_cluster" suffix
            if len(path_parts) > 1:
                # If file is in subdirectories, add "_cluster" to the immediate parent folder
                parent_dir = os.path.dirname(root_dir)
                original_folder_name = os.path.basename(root_dir)
                new_root_dir = os.path.join(parent_dir, f"{original_folder_name}_cluster")

                # Maintain subdirectory structure
                rel_dir = os.path.dirname(rel_path)
                output_data_dir = os.path.join(new_root_dir, rel_dir)
                output_plot_dir = os.path.join(new_root_dir, rel_dir, 'plots')
            else:
                # If file is directly in root, create new root with "_cluster" suffix
                parent_dir = os.path.dirname(root_dir)
                original_folder_name = os.path.basename(root_dir)
                new_root_dir = os.path.join(parent_dir, f"{original_folder_name}_cluster")

                output_data_dir = new_root_dir
                output_plot_dir = os.path.join(new_root_dir, 'plots')

            os.makedirs(output_data_dir, exist_ok=True)
            os.makedirs(output_plot_dir, exist_ok=True)

            # Save clustered data
            output_file = os.path.join(output_data_dir, f"{filename_without_ext}_cluster.csv")
            df_clustered.to_csv(output_file, index=False)
            print(f"Clustered data saved: {output_file}")

            # Create and save plot
            create_cluster_plot(df_clustered, output_plot_dir, filename_without_ext)

            # Print clustering summary
            fruit_summary = df_clustered[df_clustered['FruitID'] != 0].groupby('ClusterID').agg({
                'Temperature': ['count', 'mean', 'std'],
                'FruitID': 'nunique'
            }).round(2)
            print("Clustering Summary:")
            print(fruit_summary)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main():
    # Get root directory from user or use current directory
    root_directory = input("Enter the root directory path (or press Enter for current directory): ").strip()

    if not root_directory:
        root_directory = os.getcwd()

    if not os.path.exists(root_directory):
        print(f"Directory {root_directory} does not exist!")
        return

    print(f"Processing files in: {root_directory}")
    process_point_cloud_files(root_directory)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()