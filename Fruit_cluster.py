import os
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def is_point_cloud_csv(file_path):
    """Check if CSV file contains required point cloud columns."""
    try:
        df = pd.read_csv(file_path)
        required_columns = {'X', 'Y', 'Z', 'Intensity', 'Temperature', 'FruitID'}
        if not required_columns.issubset(df.columns):
            print(f"Missing required columns in {file_path}: {required_columns - set(df.columns)}")
            return False
        return True
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
    fruit_data = df[df['FruitID'] != 0].copy()

    if len(fruit_data) == 0:
        print("No fruit data found (all FruitID = 0)")
        df['ClusterID'] = -1
        return df, None

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    fruit_data['ClusterID'] = kmeans.fit_predict(fruit_data[['Temperature']])

    df['ClusterID'] = -1
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

        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"{filename}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot saved: {os.path.join(output_path, f'{filename}.png')}")

    except Exception as e:
        print(f"Error creating plot for {filename}: {e}")


def calculate_fruit_statistics(df, output_path, filename, all_stats, k=5):
    """Calculate per-FruitID statistics and kNN temperature difference using centroid."""
    try:
        # Filter fruit points (FruitID != 0)
        fruit_data = df[df['FruitID'] != 0].copy()
        if len(fruit_data) == 0:
            print(f"No fruit data for statistics in {filename}")
            return all_stats

        # Calculate statistics per FruitID, including centroid coordinates
        stats = fruit_data.groupby('FruitID').agg({
            'Temperature': [
                'max', 'min', 'mean', 'median', ('count', 'count'),
                ('Min80T', lambda x: np.percentile(x, 10)),  # 10th percentile
                ('Max80T', lambda x: np.percentile(x, 90))   # 90th percentile
            ],
            'X': 'mean',  # For centroid
            'Y': 'mean',  # For centroid
            'Z': 'mean'   # For centroid
        }).round(2)
        stats.columns = ['MaxT', 'MinT', 'AverageT', 'MedianT', 'NumberPoints', 'Min80T', 'Max80T', 'X_centroid', 'Y_centroid', 'Z_centroid']
        # Convert NumberPoints to integer
        stats['NumberPoints'] = stats['NumberPoints'].astype(int)
        stats = stats.reset_index()

        # Prepare non-fruit data for kNN
        non_fruit_data = df[df['FruitID'] == 0][['X', 'Y', 'Z']].values
        non_fruit_temps = df[df['FruitID'] == 0]['Temperature'].values

        if len(non_fruit_data) >= k:
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(non_fruit_data)

            knn_temp_diff = []
            for fruit_id in stats['FruitID']:
                centroid = stats[stats['FruitID'] == fruit_id][['X_centroid', 'Y_centroid', 'Z_centroid']].values
                avg_temp = stats[stats['FruitID'] == fruit_id]['AverageT'].iloc[0]

                if len(centroid) > 0:
                    distances, indices = knn.kneighbors(centroid)
                    neighbor_temps = non_fruit_temps[indices[0]]
                    temp_diff = np.abs(neighbor_temps.mean() - avg_temp).round(2)
                    knn_temp_diff.append(temp_diff)
                else:
                    knn_temp_diff.append(np.nan)

            stats['knn-TempDiff'] = knn_temp_diff
        else:
            print(f"Not enough non-fruit points for kNN (k={k}) in {filename}")
            stats['knn-TempDiff'] = np.nan

        # Add filename column for aggregation
        stats['Filename'] = filename

        # Drop centroid columns before saving
        stats = stats.drop(columns=['X_centroid', 'Y_centroid', 'Z_centroid'])

        # Append to all_stats list
        all_stats.append(stats)

        # Save individual statistics to CSV
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{filename}_fruit_stats.csv")
        stats.to_csv(output_file, index=False)
        print(f"Fruit statistics saved: {output_file}")

        print("Fruit Statistics:")
        print(stats)

        return all_stats

    except Exception as e:
        print(f"Error calculating fruit statistics for {filename}: {e}")
        return all_stats


def cluster_fruit_stats(stats_df, attribute='AverageT', n_clusters=4):
    """Perform KMeans clustering on fruit stats using specified attribute."""
    if attribute not in stats_df.columns:
        print(f"Attribute {attribute} not found in stats DataFrame.")
        return None, None

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    stats_df['StatsClusterID'] = kmeans.fit_predict(stats_df[[attribute]])

    return stats_df, kmeans


def create_stats_cluster_plot(df, attribute, output_path, filename=None):
    """Create and save clustering visualization for fruit stats."""
    try:
        plt.figure(figsize=(12, 8))

        if 'StatsClusterID' not in df.columns:
            print(f"No cluster data to plot for {filename if filename else 'aggregate stats'}")
            return

        df_sorted = df.sort_values(by=attribute)
        plt.scatter(range(len(df_sorted)), df_sorted[attribute], c=df_sorted['StatsClusterID'], cmap='Set1', alpha=0.6)
        plt.xlabel(f'Fruits (sorted by {attribute})')
        plt.ylabel(attribute)
        plt.title(f'KMeans Clusters based on {attribute}{" for " + filename if filename else ""}')
        plt.colorbar(label='Cluster ID')
        plt.grid(True, alpha=0.3)

        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"{filename}_{attribute}_clusters.png" if filename else f"{attribute}_clusters.png"
        plot_path = os.path.join(output_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Stats cluster plot saved: {plot_path}")

    except Exception as e:
        print(f"Error creating stats cluster plot for {filename if filename else 'aggregate stats'}: {e}")


def process_point_cloud_files(root_dir, attribute='AverageT'):
    """Process all point cloud files in the directory structure."""
    csv_files = find_csv_files(root_dir)
    all_fruit_stats = []  # List to store all fruit statistics DataFrames
    summary_stats = {
        'total_fruits': 0,
        'avg_fruit_temp': 0.0,
        'avg_non_fruit_temp': 0.0,
        'num_files_processed': 0,
        'total_points_processed': 0,
        'avg_knn_temp_diff': 0.0
    }

    if not csv_files:
        print("No valid point cloud CSV files found!")
        return

    print(f"Found {len(csv_files)} valid point cloud files")

    total_fruit_points = 0
    total_non_fruit_points = 0
    total_fruit_temp_sum = 0.0
    total_non_fruit_temp_sum = 0.0
    total_knn_temp_diff = 0.0
    total_fruits = 0

    for file_path in tqdm(csv_files, desc="Processing files", unit="file"):
        try:
            print(f"\nProcessing: {file_path}")

            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} points")

            df_clustered, kmeans_model = cluster_fruit_data(df)

            rel_path = os.path.relpath(file_path, root_dir)
            path_parts = Path(rel_path).parts
            filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]

            if len(path_parts) > 1:
                parent_dir = os.path.dirname(root_dir)
                original_folder_name = os.path.basename(root_dir)
                new_root_dir = os.path.join(parent_dir, f"{original_folder_name}_cluster")

                rel_dir = os.path.dirname(rel_path)
                output_data_dir = os.path.join(new_root_dir, rel_dir, 'per_point_clustering')
                output_plot_dir = os.path.join(new_root_dir, rel_dir, 'plots')
                output_stats_dir = os.path.join(new_root_dir, rel_dir, 'fruit_stats')
                output_stats_plot_dir = os.path.join(new_root_dir, rel_dir, f'{attribute}_cluster_plots')
            else:
                parent_dir = os.path.dirname(root_dir)
                original_folder_name = os.path.basename(root_dir)
                new_root_dir = os.path.join(parent_dir, f"{original_folder_name}_cluster")

                output_data_dir = os.path.join(new_root_dir, 'per_point_clustering')
                output_plot_dir = os.path.join(new_root_dir, 'plots')
                output_stats_dir = os.path.join(new_root_dir, 'fruit_stats')
                output_stats_plot_dir = os.path.join(new_root_dir, f'{attribute}_cluster_plots')

            os.makedirs(output_data_dir, exist_ok=True)
            os.makedirs(output_plot_dir, exist_ok=True)
            os.makedirs(output_stats_dir, exist_ok=True)
            os.makedirs(output_stats_plot_dir, exist_ok=True)

            output_file = os.path.join(output_data_dir, f"{filename_without_ext}_cluster.csv")
            df_clustered.to_csv(output_file, index=False)
            print(f"Clustered data saved: {output_file}")

            create_cluster_plot(df_clustered, output_plot_dir, filename_without_ext)

            all_fruit_stats = calculate_fruit_statistics(df_clustered, output_stats_dir, filename_without_ext, all_fruit_stats)

            # Cluster and plot per-file fruit stats
            stats_df = all_fruit_stats[-1]
            if not stats_df.empty:
                clustered_stats, kmeans = cluster_fruit_stats(stats_df, attribute, n_clusters=4)
                if clustered_stats is not None:
                    create_stats_cluster_plot(clustered_stats, attribute, output_stats_plot_dir, filename_without_ext)

            # Update summary statistics
            fruit_points = df_clustered[df_clustered['FruitID'] != 0]
            non_fruit_points = df_clustered[df_clustered['FruitID'] == 0]
            summary_stats['num_files_processed'] += 1
            summary_stats['total_points_processed'] += len(df_clustered)
            if len(fruit_points) > 0:
                total_fruit_points += len(fruit_points)
                total_fruit_temp_sum += fruit_points['Temperature'].sum()
                total_fruits += len(fruit_points['FruitID'].unique())
            if len(non_fruit_points) > 0:
                total_non_fruit_points += len(non_fruit_points)
                total_non_fruit_temp_sum += non_fruit_points['Temperature'].sum()
            if len(all_fruit_stats) > 0 and 'knn-TempDiff' in all_fruit_stats[-1]:
                valid_knn = all_fruit_stats[-1]['knn-TempDiff'].dropna()
                if len(valid_knn) > 0:
                    total_knn_temp_diff += valid_knn.sum()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Finalize summary statistics
    summary_stats['total_fruits'] = total_fruits
    summary_stats['avg_fruit_temp'] = round(total_fruit_temp_sum / total_fruit_points, 2) if total_fruit_points > 0 else 0.0
    summary_stats['avg_non_fruit_temp'] = round(total_non_fruit_temp_sum / total_non_fruit_points, 2) if total_non_fruit_points > 0 else 0.0
    summary_stats['avg_knn_temp_diff'] = round(total_knn_temp_diff / total_fruits, 2) if total_fruits > 0 else 0.0

    # Save summary statistics to JSON
    if summary_stats['num_files_processed'] > 0:
        stats_json_path = os.path.join(new_root_dir, 'all_fruit_stats.json')
        with open(stats_json_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Summary statistics saved: {stats_json_path}")

    # Save individual fruit statistics to CSV
    if all_fruit_stats:
        all_stats_df = pd.concat(all_fruit_stats, ignore_index=True)
        stats_csv_path = os.path.join(new_root_dir, 'all_fruit_stats.csv')
        all_stats_df.to_csv(stats_csv_path, index=False)
        print(f"All fruit statistics saved: {stats_csv_path}")

        # Perform KMeans on aggregated fruit stats and plot
        clustered_stats, kmeans = cluster_fruit_stats(all_stats_df, attribute, n_clusters=4)
        if clustered_stats is not None:
            output_stats_plot_dir = os.path.join(new_root_dir, f"{attribute}_cluster_plots")
            create_stats_cluster_plot(clustered_stats, attribute, output_stats_plot_dir)


def main():
    root_directory = input("Enter the root directory path (or press Enter for current directory): ").strip()

    if not root_directory:
        root_directory = os.getcwd()

    if not os.path.exists(root_directory):
        print(f"Directory {root_directory} does not exist!")
        return

    attribute = input("Enter the attribute for fruit stats clustering (default: AverageT): ").strip() or 'AverageT'

    print(f"Processing files in: {root_directory}")
    process_point_cloud_files(root_directory, attribute)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()