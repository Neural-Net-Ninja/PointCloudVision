import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from collections import Counter
import os
import glob

# python point_cloud_labeling_with_training.py --labeled_input Q:\Fruit\Training\normal_label --unlabeled_input Q:\Fruit\Training\0_labled --output Q:\Fruit\Training\output


def load_point_cloud(input_path):
    """Load point cloud data from CSV/TXT file or all CSV/TXT files in a directory."""
    if os.path.isdir(input_path):
        # Load all CSV and TXT files in the directory and concatenate
        file_patterns = [
            os.path.join(input_path, "*.csv"),
            os.path.join(input_path, "*.txt")
        ]
        file_paths = []
        for pattern in file_patterns:
            file_paths.extend(glob.glob(pattern))
        if not file_paths:
            raise ValueError(f"No CSV or TXT files found in directory: {input_path}")
        dfs = [pd.read_csv(fp) for fp in file_paths]  # pd.read_csv works for delimited TXT too
        return pd.concat(dfs, ignore_index=True)
    else:
        # Load single file (CSV or TXT)
        return pd.read_csv(input_path)

def load_unlabeled_files(input_dir):
    """Load all unlabeled CSV/TXT files from a directory, returning list of (filename, df) tuples."""
    if not os.path.isdir(input_dir):
        # If single file, treat as one
        filename = os.path.basename(input_dir)
        df = pd.read_csv(input_dir)
        return [(filename, df)]
    
    file_patterns = [
        os.path.join(input_dir, "*.csv"),
        os.path.join(input_dir, "*.txt")
    ]
    file_paths = []
    for pattern in file_patterns:
        file_paths.extend(glob.glob(pattern))
    if not file_paths:
        raise ValueError(f"No CSV or TXT files found in directory: {input_dir}")
    
    results = []
    for fp in file_paths:
        filename = os.path.basename(fp)
        df = pd.read_csv(fp)
        results.append((filename, df))
    return results

def train_model(labeled_df, features, label_col, sem_col=None):
    """Train an MLP classifier on labeled data for a specific label column.
    If sem_col is provided (for hierarchical SpecificClassID), include it as a feature."""
    # Filter out unlabeled points: For SpecificClassID, filter on Specific != -1 AND Sem != -1
    if sem_col:
        labeled_df = labeled_df[(labeled_df[label_col] != -1) & (labeled_df[sem_col] != -1)]
    else:
        # For SemClassID (if used): Filter on Sem != -1
        labeled_df = labeled_df[labeled_df[label_col] != -1]

    if len(labeled_df) == 0:
        raise ValueError(f"No labeled data found for {label_col}")

    X = labeled_df[features].copy()
    y = labeled_df[label_col]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test for validation (optional)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train MLP
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Validate (optional)
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        print(f"Training accuracy for {label_col}: {accuracy_score(y_test, y_pred):.2f}")

    return model, scaler

def predict_labels(model, scaler, df, features):
    """Predict labels using the model."""
    X = df[features].copy()
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

def cluster_points(df, threshold=0.01):
    """Cluster points into fruit bunches using hierarchical clustering."""
    coords = df[['X', 'Y', 'Z']].values
    Z = linkage(coords, method='ward')
    cluster_ids = fcluster(Z, t=threshold, criterion='distance')
    return cluster_ids

def enforce_consistency(initial_labels, cluster_ids):
    """Enforce consistent labels within each cluster using majority voting."""
    unique_clusters = np.unique(cluster_ids)
    final_labels = np.zeros_like(initial_labels)

    for cluster in unique_clusters:
        mask = (cluster_ids == cluster)
        cluster_labels = initial_labels[mask]
        if len(cluster_labels) > 0:
            majority_label = Counter(cluster_labels).most_common(1)[0][0]
            final_labels[mask] = majority_label

    return final_labels

def assign_labels(df, spec_labels, fruit_mask):
    """Assign final SpecificClassID labels only to fruit points (SemClassID != -1)."""
    df.loc[fruit_mask, 'SpecificClassID'] = spec_labels
    return df

def save_point_cloud(df, output_path):
    """Save the labeled point cloud to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Labeled point cloud saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and predict SpecificClassID for fruit points in point cloud data (hierarchical).")
    parser.add_argument('--labeled_input', type=str, required=True, help='Path to labeled CSV/TXT file or directory for training')
    parser.add_argument('--unlabeled_input', type=str, required=True, help='Path to unlabeled CSV/TXT file or directory for prediction')
    parser.add_argument('--output', type=str, default='predicted_point_cloud.csv', help='Path to output CSV file or directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Distance threshold for clustering')
    args = parser.parse_args()

    # Ensure output directory exists if it's a directory
    if os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    elif not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Define features for SpecificClassID (includes SemClassID as feature)
    features = ['Intensity', 'DistanceToDTM', 'SemClassID']

    # Load labeled data (file or dir) and train Specific model only (on fruits)
    labeled_df = load_point_cloud(args.labeled_input)
    model, scaler = train_model(labeled_df, features, 'SpecificClassID', sem_col='SemClassID')

    # Load and process unlabeled data (file or dir)
    unlabeled_files = load_unlabeled_files(args.unlabeled_input)
    
    for filename, df in unlabeled_files:
        print(f"Processing {filename}...")
        
        # Identify fruit points (SemClassID != -1; these are already labeled as fruits)
        fruit_mask = df['SemClassID'] != -1
        num_fruits = fruit_mask.sum()
        if num_fruits == 0:
            print(f"No fruit points (SemClassID != -1) found in {filename}; skipping prediction.")
            # Still save the unchanged df
            if os.path.isdir(args.output):
                output_filename = os.path.join(args.output, f"labeled_{filename}")
            else:
                output_filename = args.output
            save_point_cloud(df, output_filename)
            continue
        
        print(f"Found {num_fruits} fruit points for SpecificClassID prediction.")
        
        # Extract fruit points for prediction
        df_fruits = df[fruit_mask].copy()
        
        # Predict initial SpecificClassID (conditioned on SemClassID)
        initial_labels = predict_labels(model, scaler, df_fruits, features)

        # Cluster all points but enforce only on fruits (using fruit indices)
        all_cluster_ids = cluster_points(df)
        fruit_cluster_ids = all_cluster_ids[fruit_mask]
        
        # Enforce consistency on fruit clusters
        final_labels = enforce_consistency(initial_labels, fruit_cluster_ids)

        # Assign back to full df (only fruits get updated SpecificClassID)
        df = assign_labels(df, final_labels, fruit_mask)

        # Save the full point cloud (non-fruits unchanged)
        if os.path.isdir(args.output):
            output_filename = os.path.join(args.output, f"labeled_{filename}")
        else:
            output_filename = args.output
        save_point_cloud(df, output_filename)


if __name__ == "__main__":
    main()