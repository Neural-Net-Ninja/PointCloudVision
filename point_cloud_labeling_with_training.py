import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse
from collections import Counter

def load_point_cloud(file_path):
    """Load point cloud data from CSV."""
    return pd.read_csv(file_path)

def train_model(labeled_df, features):
    """Train an MLP classifier on labeled data."""
    # Filter out unlabeled points (SemClassID != -1)
    labeled_df = labeled_df[labeled_df['SemClassID'] != -1]

    X = labeled_df[features]
    y = labeled_df['SemClassID']

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
        print(f"Training accuracy: {accuracy_score(y_test, y_pred):.2f}")

    return model, scaler

def predict_labels(model, scaler, unlabeled_df, features):
    """Predict initial labels on unlabeled data."""
    X = unlabeled_df[features]
    X_scaled = scaler.transform(X)
    initial_labels = model.predict(X_scaled)
    return initial_labels

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

def assign_labels(df, final_labels):
    """Assign final labels to the SemClassID column."""
    df['SemClassID'] = final_labels
    return df

def save_point_cloud(df, output_path):
    """Save the labeled point cloud to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Labeled point cloud saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and label point cloud data for fruit bunches.")
    parser.add_argument('--labeled_input', type=str, required=True, help='Path to labeled CSV file for training')
    parser.add_argument('--unlabeled_input', type=str, required=True, help='Path to unlabeled CSV file for prediction')
    parser.add_argument('--output', type=str, default='predicted_point_cloud.csv', help='Path to output CSV file')
    parser.add_argument('--threshold', type=float, default=0.01, help='Distance threshold for clustering')
    args = parser.parse_args()

    # Define features (excluding positions to avoid location bias; adjust as needed)
    features = ['R', 'G', 'B', 'Intensity', 'DistanceToDTM']  # Add 'X', 'Y', 'Z' if positions are relevant

    # Load labeled data and train model
    labeled_df = load_point_cloud(args.labeled_input)
    model, scaler = train_model(labeled_df, features)

    # Load unlabeled data
    unlabeled_df = load_point_cloud(args.unlabeled_input)

    # Predict initial labels
    initial_labels = predict_labels(model, scaler, unlabeled_df, features)

    # Cluster points into fruit bunches
    cluster_ids = cluster_points(unlabeled_df, threshold=args.threshold)

    # Enforce consistency with majority voting
    final_labels = enforce_consistency(initial_labels, cluster_ids)

    # Assign labels to SemClassID
    unlabeled_df = assign_labels(unlabeled_df, final_labels)

    # Save the predicted point cloud
    save_point_cloud(unlabeled_df, args.output)


if __name__ == "__main__":
    main()