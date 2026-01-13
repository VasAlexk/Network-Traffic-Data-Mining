import os
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import cdist
from tqdm import tqdm

# === ΡΥΘΜΙΣΕΙΣ ===
input_file = 'pca_transformed_data_with_labels.csv'
output_file = 'minibatch_kmeans_balanced_with_smote.csv'
max_clusters_per_type = 300
benign_clusters = 1500
min_points_threshold = 2000  # μόνο traffic types με ≥2000 πάνε σε clustering αλλιώς SMOTE
num_points_per_cluster = 8  
random_state = 42
batch_size = 10000

# === 1. Διαβάζουμε δεδομένα ===
df = pd.read_csv(input_file)
pca_cols = [col for col in df.columns if col.startswith("PC")]
selected_rows = []

# === 2. Διαχωρίζουμε Benign / Malicious ===
benign_df = df[df['Label'] == 'Benign'].copy()
malicious_df = df[df['Label'] == 'Malicious'].copy()

# === 3. Clustering σε Benign ===
print(f"\nBenign → {benign_clusters} clusters")
X = benign_df[pca_cols].values
kmeans = MiniBatchKMeans(n_clusters=min(benign_clusters, len(X)),
                         batch_size=batch_size, random_state=random_state)
labels = kmeans.fit_predict(X)
benign_df['Cluster'] = labels
centroids = kmeans.cluster_centers_

for i in tqdm(range(kmeans.n_clusters), desc="Benign - closest points"):
    group = benign_df[benign_df['Cluster'] == i]
    if group.empty: continue
    dists = cdist([centroids[i]], group[pca_cols])
    closest_idxs = np.argsort(dists[0])[:num_points_per_cluster]
    for idx in closest_idxs:
        selected_rows.append(group.iloc[idx])

# === 4. Clustering ή SMOTE ανά Traffic Type ===
print(f"\nMalicious ανά Traffic Type")

for traffic_type, group in malicious_df.groupby('Traffic Type'):
    print(f"Επεξεργασία: {traffic_type} ({len(group)} σημεία)")

    if len(group) < min_points_threshold:
        print(f"{traffic_type} → κάτω από threshold ({min_points_threshold}) → SMOTE")

        try:
            le = LabelEncoder()
            group["Traffic_Type_Label"] = le.fit_transform([traffic_type] * len(group))
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(group[pca_cols], group["Traffic_Type_Label"])
            group_resampled = pd.DataFrame(X_resampled, columns=pca_cols)
            group_resampled['Label'] = 'Malicious'
            group_resampled['Traffic Type'] = traffic_type

            # Κρατάμε μέχρι 100 από το SMOTE output
            selected_rows.extend(group_resampled.sample(n=min(len(group_resampled), 100), random_state=42).to_dict(orient='records'))

        except Exception as e:
            print(f"SMOTE απέτυχε για {traffic_type}: {e}")
            print(f"Κρατάμε {len(group)} αρχικά σημεία ως fallback")
            selected_rows.extend(group.to_dict(orient='records'))

        continue

    # Αν έχει αρκετά σημεία → Clustering
    print(f"Clustering σε {max_clusters_per_type} clusters")
    X = group[pca_cols].values
    kmeans = MiniBatchKMeans(n_clusters=min(max_clusters_per_type, len(X)),
                             batch_size=batch_size, random_state=random_state)
    labels = kmeans.fit_predict(X)
    group['Cluster'] = labels
    centroids = kmeans.cluster_centers_

    for i in range(kmeans.n_clusters):
        points = group[group['Cluster'] == i]
        if points.empty: continue
        dists = cdist([centroids[i]], points[pca_cols])
        closest_idxs = np.argsort(dists[0])[:num_points_per_cluster]
        for idx in closest_idxs:
            selected_rows.append(points.iloc[idx])

# === 5. Συγχώνευση και Αποθήκευση ===
print("\nΣυγχώνευση και αποθήκευση...")
final_df = pd.DataFrame(selected_rows).drop(columns=["Cluster", "Traffic_Type_Label"], errors="ignore")
final_df.to_csv(output_file, index=False)
print(f"Αποθηκεύτηκε τελικό balanced αρχείο: {output_file} ({len(final_df):,} γραμμές)")
