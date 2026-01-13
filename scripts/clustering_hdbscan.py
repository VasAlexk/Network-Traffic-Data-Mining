import os
import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.utils import shuffle

# === ΡΥΘΜΙΣΕΙΣ ===
input_file = "pca_transformed_data_with_labels.csv"
output_dir = "hdbscan_chunks_output"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 1_000_000
num_chunks = 9
min_cluster_size = 50
points_per_cluster = 5
rare_threshold = 1500
random_state = 42

np.random.seed(random_state)

# === Διάβασμα στηλών ===
with open(input_file, "r", encoding="utf-8") as f:
    header = f.readline().strip().split(",")

pca_cols = [col for col in header if col.startswith("PC")]

# === Ανάγνωση όλων των δεδομένων για rare types και fallback ===
print("Ανάγνωση όλων των δεδομένων...")
df_all = pd.read_csv(input_file)

# === Εντοπισμός και αποθήκευση σπάνιων Traffic Types ===
rare_traffic_types = df_all["Traffic Type"].value_counts()
rare_traffic_types = rare_traffic_types[rare_traffic_types < rare_threshold].index.tolist()
rare_rows = df_all[df_all["Traffic Type"].isin(rare_traffic_types)].copy()

print(f"\nΣπάνια Traffic Types (< {rare_threshold}): {len(rare_rows)} γραμμές από {len(rare_traffic_types)} τύπους")

# === Συνάρτηση: Επιλογή top-N σημείων ανά cluster ===
def find_top_n_points_per_cluster(df, pca_cols, n=5):
    selected = []
    for _, group in df.groupby("Cluster"):
        X = group[pca_cols].to_numpy()
        centroid = np.mean(X, axis=0)
        group = group.copy()
        group["dist"] = np.linalg.norm(X - centroid, axis=1)
        top_n = group.nsmallest(n, "dist")
        selected.append(top_n.drop(columns=["dist"]))
    return pd.concat(selected)

# === Βήμα 1: HDBSCAN σε chunks ===
chunk_paths = []

for i in range(num_chunks):
    skip = i * chunk_size + 1
    print(f"\n🔹 [Chunk {i+1}] Γραμμές {skip:,} – {skip + chunk_size - 1:,}")

    try:
        chunk = pd.read_csv(input_file, skiprows=skip, nrows=chunk_size, header=None, names=header)
        X_chunk = chunk[pca_cols]

        hdb = HDBSCAN(min_cluster_size=min_cluster_size)
        chunk["Cluster"] = hdb.fit_predict(X_chunk)

        clustered = chunk[chunk["Cluster"] != -1].copy()
        clustered["Cluster"] += (i + 1) * 1000

        selected_df = find_top_n_points_per_cluster(clustered, pca_cols, n=points_per_cluster)

        chunk_path = os.path.join(output_dir, f"chunk_{i+1}_selected.csv")
        selected_df.to_csv(chunk_path, index=False)
        chunk_paths.append(chunk_path)

        print(f"Επιλέχθηκαν {selected_df.shape[0]:,} σημεία → {chunk_path}")

    except Exception as e:
        print(f"Σφάλμα στο chunk {i+1}: {e}")

# === Βήμα 2: Συγχώνευση όλων των selected chunks ===
print("\nΣυγχώνευση chunks...")
dfs = [pd.read_csv(f) for f in chunk_paths]
merged_df = pd.concat(dfs, ignore_index=True)
print(f"Επιλεγμένα σημεία από clustering: {len(merged_df):,}")

# === Βήμα 3: Oversampling Label Minority (όχι SMOTE)
label_counts = merged_df["Label"].value_counts()
print(f"\nΚατανομές Label πριν oversampling: {label_counts.to_dict()}")

if "Benign" in label_counts and "Malicious" in label_counts:
    majority = label_counts.idxmax()
    minority = label_counts.idxmin()

    df_major = merged_df[merged_df["Label"] == majority]
    df_minor = merged_df[merged_df["Label"] == minority]

    if len(df_minor) < 2000:
        df_minor_oversampled = df_minor.sample(n=2000, replace=True, random_state=random_state)
        merged_df = pd.concat([df_major, df_minor_oversampled], ignore_index=True)
        print(f"Oversampling {minority} → 2000 δείγματα")
    else:
        print("Labels ήδη ισορροπημένα.")
else:
    print("Δεν υπάρχουν και οι δύο Label – παράκαμψη oversampling.")

# === Βήμα 4: Προσθήκη rare traffic types ===
print(f"\nΠροσθήκη {len(rare_rows):,} γραμμών από σπάνια Traffic Types")
merged_df = pd.concat([merged_df, rare_rows], ignore_index=True)

# === Βήμα 4b: Duplication για Traffic Types που έχουν <500 στο merged_df ===
print("\nDuplication για Traffic Types με <500 δείγματα στο merged_df...")

# Υπολογίζουμε πλήθος εμφανίσεων ανά traffic type στο merged_df
merged_counts = merged_df["Traffic Type"].value_counts()
under_500 = merged_counts[merged_counts < 500]

for traffic_type, current_len in under_500.items():
    # Πόσα έχει στο αρχικό df_all
    total_in_all = df_all[df_all["Traffic Type"] == traffic_type].shape[0]

    # Αν το αρχικό έχει <= 500 → στόχος είναι 500
    if total_in_all <= 500:
        target_n = 500
    else:
        # Αν έχει >500 → αύξησέ το μεχρι 1000
         target_n = min(1000, total_in_all)

    needed = target_n - current_len
    if needed <= 0:
        continue

    df_subset = merged_df[merged_df["Traffic Type"] == traffic_type]
    
    if df_subset.empty:
        print(f"Το '{traffic_type}' δεν υπάρχει καθόλου στο merged_df – αγνοείται.")
        continue

    duplicated = df_subset.sample(n=needed, replace=True, random_state=random_state)
    merged_df = pd.concat([merged_df, duplicated], ignore_index=True)
    print(f"{traffic_type}': {current_len} → {current_len + needed} (στόχος: {target_n})")


# === Βήμα 4c: Μείωση Malicious + DoS κατά 20% ===
print("\nΜείωση των γραμμών με Label='Malicious' και Traffic Type='DoS' κατά 20%...")
mask_malicious_dos = (merged_df["Label"] == "Malicious") & (merged_df["Traffic Type"] == "DoS")
df_malicious_dos = merged_df[mask_malicious_dos]
keep_n = int(len(df_malicious_dos) * 0.8)

df_malicious_dos_reduced = df_malicious_dos.sample(n=keep_n, random_state=random_state)
merged_df = pd.concat([merged_df[~mask_malicious_dos], df_malicious_dos_reduced], ignore_index=True)

print(f"Malicious-DoS γραμμές μειώθηκαν σε {keep_n}")

# === Βήμα 5: Shuffle & Save ===
merged_df = shuffle(merged_df, random_state=random_state)
merged_df = merged_df.drop(columns=["Cluster"], errors="ignore")
final_path = os.path.join(output_dir, "final_hdbscan_balanced.csv")
merged_df.to_csv(final_path, index=False)

print(f"\nΟλοκληρώθηκε! Τελικό αρχείο: {final_path} με {len(merged_df):,} γραμμές")
