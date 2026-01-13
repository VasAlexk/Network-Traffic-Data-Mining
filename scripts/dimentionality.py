import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Ορισμός φακέλου εξόδου για όλα τα αρχεία
output_directory = r"C:\Users\delll\PycharmProjects\datamining"

# Δημιουργία του φακέλου αν δεν υπάρχει
os.makedirs(output_directory, exist_ok=True)

print("Ξεκινά η επεξεργασία για τη μείωση διαστατικότητας...")

# 1. Φόρτωση δεδομένων
filename = "data.csv"
try:
    df = pd.read_csv(filename)
    print(f"Φορτώθηκαν {df.shape[0]} γραμμές από το '{filename}'.")
except FileNotFoundError:
    print(f"Το αρχείο '{filename}' δεν βρέθηκε.")
    sys.exit(1) # Έξοδος αν το αρχείο δεν βρεθεί

# Στήλες προς διαγραφή (ΧΩΡΙΣ 'Label' και 'Traffic Type')
columns_to_drop_initial = [
    'Bwd Segment Size Avg', 'Fwd Segment Size Avg', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
    'Average Packet Size', 'Fwd Packets/s', 'Idle Max', 'Idle Min',
    'Fwd IAT Max', 'Idle Mean', 'Fwd IAT Mean', 'Fwd IAT Min',
    'Bwd Packet/Bulk Avg', 'Active Min', 'Active Max',
    'ACK Flag Count', 'Fwd IAT Total', 'Fwd Act Data Pkts',
    'Subflow Bwd Packets', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags','URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Fwd Bytes/Bulk Avg',
    'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp',
    'Traffic Subtype','Fwd PSH Flags', 'Fwd Header Length', 'Fwd Seg Size Min'
    
]

# Αφαίρεση στηλών (αν υπάρχουν στο αρχείο)
existing_to_drop = [col for col in columns_to_drop_initial if col in df.columns]
reduced_data = df.drop(columns=existing_to_drop) # Δημιουργία reduced_data
print(f"Αρχικές στήλες: {df.shape[1]}")
print(f"Στήλες που αφαιρέθηκαν στην αρχική φάση: {len(existing_to_drop)}")
print(f"Στήλες που παραμένουν στο 'reduced_data': {reduced_data.shape[1]}")
print(f"Οι στήλες 'Label' και 'Traffic Type' παραμένουν.")

# Αποθήκευση του reduced_data (πριν το PCA)
reduced_data_output_path = os.path.join(output_directory, "dataset_reduced_initial.csv")
reduced_data.to_csv(reduced_data_output_path, index=False)

# 2. Διαχωρισμός χαρακτηριστικών (X) και στόχων (y) από το reduced_data
X = reduced_data.drop(columns=['Label', 'Traffic Type'], errors='ignore')
y_label = reduced_data['Label']
y_traffic_type = reduced_data['Traffic Type']

# Έλεγχος για μη-αριθμητικά δεδομένα στα X
numeric_cols_X = X.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols_X) != X.shape[1]:
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Μη-αριθμητικές στήλες που εξαιρούνται: {non_numeric_cols}")
    X = X[numeric_cols_X] # Κρατάμε μόνο τις αριθμητικές

# 3. Κανονικοποίηση χαρακτηριστικών
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
print("\nΤα χαρακτηριστικά κανονικοποιήθηκαν.")

# 4. Εφαρμογή PCA για explained_variance >= 95%
pca = PCA(n_components=0.95)
X_pca_transformed = pca.fit_transform(X_scaled)

print(f"\nΑρχικές διαστάσεις χαρακτηριστικών (πριν PCA): {X.shape[1]}")
print(f"Διαστάσεις χαρακτηριστικών μετά το PCA: {X_pca_transformed.shape[1]}")
print(f"Διακύμανση από PCA: {sum(pca.explained_variance_ratio_):.4f}")

# Δημιουργία DataFrame με τα PCA components και προσθήκη ετικετών για το επόμενο script
pca_columns = [f'PC{i+1}' for i in range(X_pca_transformed.shape[1])]
df_pca_final = pd.DataFrame(X_pca_transformed, columns=pca_columns, index=X.index)
df_pca_final['Label'] = y_label
df_pca_final['Traffic Type'] = y_traffic_type


# Αποθήκευση του PCA dataset (χαρακτηριστικά + ετικέτες)
pca_output_path = os.path.join(output_directory, "pca_transformed_data_with_labels.csv")
df_pca_final.to_csv(pca_output_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Αριθμός Principal Components')
plt.ylabel('Σωρευτική Επεξηγούμενη Διακύμανση')
plt.title('PCA Explained Variance')
plt.grid(True)
plot_path = os.path.join(output_directory, "pca_explained_variance_plot.png")
plt.savefig(plot_path)
