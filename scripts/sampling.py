import pandas as pd
from sklearn.model_selection import train_test_split

# === 1. Φόρτωση αρχικού αρχείου ===
df = pd.read_csv("pca_transformed_data_with_labels.csv")
df["Label_Type"] = df["Label"] + " - " + df["Traffic Type"]

# === 2. Εύρεση σπάνιων κατηγοριών (< 1000 εμφανίσεις) ===
type_counts = df["Label_Type"].value_counts()
rare_classes = type_counts[type_counts < 1000].index.tolist()
rare_df = df[df["Label_Type"].isin(rare_classes)]
main_df = df[~df["Label_Type"].isin(rare_classes)]

print(f"Κρατάμε όλες τις {len(rare_classes)} σπάνιες κατηγορίες: {len(rare_df)} γραμμές")

# === 3. Στρωματοποιημένη δειγματοληψία 10% στο υπόλοιπο ===
sample_frac = 0.10
sampled_df, _ = train_test_split(
    main_df,
    test_size=(1 - sample_frac),
    stratify=main_df["Label_Type"],
    random_state=42,
)

# === 4. Συνένωση με σπάνιες ===
combined_df = pd.concat([sampled_df, rare_df], ignore_index=True)
print(f"Συνολικό μέγεθος μετά τη δειγματοληψία και ένωση: {len(combined_df):,}")

# === 5. Προετοιμασία Τελικού DataFrame ===
df_final = combined_df.copy()

# === 6. Υποδειγματοληψία στο Malicious ===
malicious = df_final[df_final["Label"] == "Malicious"]
benign = df_final[df_final["Label"] == "Benign"]

desired_malicious = min(len(malicious), len(benign) * 7) #αναλογία
malicious_sampled = malicious.sample(n=desired_malicious, random_state=42)

df_final_balanced = pd.concat([benign, malicious_sampled], ignore_index=True)

# === 7. Αποθήκευση ===
df_final_balanced.to_csv("stratified_sample_combined.csv", index=False)
print(f"\nΟλοκληρώθηκε. Τελικές γραμμές: {len(df_final_balanced):,}")
print(df_final_balanced["Label"].value_counts())
