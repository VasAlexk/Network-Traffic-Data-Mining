import pandas as pd

df = pd.read_csv("minibatch_kmeans_balanced_with_smote.csv")

# 1. Κατανομή Label
print("Κατανομή Label:")
print(df["Label"].value_counts(), end="\n\n")

# 2. Κατανομή Traffic Type
print("Κατανομή Traffic Type (top 10):")
print(df["Traffic Type"].value_counts().head(10), end="\n\n")

# 3. Μέγεθος dataset
print(f"Σύνολο γραμμών: {len(df):,}")
