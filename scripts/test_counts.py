import pandas as pd

df = pd.read_csv("final_hdbscan_balanced.csv")

# 1. Κατανομή Label
print("Κατανομή Label:")
print(df["Label"].value_counts(), end="\n\n")

# 2. Κατανομή Traffic Type
print("Κατανομή Traffic Type (top 10):")
print(df["Traffic Type"].value_counts().head(10), end="\n\n")

# 3. Μέγεθος dataset
print(f"Σύνολο γραμμών: {len(df):,}")
