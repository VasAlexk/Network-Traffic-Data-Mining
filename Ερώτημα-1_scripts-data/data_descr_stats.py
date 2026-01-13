# === ΑΠΑΡΑΙΤΗΤΕΣ ΒΙΒΛΙΟΘΗΚΕΣ ===
import matplotlib
matplotlib.use('TkAgg')  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime
import plotly.express as px

# === ΑΥΞΗΣΗ ΤΟΥ ΑΡΙΘΜΟΥ ΣΤΗΛΩΝ ΚΑΙ ΓΡΑΜΜΩΝ ===
pd.set_option('display.max_columns', None)  # Εμφανίζει όλες τις στήλες
pd.set_option('display.max_rows', None)     # Εμφανίζει όλες τις γραμμές
pd.set_option('display.width', None)        # Αυξάνει το πλάτος για να μην κοπούν τα δεδομένα
pd.set_option('display.max_colwidth', None) # Δεν περιορίζει το πλάτος των στηλών

# === ΚΑΤΑΓΡΑΦΗ ΟΛΩΝ ΤΩΝ PRINT ΣΕ ΑΡΧΕΙΟ TXT ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_path = f"data_analysis_{timestamp}.txt"
sys.stdout = open(report_path, "w", encoding="utf-8")

# === ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ ===
df = pd.read_csv('data.csv')  

# === ΓΕΝΙΚΗ ΠΛΗΡΟΦΟΡΙΑ ===
print("ΣΧΗΜΑ ΠΙΝΑΚΑ:", df.shape)
print("\nΤΥΠΟΙ ΣΤΗΛΩΝ:")
print(df.dtypes)

# === Αναλυτική κατηγοριοποίηση τύπων ===
type_counts = {
    'Decimal': 0,
    'String': 0,
    'DateTime': 0,
    'Other': 0
}

for col in df.columns:
    dtype = df[col].dtype

    if np.issubdtype(dtype, np.number):
        type_counts['Decimal'] += 1
    elif np.issubdtype(dtype, np.datetime64):
        type_counts['DateTime'] += 1
    elif dtype == object or dtype == "string":
        type_counts['String'] += 1
    else:
        type_counts['Other'] += 1

print("\nΚΑΤΗΓΟΡΙΟΠΟΙΗΜΕΝΟΙ ΤΥΠΟΙ ΔΕΔΟΜΕΝΩΝ (ΣΥΝΟΨΗ):")
for k, v in type_counts.items():
    print(f"{k}: {v}")

# === ΚΕΝΑ ===
print("\nΠΛΗΘΟΣ ΚΕΝΩΝ ΤΙΜΩΝ:")
print(df.isnull().sum())

# === ΠΕΡΙΓΡΑΦΙΚΑ ΣΤΑΤΙΣΤΙΚΑ ===
stats_numeric = df.describe().transpose()
stats_all = df.describe(include='all').transpose()

print("\nΒΑΣΙΚΑ ΣΤΑΤΙΣΤΙΚΑ ΑΡΙΘΜΗΤΙΚΩΝ ΣΤΗΛΩΝ:\n", stats_numeric)
print("\nΣΤΑΤΙΣΤΙΚΑ ΟΛΩΝ ΤΩΝ ΣΤΗΛΩΝ (ΚΑΙ ΚΑΤΗΓΟΡΙΚΩΝ):\n", stats_all)
print("\nΟΙ ΣΤΗΛΕΣ:\n")
print(df.columns)

# === ΠΕΡΙΓΡΑΦΗ ΒΑΣΙΚΩΝ ΜΕΤΑΒΛΗΤΩΝ ===
description_text = """
ΠΕΡΙΓΡΑΦΗ ΣΤΗΛΩΝ

Flow ID - Μοναδικό αναγνωριστικό για κάθε ροή δεδομένων.
Src IP - Διεύθυνση IP πηγής.
Src Port - Θύρα πηγής.
Dst IP - Διεύθυνση IP προορισμού.
Dst Port - Θύρα προορισμού.
Protocol - Πρωτόκολλο που χρησιμοποιείται στη ροή (π.χ., TCP, UDP).
Timestamp - Χρονική σήμανση της ροής.
Flow Duration - Διάρκεια της ροής σε μικροδευτερόλεπτα.
Total Fwd Packet - Συνολικός αριθμός πακέτων προς τα εμπρός.
Total Bwd packets - Συνολικός αριθμός πακέτων προς τα πίσω.
Total Length of Fwd Packet - Συνολικό μήκος πακέτων προς τα εμπρός.
Total Length of Bwd Packet - Συνολικό μήκος πακέτων προς τα πίσω.
Fwd Packet Length Max - Μέγιστο μήκος πακέτου προς τα εμπρός.
Fwd Packet Length Min - Ελάχιστο μήκος πακέτου προς τα εμπρός.
Fwd Packet Length Mean - Μέσο μήκος πακέτου προς τα εμπρός.
Fwd Packet Length Std - Τυπική απόκλιση μήκους πακέτων προς τα εμπρός.
Bwd Packet Length Max - Μέγιστο μήκος πακέτου προς τα πίσω.
Bwd Packet Length Min - Ελάχιστο μήκος πακέτου προς τα πίσω.
Bwd Packet Length Mean - Μέσο μήκος πακέτου προς τα πίσω.
Bwd Packet Length Std - Τυπική απόκλιση μήκους πακέτων προς τα πίσω.
Flow Bytes/s - Ρυθμός ροής σε bytes ανά δευτερόλεπτο.
Flow Packets/s - Ρυθμός ροής σε πακέτα ανά δευτερόλεπτο.
Flow IAT Mean - Μέσος χρόνος μεταξύ πακέτων στη ροή.
Flow IAT Std - Τυπική απόκλιση του χρόνου μεταξύ πακέτων στη ροή.
Flow IAT Max - Μέγιστος χρόνος μεταξύ πακέτων στη ροή.
Flow IAT Min - Ελάχιστος χρόνος μεταξύ πακέτων στη ροή.
Fwd IAT Total - Συνολικός χρόνος μεταξύ πακέτων προς τα εμπρός.
Fwd IAT Mean - Μέσος χρόνος μεταξύ πακέτων προς τα εμπρός.
Fwd IAT Std - Τυπική απόκλιση του χρόνου μεταξύ πακέτων προς τα εμπρός.
Fwd IAT Max - Μέγιστος χρόνος μεταξύ πακέτων προς τα εμπρός.
Fwd IAT Min - Ελάχιστος χρόνος μεταξύ πακέτων προς τα εμπρός.
Bwd IAT Total - Συνολικός χρόνος μεταξύ πακέτων προς τα πίσω.
Bwd IAT Mean - Μέσος χρόνος μεταξύ πακέτων προς τα πίσω.
Bwd IAT Std - Τυπική απόκλιση του χρόνου μεταξύ πακέτων προς τα πίσω.
Bwd IAT Max - Μέγιστος χρόνος μεταξύ πακέτων προς τα πίσω.
Bwd IAT Min - Ελάχιστος χρόνος μεταξύ πακέτων προς τα πίσω.
Fwd PSH Flags - Αριθμός πακέτων προς τα εμπρός με PSH σημαία.
Bwd PSH Flags - Αριθμός πακέτων προς τα πίσω με PSH σημαία.
Fwd URG Flags - Αριθμός πακέτων προς τα εμπρός με URG σημαία.
Bwd URG Flags - Αριθμός πακέτων προς τα πίσω με URG σημαία.
Fwd Header Length - Συνολικό μήκος κεφαλίδας πακέτων προς τα εμπρός.
Bwd Header Length - Συνολικό μήκος κεφαλίδας πακέτων προς τα πίσω.
Fwd Packets/s - Ρυθμός πακέτων προς τα εμπρός ανά δευτερόλεπτο.
Bwd Packets/s - Ρυθμός πακέτων προς τα πίσω ανά δευτερόλεπτο.
Packet Length Min - Ελάχιστο μήκος πακέτου στη ροή.
Packet Length Max - Μέγιστο μήκος πακέτου στη ροή.
Packet Length Mean - Μέσο μήκος πακέτου στη ροή.
Packet Length Std - Τυπική απόκλιση μήκους πακέτων στη ροή.
Packet Length Variance - Διακύμανση μήκους πακέτων στη ροή.
FIN Flag Count - Αριθμός πακέτων με FIN σημαία.
SYN Flag Count - Αριθμός πακέτων με SYN σημαία.
RST Flag Count - Αριθμός πακέτων με RST σημαία.
PSH Flag Count - Αριθμός πακέτων με PSH σημαία.
ACK Flag Count - Αριθμός πακέτων με ACK σημαία.
URG Flag Count - Αριθμός πακέτων με URG σημαία.
CWR Flag Count - Αριθμός πακέτων με CWR σημαία.
ECE Flag Count - Αριθμός πακέτων με ECE σημαία.
Down/Up Ratio - Αναλογία κατερχόμενων προς ανερχόμενα πακέτα.
Average Packet Size - Μέσο μέγεθος πακέτου στη ροή.
Fwd Segment Size Avg - Μέσο μέγεθος τμήματος προς τα εμπρός.
Bwd Segment Size Avg - Μέσο μέγεθος τμήματος προς τα πίσω.
Fwd Bytes/Bulk Avg - Μέσος αριθμός bytes ανά μαζική μεταφορά προς τα εμπρός.
Fwd Packet/Bulk Avg - Μέσος αριθμός πακέτων ανά μαζική μεταφορά προς τα εμπρός.
Fwd Bulk Rate Avg - Μέσος ρυθμός μαζικής μεταφοράς προς τα εμπρός.
Bwd Bytes/Bulk Avg - Μέσος αριθμός bytes ανά μαζική μεταφορά προς τα πίσω.
Bwd Packet/Bulk Avg - Μέσος αριθμός πακέτων ανά μαζική μεταφορά προς τα πίσω.
Bwd Bulk Rate Avg - Μέσος ρυθμός μαζικής μεταφοράς προς τα πίσω.
Subflow Fwd Packets - Αριθμός πακέτων προς τα εμπρός σε υποροή.
Subflow Fwd Bytes - Αριθμός bytes προς τα εμπρός σε υποροή.
Subflow Bwd Packets - Αριθμός πακέτων προς τα πίσω σε υποροή.
Subflow Bwd Bytes - Αριθμός bytes προς τα πίσω σε υποροή.
FWD Init Win Bytes - Αρχικό μέγεθος παραθύρου σε bytes προς τα εμπρός.
Bwd Init Win Bytes - Αρχικό μέγεθος παραθύρου σε bytes προς τα πίσω.
Fwd Act Data Pkts - Αριθμός ενεργών πακέτων δεδομένων προς τα εμπρός.
Fwd Seg Size Min - Ελάχιστο μέγεθος τμήματος προς τα εμπρός.
Active Mean - Μέσος χρόνος ενεργής περιόδου στη ροή.
Active Std - Τυπική απόκλιση χρόνου ενεργής περιόδου στη ροή.
Active Max - Μέγιστος χρόνος ενεργής περιόδου στη ροή.
Active Min - λάχιστος χρόνος ενεργής περιόδου στη ροή.
Idle Mean - Μέσος χρόνος αδράνειας στη ροή.
Idle Std - Τυπική απόκλιση χρόνου αδράνειας στη ροή.
Idle Max - Μέγιστος χρόνος αδράνειας στη ροή.
Idle Min - Ελάχιστος χρόνος αδράνειας στη ροή.
Label - Ετικέτα που υποδεικνύει αν η ροή είναι κανονική ή κακόβουλη.
Traffic Type - Τύπος κίνησης (π.χ., Audio, Video, DoS, Botnet).
Traffic SubType - Λεπτομερής κατηγοριοποίηση της ροής.
"""

# Εκτύπωση περιγραφών στο αρχείο αναφοράς
print(description_text)

# === ΕΠΑΝΑΦΟΡΑ ΣΤΗΝ ΚΟΝΣΟΛΑ ΚΑΙ ΕΝΗΜΕΡΩΣΗ ===
sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"\nΗ αναφορά αποθηκεύτηκε στο:\n{os.path.abspath(report_path)}")

# === ------------------------------------HISTOGRAMS------------------------------------------------------ ====
# === Ορισμός επιλεγμένων στηλών ===
selected_cols = [
    'Flow Duration', 'Fwd IAT Total', 'Flow IAT Max', 'Idle Max', 'Idle Mean', 'Fwd IAT Min', 
    'Fwd IAT Mean', 'Packet Length Std', 'Fwd Packet Length Max', 'Packet Length Mean'
]

# === ΙΣΤΟΓΡΑΜΜΑΤΑ ===
for col in selected_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True, bins=50)
    plt.title(f'Ιστόγραμμα της μεταβλητής: {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Πλήθος')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === ------------------------------------HISTOGRAMS-LOGY------------------------------------------------------ ====
for col in selected_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, bins=50)
    plt.title(f'Ιστόγραμμα της μεταβλητής: {col} (log y)', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Πλήθος')
    plt.yscale('log')  # Κάνει τις μικρές μπάρες πιο ορατές
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

# === ------------------------------------BOXPLOTS------------------------------------------------------ ====
# === BOXPLOTS ===
for col in selected_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot της μεταβλητής: {col}', fontsize=14)
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === ------------------------------------HEATMAP-SELECTED COLS------------------------------------------------------ ====
# === CORRELATION MATRIX ===
plt.figure(figsize=(14, 10))
corr_matrix = df[selected_cols].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title("Πίνακας συσχετίσεων για τις επιλεγμένες μεταβλητές")
plt.tight_layout()
plt.show()

# === ------------------------------------COUNTPLOTS------------------------------------------------------ ====
# === Ορισμός κατηγορικών μεταβλητών ===
categorical_cols = ['Timestamp', 'Flow ID', 'Dst Port', 'Src IP', 'Label', 'Traffic Type', 'Traffic Subtype']

# === COUNTPLOTS ===
for col in categorical_cols:
    value_counts = df[col].value_counts()
    unique_vals = len(value_counts)
    
    # Αν έχει πολλές τιμές, κράτα μόνο τις 10 πιο συχνές
    if unique_vals > 20:
        top_vals = value_counts.nlargest(10).index
        plot_data = df[df[col].isin(top_vals)]
        title_note = " (Top 10)"
    else:
        plot_data = df
        title_note = ""

    plt.figure(figsize=(14, 6))
    sns.countplot(data=plot_data, x=col, order=plot_data[col].value_counts().index)
    
    #λογαριθμική κλίμακα στον άξονα y
    plt.yscale("log")
    
    plt.title(f'Κατανομή της μεταβλητής: {col}{title_note}')
    plt.xticks(rotation=45 if unique_vals <= 10 else 60, ha='right')
    plt.tight_layout()
    plt.show()

# === ------------------------------------VIOLIN PLOTS------------------------------------------------------ ====
# === Κατηγορικές μεταβλητές για τον άξονα Χ ===
categorical_targets = ['Label', 'Traffic Type', 'Traffic Subtype']

# === Violin Plots για κάθε αριθμητική μεταβλητή ανά κατηγορική ===
for cat_col in categorical_targets:

    # Ειδική επεξεργασία μόνο για το Traffic Subtype (κρατάει τις 5 πιο συχνές τιμές)
    if cat_col == 'Traffic Subtype':
        top_values = df[cat_col].value_counts().nlargest(5).index
        plot_df = df[df[cat_col].isin(top_values)]
        title_suffix = " (Top 5)"
    else:
        plot_df = df.copy()
        title_suffix = ""

    plot_df[selected_cols] = plot_df[selected_cols].apply(lambda x: np.log10(x + 1))

    for col in selected_cols:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=plot_df, x=cat_col, y=col, inner='quartile', density_norm='width')
        plt.title(f"Κατανομή της '{col}' ανά '{cat_col}'{title_suffix}", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === ------------------------------------HEATMAP CORR>0.4------------------------------------------------------ ====
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# === Υπολογισμός μόνο του upper triangle (χωρίς διαγώνιο) ===
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# === Φιλτράρισμα για corr > 0.4 ===
threshold = 0.4
filtered_corr = upper.stack().reset_index()
filtered_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
strong_corrs = filtered_corr[filtered_corr['Correlation'] > threshold]

# === Ταξινόμηση κατά φθίνουσα συσχέτιση ===
strong_corrs = strong_corrs.sort_values(by='Correlation', ascending=False)

top_corrs = strong_corrs.head(30)  
top_cols = pd.unique(top_corrs[['Feature 1', 'Feature 2']].values.ravel())
top_corr_matrix = corr_matrix.loc[top_cols, top_cols]

fig = px.imshow(
    top_corr_matrix,
    text_auto='.2f',
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1,
    title=f"Top-30 Συσχετίσεις Μεταβλητών (corr > {threshold})"
)
fig.update_layout(
    xaxis={'side': 'bottom'},
    width=1000,
    height=1000
)
fig.show()

# === Αποθήκευση συσχετίσεων σε αρχείο ===
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"correlation_pairs_over_{threshold}_{timestamp}.txt"

with open(filename, 'w', encoding='utf-8') as f:
    f.write(f"Ζεύγη μεταβλητών με συσχέτιση > {threshold} (φθίνουσα σειρά):\n\n")
    for _, row in strong_corrs.iterrows():
        line = f"• {row['Feature 1']} <--> {row['Feature 2']}  |  corr = {row['Correlation']:.2f}\n"
        f.write(line)

print(f"\nΟι συσχετίσεις αποθηκεύτηκαν στο αρχείο: {filename}")
