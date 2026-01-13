import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Φάκελος αποθήκευσης αποτελεσμάτων 
output_dir = "svmKmeans_results"
os.makedirs(output_dir, exist_ok=True)

# Συνάρτηση για φόρτωση και διαχωρισμό δεδομένων
def load_and_prepare_data(path, target_col, test_size=0.2):
    df = pd.read_csv(path)
    X = df[[col for col in df.columns if col.startswith("PC")]].values
    y = df[target_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Πρώτο split: 20% για test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    # Δεύτερο split: 25% από τα 80% => 20% του συνόλου για validation
    val_ratio = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val), le

# Συνάρτηση για οπτικοποίηση confusion matrix 
def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"conf_matrix_{title.replace(' ', '_')}.png"))
    plt.show()

# Ορισμός datasets
datasets = {
    "minibatch_kmeans_balanced_with_smote": "minibatch_kmeans_balanced_with_smote.csv"
}

# Εκπαίδευση και αξιολόγηση SVM
results = []

for name, path in datasets.items():
    for target in ['Label', 'Traffic Type']:
        print(f"\n--- {name} | Target: {target} ---")
        (X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val), le = load_and_prepare_data(path, target)

        print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

        # Δημιουργία και εκπαίδευση SVM
        svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
        svm.fit(X_train, y_train)

        # Αξιολόγηση στο training set 
        y_train_pred = svm.predict(X_train)
        train_report = classification_report(y_train, y_train_pred, target_names=le.classes_, output_dict=True)
        print(">>> Training Report")
        print(classification_report(y_train, y_train_pred, target_names=le.classes_))
        plot_confusion(y_train, y_train_pred, le.classes_, f"{name} - {target} - Train")

        # Αξιολόγηση στο validation set 
        y_val_pred = svm.predict(X_val)
        val_report = classification_report(y_val, y_val_pred, target_names=le.classes_, output_dict=True)
        print(">>> Validation Report")
        print(classification_report(y_val, y_val_pred, target_names=le.classes_))
        plot_confusion(y_val, y_val_pred, le.classes_, f"{name} - {target} - Validation")

        # Αξιολόγηση στο test set 
        y_test_pred = svm.predict(X_test)
        test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)
        print(">>> Test Report")
        print(classification_report(y_test, y_test_pred, target_names=le.classes_))
        plot_confusion(y_test, y_test_pred, le.classes_, f"{name} - {target} - Test")

        # Αποθήκευση συγκεντρωτικών μετρικών
        results.append({
            'Dataset': name,
            'Target': target,
            'F1-score Train': train_report['weighted avg']['f1-score'],
            'F1-score Val': val_report['weighted avg']['f1-score'],
            'F1-score Test': test_report['weighted avg']['f1-score'],
            'Accuracy Train': train_report['accuracy'],
            'Accuracy Val': val_report['accuracy'],
            'Accuracy Test': test_report['accuracy']
        })

        # Learning Curve
        print(f">>> Υπολογισμός Learning Curve για {name} | Target: {target}")
        svm_for_curve = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
        train_sizes, train_scores, val_scores = learning_curve(
            svm_for_curve, X_train_val, y_train_val, cv=5, scoring='f1_weighted', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', label='Training F1-score')
        plt.plot(train_sizes, val_scores_mean, 'o-', label='Validation F1-score')
        plt.title(f"Learning Curve - {name} | Target: {target}")
        plt.xlabel("Training Set Size")
        plt.ylabel("F1-score (weighted)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"learning_curve_{name.replace(' ', '_')}_{target.replace(' ', '_')}.png"))
        plt.show()

# Δημιουργία πίνακα αποτελεσμάτων
summary_df = pd.DataFrame(results)
print("\nΣυνοπτικά Αποτελέσματα:")
print(summary_df)

summary_csv_path = os.path.join(output_dir, "svm_f1_scores_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

# Bar Plot για F1-score ανά σύνολο 
summary_melted = summary_df.melt(
    id_vars=['Dataset', 'Target'],
    value_vars=['F1-score Train', 'F1-score Val', 'F1-score Test'],
    var_name='Split',
    value_name='F1-score'
)

plt.figure(figsize=(10, 6))
sns.barplot(data=summary_melted, x='Target', y='F1-score', hue='Split', palette='Set2')
plt.title("SVM - F1-score ανά Dataset / Target / Split")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "svm_f1_score_comparison_all_splits.png"))
plt.show()
