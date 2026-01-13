import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# Φάκελος αποθήκευσης
output_dir = "nnHdbscan_results"
os.makedirs(output_dir, exist_ok=True)

# Συνάρτηση φόρτωσης και split σε train/val/test 
def load_and_prepare_data(path, target_col):
    df = pd.read_csv(path)
    X = df[[col for col in df.columns if col.startswith("PC")]].values
    y = df[target_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test), le

# Συνάρτηση απεικόνισης confusion matrix 
def plot_confusion(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Greens")
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"conf_matrix_{title.replace(' ', '_')}.png"))
    plt.show()

# Συνάρτηση για learning curves 
def plot_learning_curve(model, title, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, label='Training Loss')
    if hasattr(model, 'validation_scores_'):
        plt.plot(model.validation_scores_, label='Validation Score')
    plt.title(f'Learning Curve - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Score')
    plt.legend()
    plt.grid(True)
    filename = f"learning_curve_{title.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# Λίστα datasets
datasets = {
    "final_hdbscan_balanced": "final_hdbscan_balanced.csv",
    
}

# Εκπαίδευση και αξιολόγηση 
results = []

for name, path in datasets.items():
    for target in ['Label', 'Traffic Type']:
        print(f"\n--- {name} | Target: {target} ---")
        (X_train, X_val, X_test, y_train, y_val, y_test), le = load_and_prepare_data(path, target)

        # Υπολογισμός class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        sample_weights = np.array([class_weights[label] for label in y_train])
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42
        )

        mlp.fit(X_train, y_train)

        # Learning Curve
        plot_learning_curve(mlp, f"{name} - MLP - {target}", output_dir)

        # Προβλέψεις
        y_train_pred = mlp.predict(X_train)
        y_val_pred = mlp.predict(X_val)
        y_test_pred = mlp.predict(X_test)

        # Αναφορές & Confusion Matrices
        print("\n>>> Training Report")
        train_report = classification_report(y_train, y_train_pred, target_names=le.classes_, output_dict=True)
        print(classification_report(y_train, y_train_pred, target_names=le.classes_,zero_division=0))
        plot_confusion(y_train, y_train_pred, le.classes_, f"{name} - {target} - Train")

        print("\n>>> Validation Report")
        val_report = classification_report(y_val, y_val_pred, target_names=le.classes_, output_dict=True)
        print(classification_report(y_val, y_val_pred, target_names=le.classes_,zero_division=0))
        plot_confusion(y_val, y_val_pred, le.classes_, f"{name} - {target} - Validation")

        print("\n>>> Test Report")
        test_report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True)
        print(classification_report(y_test, y_test_pred, target_names=le.classes_,zero_division=0))
        plot_confusion(y_test, y_test_pred, le.classes_, f"{name} - {target} - Test")

        # Συγκεντρωτικά Αποτελέσματα
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

# Δημιουργία πίνακα αποτελεσμάτων
summary_df = pd.DataFrame(results)
print("\nΣυνοπτικά Αποτελέσματα:")
print(summary_df)

# Αποθήκευση σε CSV 
summary_csv_path = os.path.join(output_dir, "mlp_f1_scores_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

# Barplot F1-score ανά set
plt.figure(figsize=(10, 6))
summary_df_melted = summary_df.melt(id_vars=['Dataset', 'Target'], 
                                     value_vars=['F1-score Train', 'F1-score Val', 'F1-score Test'],
                                     var_name='Set', value_name='F1-score')
sns.barplot(data=summary_df_melted, x='Target', y='F1-score', hue='Set', palette='Set2')
plt.title("MLP - Σύγκριση F1-score ανά Target και Set")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mlp_f1_score_set_comparison.png"))
plt.show()
