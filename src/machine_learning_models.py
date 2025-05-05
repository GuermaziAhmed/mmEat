import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(df, img_size=(224, 224)):
    label_map = {"chopsticks": 0, "fork": 1, "bare_hand": 2, "fork_knife": 3, "spoon": 4}
    X, y = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["file_path"])
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        blue, _, red = cv2.split(img)
        red = red.astype(float) / 255.0
        blue = blue.astype(float) / 255.0
        velocity = np.expand_dims(red - blue, axis=-1)
        X.append(velocity)
        y.append(label_map[row["activity"]])
    X = np.array(X)
    y = np.array(y)
    return X, y

def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # row-wise normalization

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title + " (Normalized %)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_classic_ml_models(X, y, output_dir):
    X_flat = X.reshape(X.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "SVM": SVC(kernel='rbf'),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier()
    }

    labels = ["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]

    metrics_summary = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=labels))

        # Save confusion matrix plot
        plot_confusion_matrix(
            y_test, y_pred, labels,
            title=f"{name} Confusion Matrix",
            output_path=output_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
        print(f"{name} confusion matrix saved.")

        # Compute metrics
        accuracy = np.mean(y_pred == y_test)
        fdr = np.sum(y_pred != y_test) / len(y_test)

        metrics_summary.append({
            "Model": name,
            "Accuracy": accuracy,
            "FDR": fdr
        })

    # Save metrics table as CSV
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    print("Metrics summary saved to metrics_summary.csv")

    # Create bar plot
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(metrics_df["Model"]))

    plt.bar(x - bar_width/2, metrics_df["Accuracy"], bar_width, label="Accuracy", color='skyblue')
    plt.bar(x + bar_width/2, metrics_df["FDR"], bar_width, label="FDR", color='salmon')

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Accuracy and False Detection Rate per Model")
    plt.xticks(x, metrics_df["Model"])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_fdr_comparison.png")
    plt.close()
    print("Bar plot saved to accuracy_fdr_comparison.png")

def main():
    # Paths
    data_path = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = pd.read_csv(data_path)
    print(df["activity"].value_counts())
    X, y = prepare_data(df)
    print(f"Prepared data: {X.shape}, {y.shape}")

    # Run classic ML models
    run_classic_ml_models(X, y, output_dir)

if __name__ == "__main__":
    main()
