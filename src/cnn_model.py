import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU detected. Running on CPU.")
    else:
        print(f"GPUs detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def prepare_data(df, img_size=(224, 224)):
    """
    Load heatmap images in color, extract velocity from red and blue channels.

    Args:
        df (pd.DataFrame): DataFrame with file_path and activity columns.
        img_size (tuple): Target image size (height, width).

    Returns:
        X (np.array): Velocity maps with shape (N, height, width, 1).
        y (np.array): Encoded labels.
    """
    label_map = {"chopsticks": 0, "fork": 1, "bare_hand": 2, "fork_knife": 3, "spoon": 4}

    X = []
    y = []

    for _, row in df.iterrows():
        # Load image in color (BGR)
        img = cv2.imread(row["file_path"])
        if img is None:
            continue

        # Resize to 224x224
        img = cv2.resize(img, img_size)

        # Split into B, G, R channels
        blue, green, red = cv2.split(img)

        # Compute velocity: red (positive) - blue (negative)
        # Normalize channels to [0, 1], then compute velocity in [-1, 1]
        red = red.astype(float) / 255.0
        blue = blue.astype(float) / 255.0
        green = green.astype(float) / 255.0

        # Velocity: red contributes positively, blue negatively
        velocity = red - blue  # Range [-1, 1]

        # Ignore green channel (minimal in red-blue colormap)
        # Add channel dimension (224, 224, 1)
        velocity = np.expand_dims(velocity, axis=-1)

        X.append(velocity)
        y.append(label_map[row["activity"]])

    X = np.array(X)
    y = np.array(y)

    return X, y


def build_cnn(input_shape=(224, 224, 1), num_classes=5):
    """
    Build the CNN model with 3 conv+pool layers, flatten, and 2 dense layers.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def main():
    # Check GPU
    check_gpu()

    # Define paths
    data_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs\csv\all_activities.csv")
    output_dir = Path(r"C:\Users\Ahmed\OneDrive\Bureau\mmEat\outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(data_dir)
    print("Dataset loaded successfully:")
    print(df["activity"].value_counts())

    # Prepare data
    X, y = prepare_data(df)
    print(f"Data prepared: X shape={X.shape}, y shape={y.shape}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    model = build_cnn()
    model.summary()

    # Train model

    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1,callbacks=[early_stop])

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"]))

    # Save model
    model.save(output_dir / "cnn_model_velocity.h5")
    print(f"Model saved to {output_dir / 'cnn_model_velocity.h5'}")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history_velocity.png")
    print(f"Training history plot saved to {output_dir / 'training_history_velocity.png'}")



    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"],
                yticklabels=["chopsticks", "fork", "bare_hand", "fork_knife", "spoon"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    plt.savefig(output_dir / "confusion_matrix_velocity.png")
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix_velocity.png'}")

if __name__ == "__main__":
    main()