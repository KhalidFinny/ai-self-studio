import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for Palm/Fist Recognition")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset"), help="Path to the raw dataset directory")
    parser.add_argument("--model_dir", type=str, default="models/resnet50", help="Directory to save the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    return parser.parse_args()

def build_model(input_shape):
    """
    Builds the ResNet50 model with custom top layers.
    Includes on-the-fly augmentation layers.
    """
    # Data Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name="data_augmentation")

    # Base Model
    base_model = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    
    # Fine-tuning configuration
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    
    # Preprocess input for ResNet50 (scales to [-1, 1] or similar expected by ResNet)
    # Wrapping in Lambda to avoid pickling issues with module references
    x = layers.Lambda(applications.resnet50.preprocess_input, name='preprocess_input')(x)
    
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model

def main():
    args = get_args()
    
    print(f"[INFO] Dataset Directory: {args.dataset_dir}")
    if not os.path.exists(args.dataset_dir):
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        return

    # Create datasets using image_dataset_from_directory
    # We use a 70-20-10 split roughly, but image_dataset_from_directory only does train/val split.
    # So we'll do 80-20 first, then take a chunk of val for test if needed, 
    # or just use 80/20 for simplicity in this pipeline as requested "1 file pipeline".
    # Let's stick to a standard Train/Val split for simplicity and robustness.
    
    print("[INFO] Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="binary"
    )

    class_names = train_ds.class_names
    print(f"[INFO] Classes found: {class_names}")

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build Model
    print("[INFO] Building model...")
    model = build_model((args.img_size, args.img_size, 3))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    # Callbacks
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, "best_model.keras")
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Train
    print("[INFO] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print(f"[INFO] Training finished. Best model saved to {checkpoint_path}")

    # Evaluate on Validation Set
    print("[INFO] Evaluating on validation set...")
    # Load best model weights
    model.load_weights(checkpoint_path)

    y_true = []
    y_pred_probs = []

    # Iterate over the validation dataset to get true labels and predictions
    # Note: We iterate directly to ensure alignment between images and labels
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Convert probabilities to binary predictions (threshold 0.5)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n[INFO] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
