import os
import tensorflow as tf
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED


def main():
    # Training split
    # Both train and val datasets are built from the same "train" directory;
    # Keras handles the VAL_SPLIT / SEED-based partition internally.
    # NB: this script is a quick sanity-check loader only — the actual
    # training pipeline (02_train_baseline.py) uses stratified_split() instead,
    # which preserves class balance across the split.
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",            # single float label: 0=NORMAL, 1=PNEUMONIA
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    # Test split
    # shuffle=False keeps a deterministic ordering, which is required so that
    # predicted probabilities align with their ground-truth labels during evaluation.

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    print("Train classes:", train_dataset.class_names)
    print("Prepared train, validation, and test datasets successfully.")


if __name__ == "__main__":
    main()