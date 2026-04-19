import os
import tensorflow as tf
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT, SEED


def main():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
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