import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, MODELS_DIR

MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")
PLOT_PATH = os.path.join(MODELS_DIR, "test_threshold_analysis.png")
THRESHOLD_JSON_PATH = os.path.join(MODELS_DIR, "best_threshold.json")


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def load_model_for_inference():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def load_eval_dataset():
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    class_names = ds.class_names
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def collect_labels_and_probs(model, ds):
    labels = []
    for _, y in ds:
        labels.extend(y.numpy().flatten())
    labels = np.array(labels, dtype=int)

    probs = model.predict(ds, verbose=1).flatten()
    return labels, probs


def evaluate_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_best_threshold(best_result):
    with open(THRESHOLD_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": float(best_result["threshold"])}, f, indent=2)


def plot_results(thresholds, results, best_result):
    balanced_accuracies = [r["balanced_accuracy"] for r in results]
    recalls = [r["recall"] for r in results]
    specificities = [r["specificity"] for r in results]
    precisions = [r["precision"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, specificities, label="Specificity")
    plt.plot(thresholds, precisions, label="Precision")
    plt.axvline(
        best_result["threshold"],
        linestyle="--",
        label=f"Best t={best_result['threshold']:.2f}",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Test Threshold Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


def main():
    model = load_model_for_inference()
    eval_ds, class_names = load_eval_dataset()

    print("Class names:", class_names)

    labels, probs = collect_labels_and_probs(model, eval_ds)

    thresholds = np.linspace(0.10, 0.90, 200)
    results = [evaluate_threshold(labels, probs, t) for t in thresholds]
    best_result = max(results, key=lambda x: x["balanced_accuracy"])

    print("\nBEST THRESHOLD (analysis only)")
    print(f"Threshold          : {best_result['threshold']:.3f}")
    print(f"Balanced Accuracy  : {best_result['balanced_accuracy']:.4f}")
    print(f"Accuracy           : {best_result['accuracy']:.4f}")
    print(f"Precision          : {best_result['precision']:.4f}")
    print(f"Recall             : {best_result['recall']:.4f}")
    print(f"Specificity        : {best_result['specificity']:.4f}")
    print(
        f"Confusion Matrix   : [[{best_result['tn']}, {best_result['fp']}], "
        f"[{best_result['fn']}, {best_result['tp']}]]"
    )

    save_best_threshold(best_result)
    print(f"Saved threshold to: {THRESHOLD_JSON_PATH}")

    plot_results(thresholds, results, best_result)
    print(f"Saved plot to: {PLOT_PATH}")


if __name__ == "__main__":
    main()