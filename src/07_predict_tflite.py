import os
import sys
import json
import numpy as np
import tensorflow as tf
from config import IMG_SIZE, MODELS_DIR, CLASS_NAMES

TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "int8.tflite")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "best_threshold.json")


def load_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", 0.5))
    return 0.5


def load_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr


def quantize_input(image_float, input_details):
    scale, zero_point = input_details["quantization"]

    if scale == 0:
        raise ValueError("Input scale is 0. Invalid TFLite quantization parameters.")

    image_uint8 = image_float / scale + zero_point
    image_uint8 = np.clip(np.round(image_uint8), 0, 255).astype(input_details["dtype"])
    return image_uint8


def dequantize_output(output_data, output_details):
    scale, zero_point = output_details["quantization"]

    if scale == 0:
        return output_data.astype(np.float32)

    return scale * (output_data.astype(np.float32) - zero_point)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/06_predict_tflite.py <image_path>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"TFLite model not found: {TFLITE_MODEL_PATH}")
        return

    threshold = 0.745 if load_threshold() < 0.745 else 0.55

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("TFLite model:", TFLITE_MODEL_PATH)
    print("Threshold Used:", f"{threshold:.3f}")
    print("Input dtype:", input_details["dtype"])
    print("Output dtype:", output_details["dtype"])

    image_float = load_image(image_path)

    # converter.inference_input_type = tf.uint8
    input_data = quantize_input(image_float, input_details)

    interpreter.set_tensor(input_details["index"], input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details["index"])
    prob = float(dequantize_output(raw_output, output_details).flatten()[0])

    pred = 1 if prob >= threshold else 0

    print("Probability:", f"{prob:.4f}")
    print("Prediction:", CLASS_NAMES[pred])


if __name__ == "__main__":
    main()