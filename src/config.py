import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42
VAL_SPLIT = 0.2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "versions", "2", "chest_xray", "chest_xray")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline_savedmodel")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "int8.tflite")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]