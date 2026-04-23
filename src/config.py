import os

# Image & Training Hyperparameters
# MobileNetV2 expects 224×224 RGB input; BATCH_SIZE of 32
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42
VAL_SPLIT = 0.2

# Paths
# DATA_DIR points to the Kaggle chest X-ray dataset structure:
#   chest_xray/
#     train/  NORMAL/  PNEUMONIA/
#     test/   NORMAL/  PNEUMONIA/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "versions", "2", "chest_xray", "chest_xray")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, "baseline_savedmodel")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "int8.tflite")   # INT8-quantized model for edge deployment

# Class order matches tf.keras directory-scan alphabetical order (NORMAL=0, PNEUMONIA=1)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]