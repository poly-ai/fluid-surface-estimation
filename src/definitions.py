import os

# Root
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)

# Datasets
DATA_DIR = os.path.join(ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Models
TRAINED_MODEL_DIR = os.path.join(ROOT, "models")

if __name__ == "__main__":

    # Print all variables for debugging
    all_variables = dir()
    for name in all_variables:
        if not name.startswith("__") and name not in ("os", "Path"):
            value = eval(name)
            print("{:<24} = {}".format(name, value))
