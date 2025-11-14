import os
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_image

def load_dataset(data_dir, img_size=224, test_split=0.2):
    X = []
    y = []

    for label in ["healthy", "parkinson"]:
        folder = os.path.join(data_dir, label)

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            try:
                img = preprocess_image(path, img_size)
                X.append(img)
                y.append(0 if label == "healthy" else 1)
            except:
                continue

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_split, random_state=42)
