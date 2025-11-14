import cv2
import numpy as np

def preprocess_image(img_path, img_size=224):
    """
    Loads and preprocesses the image for model input
    """
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0   # normalize
    img = np.expand_dims(img, axis=-1)  # (H,W,1)
    return img

