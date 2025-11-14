import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from dataset_loader import load_dataset

def evaluate_model(data_dir, model_path):
    model = load_model(model_path)
    _, X_test, _, y_test = load_dataset(data_dir)

    preds = (model.predict(X_test) > 0.5).astype("int32")

    print("Classification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
