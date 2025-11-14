from dataset_loader import load_dataset
from model_cnn import build_cnn
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(data_dir, save_path="saved_models/cnn_model.h5"):
    X_train, X_test, y_train, y_test = load_dataset(data_dir)

    model = build_cnn()

    cp = ModelCheckpoint(save_path, save_best_only=True,
                         monitor='val_accuracy', mode='max')

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=20, batch_size=32, callbacks=[cp])

    print("Training completed. Model saved at:", save_path)
