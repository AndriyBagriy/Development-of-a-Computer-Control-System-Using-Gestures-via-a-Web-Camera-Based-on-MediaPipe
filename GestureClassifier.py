import os.path
from datetime import datetime
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split


class GestureClassifier:
    def __init__(self, dataset_path, model_save_path, tflite_save_path, backup_path):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.tflite_save_path = tflite_save_path
        self.backup_path = backup_path
        self.num_classes = 0
        self.model = None

    def load_data(self):
        x_dataset = np.loadtxt(self.dataset_path, delimiter=",", dtype="float32", usecols=list(range(1, (21*2)+1)))
        y_dataset = np.loadtxt(self.dataset_path, delimiter=",", dtype="int", usecols=(0))
        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=42)
        return x_train, x_test, y_train, y_test

    def backup_dataset(self):
        if os.path.exists(self.dataset_path):
            timestamp = datetime.now().strftime("[%d.%m.%Y %H:%M:%S]")
            with open(self.backup_path, 'a') as backup:
                id_dataset = np.loadtxt(self.dataset_path, delimiter=',', dtype="int32", usecols=(0))
                unique_ids = np.unique(id_dataset)
                backup.write(f"{timestamp} [id's = {','.join(map(str, unique_ids))}]\n")
                with open(self.dataset_path, 'r') as dataset:
                    backup.writelines(dataset.readlines())
            # open(self.dataset_path, 'w').close()
            print(f"Dataset moved to backup: {self.backup_path}")
        else:
            print("Dataset file not found")

    def build_model(self, num_classes):
        self.model = keras.models.Sequential([
            keras.layers.Input((21*2, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, x_train, y_train, x_test, y_test, epochs=1000, batch_size=128):
        cp_callback = keras.callbacks.ModelCheckpoint(
            self.model_save_path, verbose=1, save_weights_only=False
        )
        es_callback = keras.callbacks.EarlyStopping(patience=20, verbose=1)

        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[cp_callback, es_callback]
        )

    def evaluate_model(self, x_test, y_test):
        val_loss, val_acc = self.model.evaluate(x_test, y_test, batch_size=128)
        print(f"Test accuracy: {val_acc:.4f}, Test Loss: {val_loss:.4f}")

    def save_as_tflite(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(self.tflite_save_path, 'wb') as file:
            file.write(tflite_model)
        print(f"Model saved in TensorFlow Lite format at: {self.tflite_save_path}")

    def update_model(self):
        if not os.path.exists(self.dataset_path):
            print("Dataset file not found.")
            return

        x_train, x_test, y_train, y_test = self.load_data()
        num_classes = len(np.unique(np.concatenate((y_train, y_test))))
        print(f"num_classes in build model = {num_classes}")

        self.build_model(num_classes)
        self.train_model(x_train, y_train, x_test, y_test)
        self.evaluate_model(x_test, y_test)
        # self.backup_dataset()

    def predict(self, sample_data):
        if os.path.exists(self.model_save_path):
            if self.model is None:
                self.model = keras.models.load_model(self.model_save_path)
            flat_sample_data = np.array(sample_data).flatten()
            prediction = self.model.predict(np.array([flat_sample_data]))
            return np.argmax(prediction)

    # def add_new_data_and_train(self, update_existing=False):
    #     if not os.path.exists(self.dataset_path):
    #         print("Dataset file not found.")
    #         return
    #
    #     x_new = np.loadtxt(self.dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    #     y_new = np.loadtxt(self.dataset_path, delimiter=',', dtype='int32', usecols=(0))
    #     new_classes = np.unique(y_new)
    #
    #     if os.path.exists(self.model_save_path):
    #         self.load_existing_model()
    #     else:
    #         self.build_model(len(new_classes))
    #
    #     existing_classes = list(range(self.num_classes))
    #     additional_classes = [cls for cls in new_classes if cls not in existing_classes]
    #
    #     if additional_classes:
    #         print(f"Adding new classes: {additional_classes}")
    #         self.num_classes += len(additional_classes)
    #         self.build_model(self.num_classes)
    #
    #     if not update_existing:
    #         indices = [i for i, label in enumerate(y_new) if label in additional_classes]
    #         x_new = x_new[indices]
    #         y_new = y_new[indices]
    #
    #     x_train, x_test, y_train, y_test = self.load_data()
    #     x_train = np.concatenate((x_train, x_new), axis=0)
    #     y_train = np.concatenate((y_train, y_new), axis=0)
    #
    #     self.train_model(x_train, y_train, x_test, y_test)
    #     self.evaluate_model(x_test, y_test)
    #     self.backup_dataset()

    # def load_existing_model(self):
    #     if os.path.exists(self.model_save_path):
    #         self.model = keras.models.load_model(self.model_save_path)
    #         self.num_classes = self.model.layers[-1].output_shape[-1]
    #         print("Existing model loaded.")
    #     else:
    #         print("No existing model found. Building a new one.")
    #         return -1



"""
    classifier = GestureClassifier(dataset_path, model_save_path, tflite_save_path, backup_path)
    classifier.add_new_data_and_train()
    classifier.save_as_tflite()
    
    predicted_class = classifier.predict(sample_data)
"""