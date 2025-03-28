# trained_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Define the CNN Model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Define your training function
def train_model():
    # Load the dataset
    train_ds = tf.keras.utils.image_dataset_from_directory("final_data_set_no_bg_splitted/train", image_size=(224, 224), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory("final_data_set_no_bg_splitted/val", image_size=(224, 224), batch_size=32)

    # Create and train the model
    model = create_model()
    model.fit(train_ds, validation_data=val_ds, epochs=20)

    # Save the trained model
    model.save("burn_classification_cnn_final_data_set_no_bg_splitted.keras")

# Ensure training happens only when running the script directly
if __name__ == "__main__":
    train_model()
