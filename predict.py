import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import trained_model as tmodel

# Clear session to avoid any interference from previous sessions
tf.keras.backend.clear_session()

# Load the trained model
model = load_model("burn_classification_cnn.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Define class labels (match folder names)
class_labels = ["1st Degree Burn", "2nd Degree Burn", "3rd Degree Burn"]

# Load and preprocess the image
img_path = "Ak1n02 bg-rem test_eren Second_Degree_removed/0363_jpg.rf.16849040a5c99a29084160aaf77f9825.png"  # Change this to your image path
img_array = preprocess_image(img_path)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)  # Get class with highest probability

# Print the result
print(f"Predicted Burn Classification: {class_labels[predicted_class]}")

test_ds= tf.keras.utils.image_dataset_from_directory("raw_splitted/test", image_size=(224, 224), batch_size=32)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_ds)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

