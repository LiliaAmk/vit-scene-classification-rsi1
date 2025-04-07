import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your pre-trained model (for example, a ViT or Swin transformer)
model = tf.keras.models.load_model('vit_model.pth')

# Function to preprocess and classify the image
def classify_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))  # Ensure 224x224 size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    # Predict the class of the image
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)

    # Return the class index or class label (adjust as needed)
    return f"Predicted Class: {class_index[0]}"
