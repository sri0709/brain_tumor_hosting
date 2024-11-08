import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the path to the saved model
model_path = "brain_tumor_detection_model.h5"
model = load_model(model_path)

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Function to preprocess and predict an image
def predict_image(image_path):
    image_size = (150, 150)  # Use the same image size as in training

    # Load and preprocess the image
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_category = categories[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display the image and prediction result
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_category} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

# Provide the path to an image for prediction
image_path = r"D:\Brain-Tumor-Detection-main\Testing\glioma\Te-gl_0010.jpg"
predict_image(image_path)
