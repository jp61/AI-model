from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory path definitions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 150 # This must match the size used during model training
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images") # Folder containing your test photos
# MODEL_FILENAME = "cats_dogs_model.keras" # model file name
MODEL_FILENAME = "cats_dogs_model_fine_tuned.keras" # Fine-tuned model file name

def load_and_preprocess(img_path):
    """
    Function to prepare a raw image for model prediction.
    """
    # Load the image from disk and resize it to 150x150
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    # Convert the image object into a numerical array (numpy array)
    img_array = image.img_to_array(img)
    # Normalization: rescale pixel values from 0-255 to 0-1 (matches training logic)
    img_array = img_array / 255.0
    # Expand dimensions: The model expects a batch of images.
    # This transforms the shape from (150, 150, 3) to (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the pre-trained model from the .keras file
model = keras.models.load_model(os.path.join(SCRIPT_DIR, MODEL_FILENAME))

# Scan the 'images' folder and list all image files (jpg, jpeg, png)
img_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Loop through every image found in the directory
for img_name in img_files:
    img_path = os.path.join(IMAGES_DIR, img_name)
    
    # 1. Preprocess the image
    img_tensor = load_and_preprocess(img_path)
    
    # 2. Perform prediction (Model returns a value between 0 and 1)
    pred = model.predict(img_tensor)
    
    # 3. Decision Logic: 
    # If the sigmoid output is > 0.5, it's a Dog. Otherwise, it's a Cat.
    label = "Dog" if pred[0][0] > 0.5 else "Cat"
    
    # Print result to terminal including confidence score
    print(f"{img_name}: {label} (confidence: {pred[0][0]:.2f})")

    # --- Visual Display ---
    # Reload the image for display purposes
    img_display = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    plt.imshow(img_display)
    plt.title(f"{img_name}: {label}")
    plt.axis('off') # Hide graph axes for a cleaner look
    plt.show() # Display the window with the image and prediction