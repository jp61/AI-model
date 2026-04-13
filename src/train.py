import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Get the absolute path of the current script's directory for saving the model later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Dataset ---
# Download the 'cats_vs_dogs' images and split them into training (80%) and validation (20%) sets
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],  
    with_info=True,        # Returns metadata about the dataset (number of images, labels, etc.)
    as_supervised=True,    # Returns data as a tuple (image, label) instead of a dictionary
)

# --- Preprocessing ---
IMG_SIZE = 150 # Target image resolution (150x150 pixels)

def preprocess(image, label):
    # Resize every image to a uniform size of 150x150
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalization: convert pixel values from 0-255 to 0-1 range (improves training stability)
    image = image / 255.0  
    return image, label

# Data pipeline preparation: 
# .map -> apply preprocessing
# .shuffle -> randomize order to prevent the model from learning patterns based on sequence
# .batch -> process 32 images at a time
# .prefetch -> prepare the next batch while the current one is being processed (performance boost)
train_ds = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

print("Dataset is ready ✅")


from tensorflow.keras import layers, models

# --- CNN Model Architecture ---
# Building a Convolutional Neural Network (CNN) specifically for image recognition
model = models.Sequential([
    # 1st Convolutional layer: extracts basic features like edges and corners
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2), # Downsamples the data to reduce computation while keeping important features

    # 2nd and 3rd layers: detect more complex patterns (like ears, eyes, or textures)
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(), # Flattens the 2D image data into a 1D list of numbers
    layers.Dense(512, activation='relu'), # Dense layer for high-level reasoning
    layers.Dense(1, activation='sigmoid')  # Output: scalar between 0 and 1 (probability of being a dog)
])

# Training configuration:
# Optimizer (Adam) adjusts weights to minimize error
# Loss function (binary_crossentropy) calculates how far the prediction is from the truth
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model is ready ✅")


# --- Training Process ---
# The model iterates through the entire dataset 10 times (epochs) to learn and improve accuracy
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# --- Saving the Model ---
# Explicitly define the input shape before saving
model.build(input_shape=(None, 150, 150, 3))

# Save the trained "brain" in both the legacy .h5 format and the modern .keras format
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model.h5"))
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model.keras"))

print("Model saved as .h5 and .keras ✅")