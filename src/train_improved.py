import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Get the absolute path of the current script's directory for saving the model later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Dataset ---
# Download the 'cats_vs_dogs' images and split them into training (80%) and validation (20%) sets
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],  
    with_info=True,
    as_supervised=True,
)

# --- Improved Preprocessing with Data Augmentation ---
IMG_SIZE = 150

def preprocess(image, label):
    """Basic preprocessing for validation data"""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

def augment_and_preprocess(image, label):
    """Enhanced preprocessing with data augmentation for training"""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Data augmentation to make model more robust
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    
    # Clip values to valid range after augmentation
    image = tf.clip_by_value(image, 0.0, 255.0)
    
    # Normalize
    image = image / 255.0
    return image, label

# Apply augmentation to training data, basic preprocessing to validation
train_ds = ds_train.map(augment_and_preprocess).shuffle(2000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

print("Dataset with augmentation ready ✅")

# --- Improved CNN Model Architecture ---
model = models.Sequential([
    # Input layer
    layers.Input(shape=(150, 150, 3)),
    
    # First convolutional block
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),
    
    # Fourth convolutional block for deeper features
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile with improved settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Improved model architecture ready ✅")
model.summary()

# --- Training with Early Stopping and Model Checkpoint ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Train for more epochs with callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # More epochs, but early stopping will prevent overfitting
    callbacks=callbacks
)

# --- Save the Improved Model ---
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model_improved.h5"))
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model_improved.keras"))

print("Improved model saved ✅")

# Print final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

# Made with Bob
