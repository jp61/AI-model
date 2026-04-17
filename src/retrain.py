import os
import tensorflow as tf
import numpy as np

# Get the absolute path of the current script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Phase 1: Custom Dataset Extension for Pre-trained Keras Models ---
# Load the existing model and prepare new data
model_path = os.path.join(SCRIPT_DIR, "cats_dogs_model.keras")
model = tf.keras.models.load_model(model_path)
print("Model loaded from cats_dogs_model.keras ✅")

# --- Prepare the dataset from images folder ---
images_dir = os.path.join(SCRIPT_DIR, "images")

# Get list of image files
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create labels based on filename (assuming 'cat' or 'dog' in filename)
file_paths = []
labels = []
for f in image_files:
    file_paths.append(os.path.join(images_dir, f))
    if 'cat' in f.lower():
        labels.append(0)  # 0 for cat
    elif 'dog' in f.lower():
        labels.append(1)  # 1 for dog
    else:
        print(f"Warning: Could not determine label for {f}, skipping")
        continue

# Convert to numpy arrays
file_paths = np.array(file_paths)
labels = np.array(labels)

# Create TensorFlow dataset
def load_and_preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (150, 150))
    image = image / 255.0
    return image, label

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.map(load_and_preprocess_image)
dataset = dataset.shuffle(len(file_paths)).batch(32).prefetch(tf.data.AUTOTUNE)

print(f"Dataset prepared with {len(file_paths)} images ✅")

# --- Phase 2: Refining Vision Models: A Transfer Learning Implementation ---
# Unfreeze the last few layers and recompile for optimization
for layer in model.layers[:-2]:  # Freeze all but the last 2 layers
    layer.trainable = False

# Recompile the model with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model ready for fine-tuning ✅")

# --- Phase 3: Incremental Learning: Fine-Tuning Convolutional Networks for Pet Classification ---
# Execute the training process and save the updated knowledge
history = model.fit(
    dataset,
    epochs=5  # Fewer epochs for fine-tuning
)

# --- Save the fine-tuned model ---
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model_fine_tuned.h5"))
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model_fine_tuned.keras"))

print("Model saved as .h5 and .keras ✅")