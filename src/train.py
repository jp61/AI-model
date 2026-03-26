import os
import tensorflow as tf
import tensorflow_datasets as tfds

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load dataset ---
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],  # 80% train, 20% validation
    with_info=True,
    as_supervised=True,  # dáta ako (image, label)
)

# --- Preprocessing ---
IMG_SIZE = 150

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # normalize
    return image, label

train_ds = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

print("Dataset pripravený ✅")


from tensorflow.keras import layers, models

# --- CNN Model ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binárna klasifikácia
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model pripravený ✅")


# --- Tréning ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


model.build(input_shape=(None, 150, 150, 3))
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model.h5"))
model.save(os.path.join(SCRIPT_DIR, "cats_dogs_model.keras"))
print("Model uložený ako .h5 aj .keras ✅")