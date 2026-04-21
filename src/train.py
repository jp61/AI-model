"""Train the cats-vs-dogs CNN.

Flags let you turn each regularization technique on/off or tune its strength.
Run `python src/train.py --help` for the full list, or see `docs/regularization.md`
for the motivation behind each flag and how to use them together.
"""

import argparse
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, regularizers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "model")
IMG_SIZE = 150


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30,
                   help="Upper bound on training epochs (EarlyStopping may cut earlier).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable/disable RandomFlip+RandomRotation+RandomZoom.")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Dropout rate before Dense(512). Set to 0 to disable.")
    p.add_argument("--l2", type=float, default=1e-4,
                   help="L2 weight-decay strength on Conv2D/Dense kernels. Set to 0 to disable.")
    p.add_argument("--patience", type=int, default=3,
                   help="EarlyStopping patience on val_loss. Set to 0 to disable.")
    return p.parse_args()


def build_augmenter():
    # Applied in the tf.data pipeline, NOT inside the model. TF.js cannot load
    # RandomFlip/RandomRotation/RandomZoom layer types, so baking augmentation
    # into the saved model breaks browser inference.
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmenter")


def conv_block(filters, reg):
    # Conv -> BN -> ReLU -> Pool. BN before activation is the Ioffe & Szegedy 2015
    # convention. All layer types here are registered in @tensorflow/tfjs.
    return [
        layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=reg),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),
    ]


def build_model(args):
    """Four-block CNN with GlobalAveragePooling head.

    Replaces the previous Flatten -> Dense(512) transition (~16M params) with
    GlobalAveragePooling2D -> Dense(1) (~100k params). The old capacity
    bottleneck was the root cause of overfitting — no amount of augmentation
    or weight decay could rescue a head with 1000 params per training image.
    Lin et al. 2013 (Network In Network) first proposed GAP as a structural
    regularizer; it's now standard in ResNet / MobileNet / EfficientNet.
    """
    reg = regularizers.l2(args.l2) if args.l2 > 0 else None

    model_layers = [layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))]
    for filters in (32, 64, 128, 256):
        model_layers += conv_block(filters, reg)

    model_layers.append(layers.GlobalAveragePooling2D())

    if args.dropout > 0:
        model_layers.append(layers.Dropout(args.dropout))

    model_layers.append(layers.Dense(1, activation='sigmoid', kernel_regularizer=reg))

    return models.Sequential(model_layers)


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Training config:")
    print(f"  epochs          = {args.epochs}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  augmentation    = {args.augment}")
    print(f"  dropout         = {args.dropout}  ({'on' if args.dropout > 0 else 'off'})")
    print(f"  l2              = {args.l2}  ({'on' if args.l2 > 0 else 'off'})")
    print(f"  early-stop pat. = {args.patience}  ({'on' if args.patience > 0 else 'off'})")

    (ds_train, ds_val), _ = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
    )

    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        return image, label

    train_ds = ds_train.map(preprocess).shuffle(1000).batch(args.batch_size)
    val_ds = ds_val.map(preprocess).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.augment:
        augmenter = build_augmenter()
        train_ds = train_ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(args)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = []
    if args.patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience, restore_best_weights=True,
        ))

    # verbose=2 prints one line per epoch instead of a live progress bar.
    # Progress bars use \r and are token-churn when the log is tailed or read.
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=callbacks, verbose=2)

    model.save(os.path.join(MODEL_DIR, "cats_dogs_model.h5"))
    model.save(os.path.join(MODEL_DIR, "cats_dogs_model.keras"))
    print(f"Saved model to {MODEL_DIR}/ (.h5 + .keras)")


if __name__ == "__main__":
    main()
