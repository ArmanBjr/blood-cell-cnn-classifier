"""Data loading pipelines using Keras ImageDataGenerator."""

import yaml
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_generators(config: dict, image_size: tuple = None):
    """
    Returns (train_gen, val_gen, test_gen, test_simple_gen).
    Validation split is carved from TRAIN directory.
    """
    aug = config["augmentation"]
    img_size = image_size or tuple(config["preprocessing"]["image_size_default"])
    batch = config["preprocessing"]["batch_size"]
    val_split = config["preprocessing"]["val_split"]
    seed = config["preprocessing"]["seed"]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        zoom_range=aug["zoom_range"],
        horizontal_flip=aug["horizontal_flip"],
        vertical_flip=aug["vertical_flip"],
        fill_mode=aug["fill_mode"],
        validation_split=val_split,
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        config["data"]["train_dir"],
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        subset="training",
        seed=seed,
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        config["data"]["train_dir"],
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        subset="validation",
        seed=seed,
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        config["data"]["test_dir"],
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )

    test_simple_gen = val_test_datagen.flow_from_directory(
        config["data"]["test_simple_dir"],
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen, test_simple_gen
