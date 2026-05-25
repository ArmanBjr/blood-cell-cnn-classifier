"""Transfer learning model builders (VGG16, ResNet50V2, MobileNetV2, EfficientNetB3)."""

import tensorflow as tf
from tensorflow.keras import layers, models, Input

_BACKBONES = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "VGG16": tf.keras.applications.VGG16,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
}


def build_transfer_model(
    model_name: str,
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 4,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """Stage-1 model: frozen base + trainable head."""
    backbone_cls = _BACKBONES[model_name]
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name=f"{model_name}_transfer")


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze the top n_layers of the base model for fine-tuning (stage 2)."""
    base = model.layers[1]  # base is always the second layer
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
