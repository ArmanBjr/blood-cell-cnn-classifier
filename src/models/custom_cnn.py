"""Custom CNN architecture built from scratch."""

from tensorflow.keras import layers, models, Input


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4, dense_units=256, dropout_rate=0.5):
    """
    4-block CNN: [Conv→BN→ReLU→MaxPool] × 3 + [Conv→BN→ReLU] × 1
    → GAP → Dense → Dropout → Softmax
    """
    inputs = Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="custom_cnn")
