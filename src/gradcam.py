"""Grad-CAM implementation for CNN visualization."""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for the predicted class.
    img_array: shape (1, H, W, 3), values in [0, 1].
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superimpose Grad-CAM heatmap on the original image."""
    img = (img_array[0] * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def plot_gradcam(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str,
                 class_names: list, true_label: int = None, save_path: str = None):
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay = overlay_gradcam(img_array, heatmap)

    pred_proba = model.predict(img_array, verbose=0)[0]
    pred_class = np.argmax(pred_proba)
    confidence = pred_proba[pred_class]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array[0])
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    title = f"Pred: {class_names[pred_class]} ({confidence:.2%})"
    if true_label is not None:
        title += f"\nTrue: {class_names[true_label]}"
    axes[2].set_title(title, color="green" if true_label == pred_class else "red")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
