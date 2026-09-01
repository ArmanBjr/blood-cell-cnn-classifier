import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def test_config_loads():
    from src.data_loader import load_config

    config = load_config(str(ROOT / "configs" / "config.yaml"))
    assert config["data"]["num_classes"] == 4
    assert len(config["data"]["classes"]) == 4
    assert config["preprocessing"]["batch_size"] > 0


def test_custom_cnn_builds_and_predicts():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from src.models.custom_cnn import build_custom_cnn

    tf.keras.backend.clear_session()
    model = build_custom_cnn(input_shape=(64, 64, 3), num_classes=4)
    batch = np.zeros((2, 64, 64, 3), dtype=np.float32)
    output = model(batch, training=False)

    assert output.shape == (2, 4)
    assert np.isclose(output.numpy().sum(axis=1), 1.0).all()
