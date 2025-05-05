"""DTMF Classification Module.

This module provides functionality processing, and classifying
Dual-Tone Multi-Frequency (DTMF) signals. It includes tools for measuring model
latency, and building DTMF classifier models using
TensorFlow and ONNX.

Functions:
    measure_latency: Measure the latency for a model for a given tensor shape.
    build_dtmf_classifier_model: Build a DTMF classifier model using a sequential Keras model.

Classes:
    DtmfClassifierOnnx: A simple wrapper class to create an ONNX runtime session for a DTMF classification model.
"""

import logging

# from collections.abc import Callable
import tensorflow as tf
from tensorflow.keras import layers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dtmf_classifier_model(
    input_shape: tuple[int | None, int], num_classes: int
) -> tf.keras.Sequential:
    """Build a DTMF classifier model using a sequential Keras model.

    This function constructs a convolutional neural network (CNN) model for
    classifying DTMF (Dual-tone multi-frequency) signals. The model consists
    of several convolutional layers followed by max pooling and upsampling layers.

    Args:
        input_shape (tuple[int, int]): The shape of the input data (e.g., (length, channels)).
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Sequential: A Keras Sequential model configured for DTMF classification.
    """
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(
            32,
            kernel_size=32,
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling1D(padding="same"),
        layers.Conv1D(32, kernel_size=32, activation="relu", padding="same"),
        layers.MaxPooling1D(padding="same"),
        layers.Conv1D(32, kernel_size=32, activation="relu", padding="same"),
        layers.MaxPooling1D(padding="same"),
        layers.Conv1D(32, kernel_size=32, activation="relu", padding="same"),
        layers.MaxPooling1D(padding="same"),
        layers.Conv1D(32, kernel_size=16, activation="relu", padding="same"),
        layers.UpSampling1D(size=2),
        layers.Conv1D(32, kernel_size=16, activation="relu", padding="same"),
        layers.UpSampling1D(size=2),
        layers.Conv1D(32, kernel_size=16, activation="relu", padding="same"),
        layers.UpSampling1D(size=2),
        layers.Conv1D(32, kernel_size=16, activation="relu", padding="same"),
        layers.UpSampling1D(size=2),
        layers.Conv1D(32, kernel_size=16, activation="relu", padding="same"),
        layers.Conv1D(64, kernel_size=1, activation="relu"),
        layers.Conv1D(num_classes, kernel_size=1, activation="softmax"),
    ])
