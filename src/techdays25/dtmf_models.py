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
import time

# from collections.abc import Callable
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import layers


def measure_latency(
    f_predict: Callable[[np.ndarray], np.ndarray],
    tensor_shape: tuple[int, int, int] = (64, 2**11, 1),
    n_runs: int = 100,
    n_warmup: int = 10,
) -> list[float]:
    """Measure the latency for a model for a given tensor shape.

    This function measures the time it takes for a model's prediction function
    to process an input tensor of a specified shape. It performs a number of
    warmup runs before measuring the latency to ensure the model is fully
    initialized and any initial overhead is accounted for.

    Args:
        f_predict (Callable[[np.ndarray], np.ndarray]): The prediction function of the model.
            This function should take a numpy ndarray as input and return a numpy ndarray as output.
        tensor_shape (tuple[int, int, int], optional): The shape of the input tensor. Defaults to (64, 2**11, 1).
        n_runs (int, optional): The number of runs to measure latency. Defaults to 100.
        n_warmup (int, optional): The number of warmup runs before measuring latency. Defaults to 10.

    Returns:
        list[float]: A list of timings for each run, representing the latency in seconds.
    """
    rng = np.random.default_rng()

    for _ in range(n_warmup):
        input_tensor = rng.uniform(low=-1, high=1, size=tensor_shape).astype(np.float32)
        _ = f_predict(input_tensor)

    timings = []
    for _ in range(n_runs):
        input_tensor = rng.uniform(low=-1, high=1, size=tensor_shape).astype(np.float32)
        start = time.perf_counter()
        _ = f_predict(input_tensor)
        end = time.perf_counter()
        timings.append(end - start)

    return timings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DtmfClassifierOnnx:
    """A simple wrapper class to create an ONNX runtime session for a DTMF classificaiton model."""

    def __init__(
        self, onnx_model_path: str | Path, provider: str = "CUDAExecutionProvider"
    ):
        """Initialize the DtmfClassifierOnnx with the given ONNX model path and execution provider.

        Args:
            onnx_model_path (str | Path): The path to the ONNX model file.
            provider (str, optional): The execution provider to use for inference. Defaults to "CUDAExecutionProvider".
        """
        logger.info(
            "Initializing DtmfClassifierOnnx with model path: %s and provider: %s",
            onnx_model_path,
            provider,
        )

        available_providers = ort.get_available_providers()
        logger.info("Available providers: %s", available_providers)

        # Create session options if needed
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Specify execution provider (usually, as CUDA/GPU)
        session = ort.InferenceSession(
            onnx_model_path, providers=[provider], sess_options=sess_options
        )

        # Check which providers are being used
        execution_providers = session.get_providers()
        logger.info("Execution providers: %s", execution_providers)

        self.session = session

        # Get the name of the input node
        self.input_name = session.get_inputs()[0].name
        logger.info("Input node name: %s", self.input_name)

        # Get the name of the output node
        self.output_name = session.get_outputs()[0].name
        logger.info("Output node name: %s", self.output_name)

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run prediction on the input signal.

        Args:
            signal (np.ndarray): The input signal to classify.

        Returns:
            np.ndarray: The classification result.
        """
        logger.info("Running prediction on input signal with shape: %s", signal.shape)
        start = time.perf_counter()
        result = self.session.run([self.output_name], {self.input_name: signal})[0]
        end = time.perf_counter()
        logger.info("Inference time: %.6f seconds", end - start)
        return result


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
