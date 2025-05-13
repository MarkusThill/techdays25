"""Various ONNX helper functions."""

import logging
import sys
import time

# from collections.abc import Callable
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import output

    output.enable_custom_widget_manager()


def netron_visualize(model_path: Path | str):
    """Visualize a model using Netron.

    This function visualizes a model using Netron. It currently supports visualization
    only in Google Colab. If the code is not running in Google Colab, it raises a
    NotImplementedError.

    Args:
        model_path (Path | str): The path to the model file to be visualized. This can be a
                                 Path object or a string representing the file path.

    Raises:
        NotImplementedError: If the function is called outside of Google Colab.
    """
    if IN_COLAB:
        import netron
        import portpicker
        from google.colab import output

        port = portpicker.pick_unused_port()

        # Read the model file and start the netron browser.
        with output.temporary():
            netron.start(model_path, port, browse=False)

        output.serve_kernel_port_as_iframe(port, height="500")
    else:
        raise NotImplementedError(
            "Unfortunately, currently only Netron visualizations in Google Colab is supported!"
        )


class OnnxModel:
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

        self.available_providers = ort.get_available_providers()
        logger.info("Available providers: %s", self.available_providers)

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
        self.execution_providers = session.get_providers()
        self.session = session

        # Get the name of the input node
        self.input_name = session.get_inputs()[0].name
        self.input_shape = session.get_inputs()[0].shape
        self.input_dtype = session.get_inputs()[0].type

        # Get the name of the output node
        self.output_name = session.get_outputs()[0].name
        self.output_shape = session.get_outputs()[0].shape
        self.output_dtype = session.get_outputs()[0].type

        # Get the name of the output node
        logger.info("Output node name: %s", self.output_name)
        logger.info("Execution providers: %s", self.execution_providers)
        logger.info("Input node name: %s", self.input_name)

    def __repr__(self) -> str:
        """Retreive some general infos of the object as a string.

        Returns:
            str: General information regarding the loaded onnx model and the running session.
        """
        return f"Available execution providers:{self.available_providers}\nExecution providers: {self.execution_providers}\nInput name: {self.input_name}\nInput shape: {self.input_shape}\nInput type:{self.input_dtype}\nOutput name:{self.output_name}\nOutput shape:{self.output_shape}\nOutput type:{self.output_dtype}"

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Run prediction on the input signal.

        Args:
            signal (np.ndarray): The input signal to classify.

        Returns:
            np.ndarray: The classification result.
        """
        logger.debug("Running prediction on input signal with shape: %s", signal.shape)
        start = time.perf_counter()
        result = self.session.run([self.output_name], {self.input_name: signal})[0]
        end = time.perf_counter()
        logger.debug("Inference time: %.6f seconds", end - start)
        return result


def measure_latency_(
    f_predict: Callable[[np.ndarray], np.ndarray],
    tensor_shape: tuple[int, int, int] = (64, 2**11, 1),
    n_runs: int = 100,
    n_warmup: int = 10,
    verbose: bool = True,
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
        verbose: (bool, optional): Plots the progress of the measurements, if desired

    Returns:
        list[float]: A list of timings for each run, representing the latency in seconds.
    """
    rng = np.random.default_rng()

    for _ in range(n_warmup):
        input_tensor = rng.uniform(low=-1, high=1, size=tensor_shape).astype(np.float32)
        _ = f_predict(input_tensor)

    timings = []
    modd = max(1, n_runs // 100)
    for r_idx in range(n_runs):
        input_tensor = rng.uniform(low=-1, high=1, size=tensor_shape).astype(np.float32)
        start = time.perf_counter()
        _ = f_predict(input_tensor)
        end = time.perf_counter()
        timings.append(end - start)
        if verbose and r_idx % modd == 0:
            print(".", end="")  # noqa: T201

    return timings


def benchmark_models_on_batch_size(
    model_dict: dict[str, Callable[[np.ndarray], np.ndarray]],
    input_shape: tuple[int, ...],
    batch_sizes: list[int] = [2**i for i in range(5)],
    n_runs: int = 100,
    n_warmup: int = 20,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Benchmark multiple models on different batch sizes.

    This function benchmarks the latency of multiple models' prediction functions
    for different batch sizes. It measures the time it takes for each model to
    process input tensors of specified shapes and returns the results in a dictionary
    of pandas DataFrames.

    Args:
        model_dict (dict[str, Callable[[np.ndarray], np.ndarray]]): A dictionary where the keys are model names (strings)
            and the values are prediction functions that take a numpy ndarray as input and return a numpy ndarray as output.
        input_shape (tuple[int, ...]): The shape of the input tensor (excluding the batch size).
        batch_sizes (list[int], optional): A list of batch sizes to test. Defaults to [2**i for i in range(5)].
        n_runs (int, optional): The number of runs to measure latency. Defaults to 100.
        n_warmup (int, optional): The number of warmup runs before measuring latency. Defaults to 20.
        verbose (bool, optional): If True, prints progress information. Defaults to True.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where the keys are model names (strings) and the values are pandas DataFrames
            containing the timing results for each batch size.
    """
    results = {}
    for model_name, f_predict in model_dict.items():
        if verbose:
            print(model_name)  # noqa: T201
        timings = {}
        for batch_size in batch_sizes:
            if verbose:
                print(f"  b={batch_size}", end="")  # noqa: T201
            tensor_shape = batch_size, *input_shape  # (batch_size, signal_length, 1)
            batch_timings = measure_latency_(
                f_predict,
                tensor_shape=tensor_shape,
                n_runs=n_runs,
                n_warmup=n_warmup,
                verbose=verbose,
            )
            timings[batch_size] = batch_timings
            print()  # noqa: T201

        df = pd.DataFrame(timings)
        results[model_name] = df
    return results
