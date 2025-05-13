import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart

from techdays25.dtmf_generation import DtmfGenerator

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER: trt.Logger = trt.Logger(trt.Logger.VERBOSE)

DEBUG = False


def get_gpu_type() -> str:
    """Get the type of GPU available on the system.

    This function checks if a CUDA-capable GPU is available using PyTorch.
    If a GPU is available, it returns the name of the GPU in lowercase with spaces replaced by underscores.
    If no GPU is available, it returns 'cpu'.

    Returns:
        str: The type of GPU available or 'cpu' if no GPU is available.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return "_".join(torch.cuda.get_device_name(0).lower().split(" "))


def check_cuda_err(err: cuda.CUresult | cudart.cudaError_t) -> None:
    """Check for CUDA errors and raise an exception if an error is found.

    Args:
        err (Union[cuda.CUresult, cudart.cudaError_t]): The CUDA error code.

    Raises:
        RuntimeError: If a CUDA error is detected.
    """
    if isinstance(err, cuda.CUresult) and err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Cuda Error: {err}")
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Runtime Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def cuda_call(call: Any) -> Any:
    """Make a CUDA call and check for errors.

    Args:
        call (Any): The CUDA call to make.

    Returns:
        Any: The result of the CUDA call.
    """
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray) -> None:
    """Copy data from host to device.

    Args:
        device_ptr (int): The device pointer.
        host_arr (np.ndarray): The host array.
    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
    )


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int) -> None:
    """Copy data from device to host.

    Args:
        host_arr (np.ndarray): The host array.
        device_ptr (int): The device pointer.
    """
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    )


class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for our DTMF classifier model."""

    def __init__(
        self, training_data: str, cache_file: str, batch_size: int = 16
    ) -> None:
        """Initialize the MNISTEntropyCalibrator.

        Args:
            training_data (str): The path to the training data.
            cache_file (str): The path to the cache file.
            batch_size (int, optional): The batch size. Defaults to 16.
        """
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        # self.data = 2 * np.random.rand(32*batch_size, 2**12, 1).astype(np.float32) - 1.0
        self.data = dtmf_gen.generate_dataset(
            n_samples=32 * batch_size, t_length=2**12, with_labels=None
        ).astype(np.float32)

        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        # self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
        n_bytes = self.data[0].nbytes * self.batch_size

        self.device_input = cuda_call(cudart.cudaMalloc(n_bytes))

    def get_batch_size(self) -> int:
        """Get the batch size.

        Returns:
            int: The batch size.
        """
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names: list[str]) -> list[int] | None:
        """Get a batch of data.

        Args:
            names (List[str]): The names of the engine bindings.

        Returns:
            Optional[List[int]]: The device input pointer, or None if there is no more data.
        """
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print_dbg(
                f"Calibrating batch {current_batch}, containing {self.batch_size} images"
            )

        batch = self.data[
            self.current_index : self.current_index + self.batch_size
        ].ravel()
        # cuda.memcpy_htod(self.device_input, batch)
        # memcpy_host_to_device(self.device_input, batch)
        memcpy_host_to_device(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        """Read the calibration cache.

        Returns:
            Optional[bytes]: The calibration cache, or None if it does not exist.
        """
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if Path.exists(self.cache_file):
            return Path(self.cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write the calibration cache.

        Args:
            cache (bytes): The calibration cache.
        """
        return  # for now
        Path(self.cache_file).write_bytes(cache)


def build_engine_onnx(model_file: str, trt_engine_path: str, precision: str) -> None:
    """Builds a TensorRT engine from an ONNX model file.

    Args:
        model_file (str): The path to the ONNX model file.
        trt_engine_path (str): The path to save the TensorRT engine.
        precision (str): The precision mode to use ('fp16', 'int8', 'mixed').

    Returns:
        Optional[None]: Returns None if the engine creation fails.
    """
    seq_len: int = 2**12
    max_batch_size: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    calibration_batch_size: int = 16
    max_memory_pool_size: int = 8 * (1 << 30)  # 8GB

    builder: trt.Builder = trt.Builder(TRT_LOGGER)
    network: trt.INetworkDefinition = builder.create_network(0)
    config: trt.IBuilderConfig = builder.create_builder_config()
    parser: trt.OnnxParser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_memory_pool_size)

    # Load the Onnx model and parse it in order to populate the TensorRT network.
    if not parser.parse(Path(model_file).read_bytes()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return

    # different batch sizes: index 0-9
    for b in max_batch_size:
        profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        profile.set_shape(
            "input", [b // 2 + 1, seq_len, 1], [b, seq_len, 1], [b, seq_len, 1]
        )
        config.add_optimization_profile(profile)

    # add one profile for longer sequences: index 10
    profile: trt.IOptimizationProfile = builder.create_optimization_profile()
    profile.set_shape("input", [128, 2**14, 1], [128, 2**14, 1], [128, 2**14, 1])
    config.add_optimization_profile(profile)

    # TODO: later add one profile for rather long sequences with batch_size 1:
    # profile: trt.IOptimizationProfile = builder.create_optimization_profile()
    # profile.set_shape("input", [1, 2**20, 1], [1, 2**20, 1], [1, 2**20, 1])
    # config.add_optimization_profile(profile)

    if precision in ["fp16", "int8", "mixed"]:
        if not builder.platform_has_fast_fp16:
            print("FP16 is not supported natively on this platform/device")
        config.set_flag(trt.BuilderFlag.FP16)
    if precision in ["int8", "mixed"]:
        if not builder.platform_has_fast_int8:
            print("INT8 is not supported natively on this platform/device")
        config.set_flag(trt.BuilderFlag.INT8)
        # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        calib = MNISTEntropyCalibrator(
            "", cache_file="cache.file", batch_size=calibration_batch_size
        )
        config.int8_calibrator = calib

        calib_profile: trt.IOptimizationProfile = builder.create_optimization_profile()
        calib_profile.set_shape(
            "input",
            [calibration_batch_size, seq_len, 1],
            [calibration_batch_size, seq_len, 1],
            [calibration_batch_size, seq_len, 1],
        )
        config.set_calibration_profile(calib_profile)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        print("int 8 model")

    engine_bytes: bytes | None = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("Failed to create the TensorRT engine")
        return
    trt.Runtime(TRT_LOGGER)

    # Save the engine to a file
    Path(trt_engine_path).write_bytes(engine_bytes)

    print(f"TensorRT engine saved to {trt_engine_path}")


# Example Usage
# precision: str = "int8"
# onnx_path: str = "dtmf_classifier.onnx"
# trt_path: str = "dtmf_classifier_" + precision + "_" + get_gpu_type() + ".trt"
# build_engine_onnx(onnx_path, trt_path, precision=precision)
# from google.colab import files
# files.download(trt_path)


def print_dbg(*x: Any) -> None:
    """Print debug information if DEBUG is set to True.

    Args:
        *x (Any): The information to print.
    """
    if DEBUG:
        print(x)


class CustomProfiler(trt.IProfiler):
    """Custom Profiler for logging layer-wise latency."""

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layers = {}

    def report_layer_time(self, layer_name, ms):
        if layer_name not in self.layers:
            self.layers[layer_name] = []

        self.layers[layer_name].append(ms)


class TensorRTInfer:
    """Implements inference for the TensorRT engine."""

    def __init__(self, engine_path: str) -> None:
        """Initialize the TensorRTInfer class.

        Args:
            engine_path (str): The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        # with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
        #    assert runtime
        #    self.engine = runtime.deserialize_cuda_engine(f.read())
        runtime = trt.Runtime(self.logger)
        assert runtime
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())

        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Some Infos about the engine
        print_dbg("num optimization profiles:", self.engine.num_optimization_profiles)
        print_dbg("num io tensors:", self.engine.num_io_tensors)

        # Create CUDA stream for asynchronous tasks
        _, self.stream = cudart.cudaStreamCreate()

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for prof_idx in range(self.engine.num_optimization_profiles):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                is_input = False
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                shape = self.engine.get_tensor_shape(name)
                if is_input and shape[0] < 0:
                    assert self.engine.num_optimization_profiles >= 1
                    profile_shape = self.engine.get_tensor_profile_shape(name, prof_idx)
                    print_dbg("profile_shape", name, profile_shape)
                    assert len(profile_shape) == 3  # min,opt,max

                    # Set the *max* profile as binding shape
                    self.switch_profile(prof_idx)
                    self.context.set_input_shape(name, profile_shape[2])
                    shape = self.context.get_tensor_shape(name)

                if not is_input:
                    shape = self.context.get_tensor_shape(name)
                    print_dbg("shape for output:", name, shape)

                if is_input:
                    self.batch_size = shape[0]
                size = dtype.itemsize
                for s in shape:
                    size *= s
                allocation = cuda_call(cudart.cudaMalloc(size))
                host_allocation = None if is_input else np.zeros(shape, dtype)
                binding = {
                    "index": i,
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "allocation": allocation,
                    "host_allocation": host_allocation,
                }
                self.allocations.append(allocation)
                if is_input:
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)
                print_dbg(
                    "{} '{}' with shape {} and dtype {}".format(
                        "Input" if is_input else "Output",
                        binding["name"],
                        binding["shape"],
                        binding["dtype"],
                    )
                )
            print_dbg()

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def __del__(self):
        for allocation in self.allocations:
            cuda_call(cudart.cudaFree(allocation))

    def enable_profiling(self, profiler: trt.IProfiler = None) -> None:
        """Enable TensorRT profiling.

        TensorRT will report time spent on each layer in stdout for each forward run.
        """
        if not self.context.profiler:
            self.context.profiler = CustomProfiler() if profiler is None else profiler

    def input_spec(self) -> list[tuple[list[int], np.dtype]]:
        """Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        Returns:
            list[tuple[list[int], np.dtype]]: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        # Old: return self.inputs[0]["shape"], self.inputs[0]["dtype"]
        specs = []
        for i in self.inputs:
            specs.append((i["shape"], i["dtype"]))
        return specs

    def output_spec(self) -> list[tuple[list[int], np.dtype]]:
        """Get the specs for the output tensors of the network. Useful to prepare memory allocations.

        Returns:
            list[tuple[list[int], np.dtype]]: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o["shape"], o["dtype"]))
        return specs

    def switch_profile(self, idx: int) -> None:
        """Switch to a different optimization profile.

        Args:
            idx (int): The index of the optimization profile to switch to.
        """
        self.context.set_optimization_profile_async(idx, self.stream)

    def infer(self, batch: np.ndarray) -> list[np.ndarray]:
        """Execute inference on a batch of images.

        Args:
            batch (np.ndarray): A numpy array holding the image batch.

        Returns:
            List[np.ndarray]: A list of outputs as numpy arrays.
        """
        # If the optimization profile does not match, change it here:
        # In our setup the opt. profiles are selected in a way that the
        # optimal batch sizes are powers of 2. In practice, one would not do
        # it in this way:
        # TODO: If the profile does not fit in the range, find the profile with
        # the closest optimal settings...
        # A lot is still hard-coded here...
        if batch.shape == (128, 2**14, 1):
            # For the moment, hard coded
            print_dbg("Batch shape", batch.shape)
            print_dbg("Changing to profile", 10)
            self.switch_profile(10)

            if self.context.get_tensor_shape("input") != batch.shape:
                print_dbg("Changing batch shape for inference to", batch.shape)
                self.context.set_input_shape("input", batch.shape)

        elif True:
            expected_profile = int(np.log2(batch.shape[0]))
            if self.context.active_optimization_profile != expected_profile:
                print_dbg("Changing to profile", expected_profile)
                self.switch_profile(expected_profile)

            if self.context.get_tensor_shape("input") != batch.shape:
                print_dbg("Changing batch size for inference!")

                # Adapt the input shape:
                self.context.set_input_shape("input", batch.shape)
        print_dbg(
            "self.engine.get_tensor_shape(input)", self.engine.get_tensor_shape("input")
        )
        print_dbg(
            "self.context.get_tensor_shape(input)",
            self.context.get_tensor_shape("input"),
        )
        print_dbg(
            "self.context.get_tensor_shape(output)",
            self.context.get_tensor_shape("output"),
        )
        print_dbg()

        o_idx = self.context.active_optimization_profile
        print_dbg("Active output index (opt. profile)", o_idx)

        # Copy I/O and Execute
        memcpy_host_to_device(self.inputs[o_idx]["allocation"], batch)

        self.context.execute_v2(self.allocations)
        memcpy_device_to_host(
            self.outputs[o_idx]["host_allocation"], self.outputs[o_idx]["allocation"]
        )

        return [self.outputs[o_idx]["host_allocation"]]


# TODO: remove the following:
if False:
    dtmf_gen = DtmfGenerator(
        dur_key=(0.04, 0.05),
        dur_pause=(0.03, 0.04),
        noise_factor=(0.0, 60.0),
        noise_freq_range=(0.0, 20000.0),
    )

    trt_path = "techdays25/assets/lab2/dtmf_classifier_int8_tesla_t4.trt"
    seq_len = 2**12
    trt_infer = TensorRTInfer(trt_path)
    # trt_infer.enable_profiling() # Use only for debugging/analysis purposes, since it slows down inference

    print("Starting inference")
    if True:
        # trt_infer.switch_profile(6)
        spec = trt_infer.input_spec()
        print("spec", spec)
        # batch = my_dialed_sequence_signal.reshape(1, -1, 1).astype(np.float32)
        X, Y = dtmf_gen.generate_dataset(n_samples=256, t_length=seq_len)
        o = trt_infer.infer(X.astype(np.float32))[0][: X.shape[0]]
        # o = keras_model.predict(X.astype(np.float32), verbose=0)
        print("o.shape", o.shape)

        thresholded = (o > 0.5).astype(int)
        print((thresholded == Y).sum() / Y.size)
        for iidx in range(X.shape[0]):
            predicted_key_sequence = dtmf_gen.decode_prediction(o[iidx])
            original_key_sequence = dtmf_gen.decode_prediction(Y[iidx])
            if predicted_key_sequence != original_key_sequence:
                print("predicted_key_sequence", predicted_key_sequence)
                print("original_key_sequence", original_key_sequence)
            else:
                pass
                # print("OK", predicted_key_sequence)
        print("Done!")
    else:
        print("No input provided, running in benchmark mode")
        trt_infer.switch_profile(0)
        spec = trt_infer.input_spec()

        spec = (512, 4096, 1), np.float32

        rng = np.random.default_rng()
        batch = rng.random(spec[0]).astype(spec[1])
        # batch = np.random.rand(*spec[0]).astype(spec[1])

        print("batch.shape", batch.shape)
        print("batch.dtype", batch.dtype)
        print("min/max/mean", batch.min(), batch.max(), batch.mean())
        iterations = 100
        times = []
        for i in range(20):  # GPU warmup iterations
            trt_infer.infer(batch)
        for i in range(iterations):
            start = time.time()
            o = trt_infer.infer(batch)
            times.append(time.time() - start)
            print(f"Iteration {i + 1} / {iterations}", end="\r")
        print("Benchmark results include time for H2D and D2H memory copies")
        print(f"Average Latency: {1000 * np.average(times):.3f} ms")
        print(f"Average Throughput: {trt_infer.batch_size / np.average(times):.1f} ips")

    print()
    print("Finished Processing")
