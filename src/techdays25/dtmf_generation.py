"""DTMF Signal Generation Module.

This module provides functionality to generate Dual-Tone Multi-Frequency (DTMF) signals,
which are used in telecommunication signaling. The module includes a class `DtmfGenerator`
that can generate individual DTMF tones, sequences of tones, and datasets for DTMF signal
classification. It also supports adding noise to the generated signals and labeling the
signals with either frequency or key information.

Classes:
    DtmfGenerator: A class to generate DTMF signals with various configurations.

Example:
    from techdays25.dtmf_generation import DtmfGenerator

    # Initialize the DTMF generator
    dtmf_gen = DtmfGenerator()

    # Generate a single DTMF tone for the key '5'
    tone, freqs, key = dtmf_gen.get_key_tone('5')

    # Generate a sequence of DTMF tones for the key sequence '1234'
    sequence = dtmf_gen.get_tone_sequence('1234')

    # Generate a dataset of DTMF tone sequences for training or validation purposes
    X, Y = dtmf_gen.generate_dataset(n_samples=100)
"""

import random
from typing import ClassVar, Literal

import numpy as np


class DtmfGenerator:
    """A class to generate Dual-Tone Multi-Frequency (DTMF) signals.

    This class provides methods to generate DTMF tones, sequences of tones,
    and datasets for DTMF signal classification. It supports adding noise
    to the generated signals and labeling the signals with either frequency
    or key information.
    """

    key_matrix = np.array([
        "1",
        "2",
        "3",
        "A",
        "4",
        "5",
        "6",
        "B",
        "7",
        "8",
        "9",
        "C",
        "*",
        "0",
        "#",
        "D",
    ]).reshape(4, 4)
    freqz_cols: ClassVar[list[int]] = [1209, 1336, 1477, 1633]
    freqz_rows: ClassVar[list[int]] = [697, 770, 852, 941]

    FREQS: ClassVar[list[int]] = freqz_rows + freqz_cols
    FREQ_TO_IDX: ClassVar[dict[int, int]] = {f: i for i, f in enumerate(FREQS)}

    def __init__(
        self,
        sample_rate: int = 44100,
        dur_key: float | tuple[float, float] = (
            0.2,
            0.3,
        ),
        dur_pause: float | tuple[float, float] = (0.01, 0.1),
        noise_factor: float | tuple[float, float] = (10.0, 50.0),
        noise_freq_range: tuple[float, float] = (0.0, 2.0 * 10**4),
    ):
        """Initialize the DtmfGenerator with given parameters.

        Args:
            sample_rate (int): The sample rate of the signal. Defaults to 44100.

            dur_key (float | tuple[float, float]): Duration of each key press.
                If a float is provided, it specifies a fixed duration for all key presses.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each key press will be randomly chosen. Defaults to (0.2, 0.3).

            dur_pause (float | tuple[float, float]): Duration of each pause between key presses.
                If a float is provided, it specifies a fixed duration for all pauses.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each pause will be randomly chosen. Defaults to (0.01, 0.1).

            noise_factor (float | tuple[float, float]): The factor by which noise is scaled.
                If a float is provided, it specifies a fixed noise factor.
                If a tuple of two floats is provided, it specifies a range from which the noise factor
                will be randomly chosen. Defaults to (10.0, 50.0).

            noise_freq_range (tuple[float, float]): The frequency range for the noise in Hz.
                This specifies the minimum and maximum frequencies of the noise. Defaults to (0.0, 2.0 * 10**4).
        """
        self.sample_rate = sample_rate
        self.dur_key = dur_key
        self.dur_pause = dur_pause
        self.noise_factor = noise_factor
        self.noise_freq_range = noise_freq_range

    @staticmethod
    def get_num_keys() -> int:
        """Get the number of keys in the key matrix.

        The key matrix is a 4x4 array representing the DTMF keypad,
        containing 16 keys in total. This method returns the total
        number of keys in the key matrix.

        Returns:
            int: The total number of keys in the key matrix.
        """
        return DtmfGenerator.key_matrix.size

    @staticmethod
    def get_num_freqz() -> int:
        """Get the number of frequencies used in DTMF.

        The DTMF (Dual-tone multi-frequency) system uses a set of specific
        frequencies to represent keys on the keypad. This method returns
        the total number of unique frequencies used in the DTMF system.

        Returns:
            int: The total number of unique frequencies used in DTMF.
        """
        return len(DtmfGenerator.FREQS)

    def get_sample_rate(self) -> int:
        """Returns the specified sample rate in Hz for the audio signal generation.

        Returns:
            int: The sample rate in Hz, e.g., 44100
        """
        return self.sample_rate

    @staticmethod
    def get_key(key_idx: int | tuple[int, int] | np.ndarray) -> str:
        """Get the key from the key matrix based on the index.

        Args:
            key_idx (int | tuple[int, int] | np.ndarray): Index of the key.
                - If an int is provided, it is treated as a flat index.
                - If a tuple of two ints is provided, it is treated as a row and column index.
                - If a numpy array is provided, it is treated as a boolean mask or a set of indexes.

        Returns:
            str: The key at the given index.
        """
        if isinstance(key_idx, int):
            return DtmfGenerator.key_matrix.flatten()[key_idx]
        if isinstance(key_idx, tuple):
            return DtmfGenerator.key_matrix[*key_idx]

        # a numpy array containing a boolean mask or a set of indexes
        return DtmfGenerator.key_matrix.flatten()[key_idx]

    @staticmethod
    def fftnoise(ff: np.ndarray) -> np.ndarray:
        """Generates noise in the frequency domain and transforms it back to the time domain.

        Args:
            ff (np.ndarray): An array representing the frequency components of the signal.

        Returns:
            np.ndarray: A real-valued array representing the noise in the time domain.

        Notes:
            based on: https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
        """
        rng = np.random.default_rng()
        ff = np.array(ff, dtype="complex")
        Np = (len(ff) - 1) // 2
        phases = rng.random(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        ff[1 : Np + 1] *= phases
        ff[-1 : -1 - Np : -1] = np.conj(ff[1 : Np + 1])
        return np.fft.ifft(ff).real

    @staticmethod
    def band_limited_noise(
        min_freq: float, max_freq: float, samples: int, samplerate: int = 44100
    ) -> np.ndarray:
        """Generates band-limited noise within a specified frequency range.

        Args:
            min_freq (float): The minimum frequency of the noise band.
            max_freq (float): The maximum frequency of the noise band.
            samples (int): The number of samples to generate.
            samplerate (int, optional): The sample rate of the signal. Defaults to 44100.

        Returns:
            np.ndarray: An array containing the generated band-limited noise.
        """
        freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
        ff = np.zeros(samples)
        idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        ff[idx] = 1
        return DtmfGenerator.fftnoise(ff)
        # A = 10 #np.iinfo(np.int32).max * 10
        # return A * nn

    @staticmethod
    def run_length_encoding(
        arr: np.ndarray, min_run_length: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform run-length encoding on a numpy array.

        This function compresses the input array by encoding consecutive
        runs of the same value. It can also filter out runs shorter than
        a specified minimum length and then merge neighboring runs with the
        same value.

        Args:
            arr (np.ndarray): The input array to be encoded.
            min_run_length (int, optional): The minimum length of runs to be kept. Defaults to 1.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two numpy arrays, one containing the values of the runs
            and the other containing the lengths of the runs.
        """
        if len(arr) == 0:
            return np.array([]), np.array([])

        # Step 1: Basic run-length encoding
        change_indices = np.where(np.diff(arr) != 0)[0] + 1
        run_starts = np.insert(change_indices, 0, 0)
        run_lengths = np.diff(np.append(run_starts, len(arr)))
        run_values = arr[run_starts]

        # Step 2: Remove short runs
        mask = run_lengths >= min_run_length
        filtered_values = run_values[mask]
        filtered_lengths = run_lengths[mask]

        # Step 3: Merge neighboring runs with same value
        merged_values = []
        merged_lengths = []

        for i in range(len(filtered_values)):
            if merged_values and filtered_values[i] == merged_values[-1]:
                # Merge with previous run
                merged_lengths[-1] += filtered_lengths[i]
            else:
                # Start a new run
                merged_values.append(filtered_values[i])
                merged_lengths.append(filtered_lengths[i])

        return np.array(merged_values), np.array(merged_lengths)

    def get_key_tone(
        self, key: str, dur: float = 0.4
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Generates the Dual-Tone Multi-Frequency (DTMF) signal for a given key.

        Args:
            key (str): The key for which to generate the DTMF signal.
            dur (float, optional): The duration of the signal in seconds. Defaults to 0.4.

        Returns:
            tuple[np.ndarray, tuple[int,int]] | None: An array containing the generated DTMF signal.
        """
        key = key.upper()
        if len(key) != 1:
            return None
        if key not in DtmfGenerator.key_matrix:
            return None
        r_idx, c_idx = np.where(DtmfGenerator.key_matrix == key)
        f1 = DtmfGenerator.freqz_rows[int(r_idx)]
        f2 = DtmfGenerator.freqz_cols[int(c_idx)]

        # Create signal with both frequencies
        tt = np.arange(0.0, dur, 1 / self.sample_rate)
        A = 1 / 3.0  # np.iinfo(np.int32).max / 4
        return (
            A * (np.sin(2.0 * np.pi * f1 * tt) + np.sin(2.0 * np.pi * f2 * tt)),
            (f1, f2),
            key,
        )

    def get_tone_sequence(
        self,
        key_sequence: str,
        dur_key: float | tuple[float, float] | None = None,
        dur_pause: float | tuple[float, float] | None = None,
        noise_factor: float | tuple[float, float] | None = None,
        noise_freq_range: tuple[float, float] | None = None,
        with_labels: Literal["freqz", "keys"] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """Generates a sequence of DTMF tones for a given key sequence, with pauses and added noise.

        Args:
            key_sequence (str): The sequence of keys for which to generate the DTMF tones.

            dur_key (float | tuple[float, float] | None, optional): The duration of each key tone in seconds.
                If a float is provided, it specifies a fixed duration for all key tones.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each key tone will be randomly chosen. Defaults to None, which uses the instance's `dur_key`.

            dur_pause (float | tuple[float, float] | None, optional): The duration of the pause between key tones in seconds.
                If a float is provided, it specifies a fixed duration for all pauses.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each pause will be randomly chosen. Defaults to None, which uses the instance's `dur_pause`.

            noise_factor (float | tuple[float, float] | None, optional): The factor by which noise is scaled.
                If a float is provided, it specifies a fixed noise factor.
                If a tuple of two floats is provided, it specifies a range from which the noise factor
                will be randomly chosen. Defaults to None, which uses the instance's `noise_factor`.

            noise_freq_range (tuple[float, float] | None, optional): The frequency range for the noise.
                This specifies the minimum and maximum frequencies of the noise. Defaults to None, which uses the instance's `noise_freq_range`.

            with_labels (Literal["freqz", "keys"] | None, optional): Whether to include labels and their type.
                If "freqz" is provided, labels will indicate the presence of specific frequencies.
                If "keys" is provided, labels will indicate the presence of specific keys.
                Defaults to None, which means no labels are included.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray] | None: An array containing the generated DTMF tone sequence
            with pauses and noise, and optionally the labels, or None if any key is invalid.
        """
        if dur_key is None:
            dur_key = self.dur_key
        if dur_pause is None:
            dur_pause = self.dur_pause
        if noise_factor is None:
            noise_factor = self.noise_factor
        if noise_freq_range is None:
            noise_freq_range = self.noise_freq_range

        def rand_uniform_or_scalar(x):
            return random.uniform(*x) if isinstance(x, tuple) else x

        tones = [
            self.get_key_tone(c, dur=rand_uniform_or_scalar(dur_key))
            for c in key_sequence
        ]

        # Quit, if at least one of the keys was not found
        if any(t is None for t in tones):
            return None

        pause_signal = [0] * int(rand_uniform_or_scalar(dur_pause) * self.sample_rate)

        tones_with_pauses = [pause_signal]
        labels = None
        if with_labels == "freqz":
            labels = [
                np.zeros(
                    (len(pause_signal), DtmfGenerator.get_num_freqz()), dtype=np.int32
                )
            ]  # TODO: int32 necessary?
        elif with_labels == "keys":
            lab = np.zeros(
                (len(pause_signal), DtmfGenerator.get_num_keys() + 1), dtype=np.int32
            )
            lab[:, -1] = 1
            labels = [lab]

        for tone, (f1, f2), key in tones:
            pause_signal = [0] * int(
                rand_uniform_or_scalar(dur_pause) * self.sample_rate
            )
            tones_with_pauses.extend([tone, pause_signal])

            # Label per timestep
            if with_labels == "freqz":
                pause_label = np.zeros(
                    (len(pause_signal), DtmfGenerator.get_num_freqz()), dtype=np.int32
                )
                label = np.zeros(
                    (len(tone), DtmfGenerator.get_num_freqz()), dtype=np.int32
                )
                label[:, DtmfGenerator.FREQ_TO_IDX[f1]] = 1
                label[:, DtmfGenerator.FREQ_TO_IDX[f2]] = 1
                labels.extend([label, pause_label])
            elif with_labels == "keys":
                pause_label = np.zeros(
                    (len(pause_signal), DtmfGenerator.get_num_keys() + 1),
                    dtype=np.int32,
                )
                pause_label[:, -1] = 1
                label = np.zeros(
                    (len(tone), DtmfGenerator.get_num_keys() + 1), dtype=np.int32
                )
                label[:, np.where(DtmfGenerator.key_matrix.flatten() == key)[0]] = 1
                labels.extend([label, pause_label])

        signal = np.hstack(tones_with_pauses)

        # Finally, add some noise to the signal
        if noise_factor:
            noise = DtmfGenerator.band_limited_noise(
                *noise_freq_range, samples=signal.shape[0]
            )
            signal = signal + rand_uniform_or_scalar(noise_factor) * noise

        if with_labels:
            return signal, np.vstack(labels)
        return signal

    def generate_dataset(
        self,
        n_samples: int = 512,
        t_length: int = 2**13,
        dur_key: float | tuple[float, float] | None = None,
        dur_pause: float | tuple[float, float] | None = None,
        noise_factor: float | tuple[float, float] | None = None,
        noise_freq_range: tuple[float, float] | None = None,
        with_labels: Literal["freqz", "keys"] | False | None = "keys",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a dataset of DTMF tone sequences for classification.

        Args:
            n_samples (int, optional): The number of samples to generate. Defaults to 512.
            t_length (int, optional): The target length of each sequence. Defaults to 2**13.
            dur_key (float | tuple[float, float] | None, optional): The duration of each key tone in seconds.
                If a float is provided, it specifies a fixed duration for all key tones.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each key tone will be randomly chosen. Defaults to None, which uses the instance's `dur_key`.
            dur_pause (float | tuple[float, float] | None, optional): The duration of the pause between key tones in seconds.
                If a float is provided, it specifies a fixed duration for all pauses.
                If a tuple of two floats is provided, it specifies a range from which the duration
                of each pause will be randomly chosen. Defaults to None, which uses the instance's `dur_pause`.
            noise_factor (float | tuple[float, float] | None, optional): The factor by which noise is scaled.
                If a float is provided, it specifies a fixed noise factor.
                If a tuple of two floats is provided, it specifies a range from which the noise factor
                will be randomly chosen. Defaults to None, which uses the instance's `noise_factor`.
            noise_freq_range (tuple[float, float] | None, optional): The frequency range for the noise.
                This specifies the minimum and maximum frequencies of the noise. Defaults to None, which uses the instance's `noise_freq_range`.
            with_labels (Literal["freqz", "keys"] | False | None, optional): Whether to include labels and their type.
                If "freqz" is provided, labels will indicate the presence of specific frequencies.
                If "keys" is provided, labels will indicate the presence of specific keys. Defaults to "keys".

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The numpy array containing the generated sequences (X) (signals) if `with_labels` is `None` or `False`. Otherwise, Two numpy arrays, one containing the generated sequences (X)
            and the other containing the corresponding labels (Y).
        """
        X, Y = [], []
        rng = np.random.default_rng()
        for _ in range(n_samples):
            current_length = 0
            xx, yy = [], []
            while current_length < t_length:
                seq = "".join(rng.choice(DtmfGenerator.key_matrix.flatten(), size=5))
                xy = self.get_tone_sequence(
                    seq,
                    dur_key=dur_key,
                    dur_pause=dur_pause,
                    with_labels=with_labels,
                    noise_factor=noise_factor,
                    noise_freq_range=noise_freq_range,
                )
                x, y = xy if isinstance(xy, tuple) else (xy, None)

                xx.append(x[..., np.newaxis])
                yy.append(y)
                current_length += len(x)

            X.append(np.concatenate(xx)[:t_length])
            if with_labels:
                Y.append(np.concatenate(yy)[:t_length])

        return (np.array(X), np.array(Y)) if with_labels else np.array(X)

    def decode_prediction(self, prediction: np.ndarray) -> str:
        """Decode the model's prediction into a sequence of DTMF keys.

        This method takes the model's prediction, which is an array of shape (T, 17) or (1, T, 17),
        and decodes it into a sequence of DTMF keys. It applies run-length encoding to filter out
        short breaks and invalid key signals, then maps the remaining indexes to the corresponding keys.

        Args:
        prediction (np.ndarray): The model's prediction array. It should have shape (T, 17) or (1, T, 17),
            where T is the number of time steps and 17 is the number of possible classes (16 keys + 1 for pause).

        Returns:
        str: The decoded sequence of DTMF keys.
        """
        # a prediction of shape (T, 17) or (1, T, 17)
        predicted_indexes = prediction.squeeze().argmax(axis=-1)

        # According to the standard, breaks shorter than 10 ms should be discarded
        # Also, key signals shorter than 23ms are invalid (for now, we ignore this rule
        # and only discard those, which are shorter than 10 ms).
        min_run_length = int(self.sample_rate * 0.01)
        rle_values, _ = DtmfGenerator.run_length_encoding(
            predicted_indexes, min_run_length=min_run_length
        )

        # Remove the pauses and map indexes to keys:
        which_rle_values = (
            rle_values < DtmfGenerator.get_num_keys() if rle_values.any() else None
        )
        return (
            "".join(DtmfGenerator.get_key(rle_values[which_rle_values]))
            if which_rle_values is not None
            else ""
        )
