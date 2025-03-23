"""A module to generate some synthetic data for various examples."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def generate_measurement_data(
    num_samples=100,
    voltage_min=0,
    voltage_max=12,
    resistance=470,
    offset_voltage=3,
    noise_factor=2,
    random_seed=42,
):
    """Generate synthetic measurement data for voltage and current.

    This function simulates the measurement of voltage and current values
    across a resistor, with added noise to mimic real-world conditions.

    Args:
        num_samples (int, optional): The number of measurements to generate. Default is 100.
        voltage_min (float, optional): The minimum voltage value. Default is 0.
        voltage_max (float, optional): The maximum voltage value. Default is 12.
        resistance (float, optional): The resistance value in ohms. Default is 470.
        offset_voltage (float, optional): The offset voltage to be added to the measured voltage. Default is 3.
        noise_factor (float, optional): The factor by which to amplify the noise in the current measurements. Default is 2.
        random_seed(int, optional): The seed for initializing the pseudo random number generator.

    Returns:
        pd.DataFrame: A DataFrame containing the generated voltage and current measurements.
            The DataFrame has two columns:
            - "voltage [V]": The measured voltage values in volts.
            - "current [mA]": The measured current values in milliamps.

    Example:
        >>> df = generate_measurement_data(
        ...     num_samples=50,
        ...     voltage_min=1,
        ...     voltage_max=10,
        ...     resistance=500,
        ...     offset_voltage=2,
        ...     noise_factor=1.5,
        ... )
        >>> print(df.head())
           Voltage [V]  Current [mA]
        0     3.467083      12.452112
        1     8.152498      21.434251
        2     2.987320      10.412532
        3     4.671839      14.876231
        4     1.238495       7.123456
    """
    rs = np.random.RandomState(random_seed)
    voltages = rs.uniform(size=num_samples) * (voltage_max - voltage_min) + voltage_min
    currents = (voltages + offset_voltage) / resistance + rs.standard_normal(
        size=num_samples
    ) / resistance * noise_factor
    return pd.DataFrame({
        "voltage [V]": voltages,
        "current [mA]": currents * 1000.0,
    })


def plot_measurement_data(
    xx: Sequence[float],
    yy: Sequence[float],
    yy_pred: Sequence[float] | dict[str, Sequence[float]] | None = None,
    xx_pred: Sequence[float] | None = None,
    xlabel: str = "$U \\ [V]$",
    ylabel: str = "$I \\ [mA]$",
    title: str | None = None,
) -> None:
    r"""Plots measurement data and optionally model predictions.

    Args:
        xx (Sequence[float]): The x-axis data points (e.g., voltage measurements).
        yy (Sequence[float]): The y-axis data points (e.g., current measurements).
        yy_pred (Sequence[float] | None, optional): The predicted y-axis data points (e.g., model output). Defaults to None.
        xx_pred (Sequence[float] | None, optional): The data used to predict `yy_pred`. Can be ignored, if it is th same as `xx`. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to "$U \\ [V]$".
        ylabel (str, optional): The label for the y-axis. Defaults to "$I \\ [mA]$".
        title (str | None, optional): The title of the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.toolbar_visible = False
    fig.canvas.resizable = False
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.capture_scroll = False

    ax.scatter(xx, yy, label="measured data")

    if yy_pred is not None:
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        xx_pred = xx if xx_pred is None else xx_pred
        if not isinstance(yy_pred, dict):
            yy_pred = {"model": yy_pred}

        for idx, (model_name, yy_model) in enumerate(yy_pred.items()):
            # It is sufficient to use only both ends of the line:
            ax.plot(
                [min(xx_pred), max(xx_pred)],
                [min(yy_model), max(yy_model)],
                color=colors[-idx - 1],
                label=model_name,
            )
            ax.scatter(
                xx_pred, yy_model, color=colors[-idx - 1], label=None, marker="x"
            )

    ax.minorticks_on()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)
    ax.set_xlim(None, None)

    if title:
        ax.set_title(title)

    ax.grid(True, which="both")

    ax.legend()
    plt.show()


def sse(xx: np.ndarray, yy: np.ndarray):
    """Calculate the Sum of Squared Errors (SSE) between two NumPy arrays.

    Args:
        xx (np.ndarray): The first input array.
        yy (np.ndarray): The second input array.

    Returns:
        float: The sum of squared errors between the versions of the input arrays.
    """
    return float(((xx - yy) ** 2).sum())
