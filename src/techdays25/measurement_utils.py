"""A module to generate some synthetic data for various examples."""

import numpy as np
import pandas as pd


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
