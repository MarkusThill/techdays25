"""A collection of plotting utils for lab 1 and lab 2."""

# from collections.abc import Callable
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def plot_speedup(df, reference_model: str, batch_size: int):
    # Ensure the batch size exists in the DataFrame
    if batch_size not in df.index:
        raise ValueError(f"Batch size {batch_size} not found in the DataFrame index.")

    # Ensure the reference model exists in the DataFrame columns
    if reference_model not in df.columns:
        raise ValueError(
            f"Reference model {reference_model} not found in the DataFrame columns."
        )

    # Extract the runtimes for the given batch size
    runtimes = df.loc[batch_size]

    # Calculate the speedup relative to the reference model
    reference_runtime = runtimes[reference_model]
    speedup = reference_runtime / runtimes

    # Plot the bar chart
    plt.figure(figsize=(7, 5))

    # colors = plt.cm.viridis(np.linspace(0, 1, len(speedup)))  # Use a colormap for different colors
    cmap = plt.get_cmap("tab10")

    colors = [cmap(i) for i in range(16)]  # Get 16 distinct colors
    bars = speedup.plot(kind="bar", color=colors)

    # Annotate each bar with the speedup value
    for bar in bars.patches:
        height = bar.get_height()
        bars.annotate(
            f"x{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height - 0.1),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.title(f'Speedup relativ zu "{reference_model}" für Batch-Größe {batch_size}')
    plt.xlabel("Modell")
    plt.ylabel("Speedup")
    plt.xticks(rotation=45)
    plt.grid(axis="y", which="both")
    plt.show()


def plot_benchmark_results(
    results: dict[str, pd.DataFrame],
    title: str = "",
    xscale: str | None = None,
    yscale: str | None = None,
):
    """Plot benchmark results for multiple models.

    This function plots the latency benchmarks for multiple models, showing the median
    latency and standard deviation for different batch sizes.

    Args:
        results (dict[str, pd.DataFrame]): A dictionary where the keys are model names (strings)
            and the values are pandas DataFrames containing the timing results for each batch size.
        title (str, optional): The title of the plot. Defaults to an empty string.
        xscale (str | None, optional): The scale for the x-axis (e.g., 'log'). Defaults to None.
        yscale (str | None, optional): The scale for the y-axis (e.g., 'log'). Defaults to None.
    """
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(results))]

    plt.figure(figsize=(8, 5))
    for (model_name, df), color in zip(results.items(), colors):
        # Convert columns to integers for sorting
        df.columns = df.columns.astype(int)
        df = df[sorted(df.columns)]

        # Compute medians and standard deviations
        medians = df.median()
        stds = df.std()

        # X-axis values (sorted batch sizes)
        x = medians.index

        # Plot with connected points and error bars

        plt.errorbar(
            x,
            medians,
            yerr=stds,
            fmt="-o",
            capsize=5,
            color=color,
            ecolor=color,
            elinewidth=2,
            markerfacecolor="white",
            label=model_name,
        )

    # Make it pretty
    plt.xlabel("Batch-Größe")
    plt.ylabel("Latenz [s]")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    if xscale:
        plt.xscale(xscale)
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().xaxis.set_ticks(medians.index)
        plt.gca().xaxis.set_tick_params(rotation=90 if max(medians.index) > 1024 else 0)
    if yscale:
        plt.yscale(yscale)
    plt.grid(True, linestyle="--", alpha=0.6, which="major")
    plt.show()


def multiple_formatter(
    denominator: int = 2, number: float = np.pi, latex: str = r"\pi"
) -> Callable[[float, int], str]:
    """Creates a formatter function for matplotlib that formats axis labels as multiples of a given number.

    Args:
        denominator (int, optional): The denominator to use for the fraction representation. Defaults to 2.
        number (float, optional): The base number to use for the multiples. Defaults to np.pi.
        latex (str, optional): The LaTeX string to use for the base number.

    Returns:
        Callable[[float, int], str]: A function that formats a given value as a LaTeX fraction of the base number.
    """

    def gcd(a: int, b: int) -> int:
        """Computes the greatest common divisor of two integers.

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The greatest common divisor of a and b.
        """
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x: float, pos: int) -> str:
        """Formats a given value as a LaTeX fraction of the base number.

        Args:
            x (float): The value to format.
            pos (int): The position (not used in this implementation).

        Returns:
            str: The formatted string.
        """
        den = denominator
        num = np.int64(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return rf"${latex}$"
            if num == -1:
                return rf"$-{latex}$"
            return rf"${num}{latex}$"
        if num == 1:
            return rf"$\frac{{{latex}}}{{{den}}}$"
        if num == -1:
            return rf"$\frac{{-{latex}}}{{{den}}}$"
        return rf"$\frac{{{num}{latex}}}{{{den}}}$"

    return _multiple_formatter


class Multiple:
    """A class to create locators and formatters for matplotlib axes based on multiples of a given number."""

    def __init__(
        self, denominator: int = 2, number: float = np.pi, latex: str = r"\pi"
    ):
        """Initializes the Multiple class with the given parameters.

        Args:
            denominator (int, optional): The denominator to use for the fraction representation. Defaults to 2.
            number (float, optional): The base number to use for the multiples. Defaults to np.pi.
            latex (str, optional): The LaTeX string to use for the base number.
        """
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self) -> plt.MultipleLocator:
        """Creates a locator for matplotlib axes based on multiples of the base number.

        Returns:
            plt.MultipleLocator: A locator for matplotlib axes.
        """
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self) -> plt.FuncFormatter:
        """Creates a formatter for matplotlib axes that formats labels as multiples of the base number.

        Returns:
            plt.FuncFormatter: A formatter for matplotlib axes.
        """
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


def zplane(b, a=np.array([1])):
    """Plot the complex z-plane given a transfer function."""
    # Create a unit circle
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    unit_circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="dotted")
    ax.add_artist(unit_circle)

    # Plot zeros and poles
    zeros = np.roots(b)
    poles = np.roots(a)
    plt.scatter(
        np.real(zeros),
        np.imag(zeros),
        s=50,
        marker="o",
        facecolors="none",
        edgecolors="b",
        label="Zeros",
    )
    plt.scatter(
        np.real(poles), np.imag(poles), s=50, marker="x", color="r", label="Poles"
    )

    # Set plot limits and labels
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color="black", lw=1)
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Z-Plane Diagram")
    plt.grid()
    plt.legend()
    plt.show()


# Example: Design a low-pass FIR filter using the window method
# numtaps = 21  # Number of filter coefficients (taps)
# cutoff = 0.3  # Normalized cutoff frequency (0 to 1, where 1 corresponds to Nyquist frequency)
# b = firwin(numtaps, cutoff)
