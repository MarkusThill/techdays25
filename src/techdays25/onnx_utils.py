"""Various ONNX helper functions."""

import sys
from pathlib import Path

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
