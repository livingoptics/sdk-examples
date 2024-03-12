#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Matplotlib image viewer

A simple way to display images captured by the LOCamera using Matplotlib

Example:
    ```python
    import numpy as np
    from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer

    image = np.random.rand(256, 256, 3) # create fake image
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    scene_view.update(image)
    viewer.render()
    ```

# Matplotlib spectra viewer

A simple way to display the spectra extracted from the encoded view of the LOCamera.

Example:
    ```python
    import numpy as np
    from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer

    spectra = np.random.rand(10, 96) # create fake list of spectra
    wavelengths = np.arange(96)
    viewer = LOMPLViewer()
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")
    spectra_view.update(spectra=spectra, wavelengths=wavelengths)
    ```
"""

import logging
from typing import Any, List

import numpy as np
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers._import import import_extra

# IMPORTANT: You MUST add the following line to set the log level BEFORE
# you import matplotlib, otherwise (on Linux) you will get a HUGE amount
# of debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
plt = import_extra("matplotlib.pyplot", extra="matplotlib")

_logger = logging.getLogger(__name__)

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PngImagePlugin").setLevel(logging.WARNING)


class SceneViewHandle:
    """Scene view handle returned to the user from LOMPLViewer when its add_scene_view is called.
    Allows users to update scene view.
    """

    def __init__(
        self,
        index: int,
        axes_container: List,
        title: str = "",
    ):
        """Initialises scene view handle.

        Args:
            index (int): Index used to retrieve the corresponding matplotlib axis.
            axes_container (List): List container containing the matplotlib axes.
            title (str, optional): Title of the axis. Defaults to "".
        """
        self.index = index
        self.axes_container = axes_container
        self.title = title
        self.img = None

    def update(self, frame: np.ndarray):
        """Update the corresponding matplotlib axis with the given frame.

        Args:
            frame (np.ndarray): Image numpy array to render onto the axis.
        """
        ax = self.axes_container[0][self.index]
        ax.set_title(self.title)

        if self.img is None:
            # On the first loop, run imshow and store the imshow handler
            self.img = ax.imshow(frame)
        else:
            # On subsequent loops, just update the data which is much faster.
            self.img.set_data(frame)


class SpectraViewHandle:
    """Spectra view handle returned to the user from LOMPLViewer when its add_spectra_view is called.
    Allows users to update spectra view.
    """

    def __init__(self, index: int, axes_container: List, title: str = "", spectral_units: str = ""):
        """Initialises spectra view handle.

        Args:
            index (int): Index used to retrieve the corresponding matplotlib axis.
            axes_container (List): List container containing the matplotlib axes.
            title (str, optional): Title of the axis. Defaults to "".
            spectral_units (str, optional): The units of the spectra being plotted. Default to ""
        """
        self.index = index
        self.axes_container = axes_container
        self.title = title
        self.spectral_units = spectral_units

    def update(self, spectra: np.ndarray, wavelengths: np.ndarray):
        """Update the corresponding matplotlib axis with the spectra data.

        Args:
            spectra (np.ndarray): (SAMPLES, CHANNELS) array of spectra information.
            wavelengths (np.ndarray): (Channels,) array containing wavelength(nm) information for each channel.
        """
        ax = self.axes_container[0][self.index]
        ax.cla()
        ax.set_title(self.title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(self.spectral_units)
        for spectrum in spectra:
            ax.plot(wavelengths, spectrum)


class LOMPLViewer:
    """
    Matplotlib viewer for multiple scene and or spectra views.
    Plots a list of scene and or spectra.
    """

    def __init__(self, fig_scale: int = 5, frame_rate: float = 100):
        """Instantiate a wrapper class for a plt figure and axis which can be iteratively updated with new scene/spectral data.

        Args:
            fig_scale (int, optional): Matplotlib figure scale in inches. Defaults to 5.
            frame_rate (float, optional): Frame rate(hz) in for each render of the viewer. Defaults to 100.
        """
        self.index = 0
        self.fig_scale = fig_scale
        self.frame_rate = frame_rate
        self.axes_container: List[Any] = [None]

    def add_scene_view(self, title: str = "") -> SceneViewHandle:
        """Adds a scene view and returns a scene view handle.

        Args:
            title (str, optional): Title of the axis. Defaults to "".

        Returns:
            SceneViewHandle: Scene view handle with which you can update data.
        """
        plt.close()
        scene_view_handle = SceneViewHandle(
            index=self.index,
            axes_container=self.axes_container,
            title=title,
        )
        self.index += 1
        self.fig, axes = plt.subplots(1, self.index)

        # plt.subplots(1, 1) returns an axis but plt.subplots(1, >1) returns a numpy array of axes.
        self.axes_container[0] = np.array([axes]) if type(axes) != np.ndarray else axes

        self.fig.set_size_inches(self.fig_scale * self.index, self.fig_scale)
        return scene_view_handle

    def add_spectra_view(self, title: str = "", spectral_units: str = "") -> SpectraViewHandle:
        """Adds a spectra view and returns a spectra view handle.

        Args:
            title (str, optional): Title of the axis. Defaults to "".
            spectral_units (str, optional): The units of the spectra being plotted. Default to "".

        Returns:
            SpectraViewHandle: _description_
        """
        plt.close()
        spectra_view_handle = SpectraViewHandle(
            index=self.index, axes_container=self.axes_container, title=title, spectral_units=spectral_units
        )
        self.index += 1
        self.fig, self.axes_container[0] = plt.subplots(1, self.index)
        self.fig.set_size_inches(self.fig_scale * self.index, self.fig_scale)
        return spectra_view_handle

    def render(self):
        """Renders the matplotlib figure."""
        plt.pause(1 / self.frame_rate)


def _main():
    """main entry point"""
    # get an image viewer for each sensor
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view()

    filepath = "simulated"  # defaults to simulated camera
    with LOCamera(filepath) as stream:
        stream.gain = 0
        for i, frame in enumerate(stream):
            (encoded_info, encoded_frame), (scene_info, scene_frame) = frame
            print(encoded_info)
            scene_view.update(scene_frame)
            viewer.render()


if __name__ == "__main__":
    _main()
