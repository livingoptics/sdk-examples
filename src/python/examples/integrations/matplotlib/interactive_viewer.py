#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# MPLSpectraViewer

Interactively display the scene and spectral list and plot the spectrum of selected pixels from the scene.
Explore the spectral content of the scene and select the spectrum of an object using the mouse.

Matplotlib isn't the fastest library for plotting video rate data but applying a few tricks, it can be
used to quickly build some interactive tools.

Below we show how to interact with the coordinate system of the camera to link together spectra and the scene view.


Example:
    ```python
        # Latest Calibration is a symlink to the calibration for the camera that is connected.
        calibration_folder = Path("/datastore/lo/share/calibrations/latest_calibration").as_posix()

        decode = SpectralDecoder.from_calibration(calibration_folder)

        mpl_viewer = MPLSpectraViewer(decode.sampling_coordinates)

        with LOCamera() as cam:
            cam.gain = 0  # change the settings if running live
            for i, frame in enumerate(cam):
                if not mpl_viewer.running:
                    break

                metadata, scene_frame, spectra = decode(frame)  # decode

                mpl_viewer.plot(scene_frame, spectra, metadata.wavelengths)
                mpl_viewer.update(0.05)

                spectrum = mpl_view.get_selected_spectrum()

                # Spectrum will be None until the user selects a pixel with the mouse.
                if spectrum is not None:
                    do_stuff(spectrum)
    ```

"""

import logging
import os

import numpy as np
from lo.sdk.api.acquisition.data.coordinates import (
    SceneToSpectralIndex,
    SpectralIndextoScene,
)
from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers._import import import_extra
from lo.sdk.helpers.path import getdatastorepath

# IMPORTANT: You MUST add the following line to set the log level BEFORE
# you import matplotlib, otherwise (on Linux) you will get a HUGE amount
# of debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
plt = import_extra("matplotlib.pyplot", extra="matplotlib")


class MPLSpectraViewer:
    def __init__(self, sampling_coordinates, title="", spectral_units="arb units"):
        # Make some axis to hold the figures
        self.fig, (
            self.ax_scene,
            self.ax_spectra,
            self.ax_spectral_image,
        ) = plt.subplots(1, 3)
        self.fig.suptitle(title)

        # Label our axis
        self.spectral_units = spectral_units
        self.ax_scene.set_title("Scene view")
        self.ax_spectra.set_title(f"Extracted Spectra, {self.spectral_units}")
        self.ax_spectra.set_xlabel("Wavelength (nm)")
        self.ax_spectral_image.set_title("Spectral Image")
        self.ax_spectral_image.set_xlabel("Wavelength (nm)")

        # Make some variables to store info in later
        self.wavelengths = None
        self._selected_spectrum = None
        self._current_spectra = None
        self._im_scene = None
        self._im_spec = None
        self._running = True
        self._spectral_lines = []  # (store idx, plot object)
        self._scene_markers = []

        # Attach a callback so we can detect clicks on the scene
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        # Helper classes for translating coordinate spaces.
        self.get_spectral_idx = SceneToSpectralIndex(sampling_coordinates)
        self.get_scene_idx = SpectralIndextoScene(sampling_coordinates)

    def plot(self, scene, spectra, wavelengths):
        """Add more data, and update the plots.

        Args:
            scene (np.ndarray): Spacial data (y,x)
            spectra (np.ndarray): Spectral list (n,ch)
            wavelengths (np.ndarray): List of wavelengths in nm (ch)
        """
        self.wavelengths = wavelengths
        self._current_spectra = spectra

        # If we haven't plotted anything before make the axis
        if self._im_scene is None:
            self._im_scene = self.ax_scene.imshow(scene)
            self._im_spec = self.ax_spectral_image.imshow(spectra, aspect=0.02)

            # Update the ticks on the axis (we don't want to label all of them)
            # ::13 writes a label out every 13th position
            self.ax_spectral_image.set_xticks(np.arange(len(wavelengths))[::13])
            self.ax_spectral_image.set_xticklabels(wavelengths[::13])
            self.ax_spectra.set_xticks(np.arange(len(wavelengths))[::13])
            self.ax_spectra.set_xticklabels(wavelengths[::13])

        else:  # We plotted before
            # Use set_data as it's much faster
            self._im_scene.set_data(scene)
            self._im_spec.set_data(spectra)

    def update(self, delay=0.05):
        """Update the spectra with the latest point.

        Args:
            delay (float, optional): delay for plot update in seconds. Defaults to 0.05 seconds.
        """

        # for every point we have already clicked on update the plot with new data that just arrived.
        for idx, line in self._spectral_lines:
            line.set_ydata(self._current_spectra[idx, :])
        # Allow matplotlib to update itself.
        plt.pause(delay)

    def on_close(self, event):
        print("Closed Figure!")
        self._running = False
        plt.close()

    def onclick(self, event):
        # First detect if we have clicked on one of the images
        if event.inaxes is self.ax_scene:
            # We clicked on the scene
            ix, iy = event.xdata, event.ydata

            # Lookup nearest spectral index
            spectral_idx = self.get_spectral_idx([iy, ix])

            # Convert that spectral index back to a scene index
            sy, sx = self.get_scene_idx(spectral_idx)
            print(f"x = {ix}, y = {iy}, {spectral_idx=},{sx=} {sy=}")

        elif event.inaxes is self.ax_spectral_image:
            # We clicked on the spectral list

            # Select the nearest spectral index
            spectral_idx = np.round(event.ydata).astype(int)

            # find where that spectral index is in the scene
            sy, sx = self.get_scene_idx(spectral_idx)
            print(f"{spectral_idx=},{sx=} {sy=}")
            # Appending Spectral idx and plot object, so we can update later
        else:
            return

        # Save the current spectra at the point we clicked for later
        self._selected_spectrum = self._current_spectra[spectral_idx]

        # Add to our list of points to watch and plot on the graph and add a marker
        self._spectral_lines.append((spectral_idx, self.ax_spectra.plot(self.wavelengths, self._selected_spectrum)[0]))
        self._scene_markers.append(self.ax_scene.scatter(sx, sy))

    def get_selected_spectra(self):
        return self._selected_spectrum

    @property
    def running(self):
        return self._running


def _main():
    """main entry point"""

    # Latest Calibration is a symlink to the calibration for the camera that is connected.
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")
    decode = SpectralDecoder.from_calibration(calibration_folder)

    mpl_viewer = MPLSpectraViewer(decode.sampling_coordinates)

    with LOCamera() as cam:
        cam.gain = 0  # change the settings if running live
        for i, frame in enumerate(cam):
            # detect if someone closed the window
            if not mpl_viewer.running:
                break

            # Unpack the decoded object
            metadata, scene_frame, spectra = decode(frame)

            mpl_viewer.plot(scene_frame, spectra, metadata.wavelengths)
            mpl_viewer.update(0.05)


if __name__ == "__main__":
    _main()
