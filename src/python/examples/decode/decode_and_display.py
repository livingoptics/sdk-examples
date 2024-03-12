#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Capture decode and view spectra live.

This example demonstrates how to connect to the LOCamera, capture data, decode the spectral content of the encoded view
and display the scene and spectra using matplotlib.

Commandline:
    `python decode_and_display.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : None
    - Output : None

    To run this example application you should be in {install-dir}/src/python/examples/decode

    This example will run indefinitely, so press ctrl+c to close it.

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""
import os.path
from pathlib import Path

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoRGB8
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    # Latest Calibration is a symlink to the calibration for the camera that is connected.
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")
    decoder = SpectralDecoder.from_calibration(calibration_folder)

    # get image and spectra viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view(title="Scene view")
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")

    with LOCamera() as cam:
        cam.frame_rate = int(5120e3)  # μhz
        cam.exposure = int(1950e3)  # μs
        cam.gain = 0  # change the settings if running live
        for frame in cam:
            try:
                # Decode the spectra from the encoded view and convert the scene view to RGB8
                processed_frame = decoder(frame, scene_decoder=LORAWtoRGB8)

                metadata, scene, spectra = processed_frame

                # Display the scene
                scene_view.update(scene)
                # For 8bit and fp32 format converters see lo.sdk.api.acquisition.data.formats: LORAWtoSPECTRA8
                # and LORAWtoSPECTRAF32

                # Display the spectra
                spectra_view.update(spectra=spectra[::4, :], wavelengths=metadata.wavelengths)
                viewer.render()

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
