#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Run a live field calibration to improve spectral decoding accuracy

This example demonstrates how to connect to the LOCamera, capture data, use a 600nm filtered image to improve spectral
decoding and save that image for later use.

Commandline:
    `python field_calibration_live.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/data/flat-field-filtered.loraw
    - Output : None

    To run this example application you should be in {install-dir}/src/python/examples/field_calibration

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:

    - `pip install {wheel}['matplotlib']`

"""
import os

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.helpers.time import timestamp
from lo.sdk.integrations.matplotlib.simple_viewer import LOMPLViewer


def main():
    # Latest Calibration is a symlink to the calibration for the camera that is connected.
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")

    decoder = SpectralDecoder.from_calibration(calibration_folder)

    data_folder = os.path.join(getdatastorepath(), "lo", "share", "data")
    os.makedirs(data_folder, exist_ok=True)
    current_time = timestamp()
    filename = f"flat-field-filtered-{current_time}.loraw"
    filepath = os.path.join(data_folder, filename)
    print(f"Saving to {filepath}")

    # get image viewers
    viewer = LOMPLViewer()
    scene_view = viewer.add_scene_view()
    spectra_view = viewer.add_spectra_view(title="Extracted spectra")

    with LOCamera() as cam:
        # Set up the frame rate, exposure and gain
        cam.frame_rate = int(5120e3)  # μhz
        cam.exposure = int(1950e3)  # μs
        # We use a large gain as the 600nm filter blocks out all light except for a very narrow band.
        cam.gain = 100

        input("Put the Calibration Filter on the camera and press enter to continue")
        [cam.get_frame() for i in range(3)]  # flush the buffer
        frame_600nm = cam.get_frame()

        decoder.update_calibration(frame_600nm, filter_wavelength=600)
        with open(filepath, mode="w", format="loraw") as f:
            f.write(frame_600nm)

        input("Please remove filter and press enter to continue")

        # Set the gain back to a sensible value after the filter is removed.
        cam.gain = 0
        for i, frame in enumerate(cam):
            processed_frame = decoder(frame)

            info, scene, spectra = processed_frame
            print(info)

            # Display scene
            scene_view.update(frame=scene)

            # Display every 50th spectra
            spectra_view.update(spectra=spectra[::50, :], wavelengths=info.wavelengths)
            viewer.render()

            # Leave the 'with' context after displaying 10 frames.
            if i >= 10:
                break


if __name__ == "__main__":
    main()
