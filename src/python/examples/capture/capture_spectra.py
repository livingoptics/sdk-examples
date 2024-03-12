#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Capture, decode spectra and save to file.

This example demonstrates how to connect to the LOCamera, capture data, decode the spectral content of the encoded view
and save it in .lo format and then read it back.

Commandline:
    `python capture_spectra.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : None
    - Output : /datastore/lo/share/data/decode_and_capture_test.lo

    To run this example application you should be in {install-dir}/src/python/examples/capture

Installation:
    Installed as part of the sdk.

"""
import os
from pathlib import Path

from lo.sdk.api.acquisition.data.decode import SpectralDecoder
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12
from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.helpers.time import timestamp


def main():
    # We need the factory calibration for our camera in order to decode the spectral information from the encoded view.
    # A simlink to a default is stored here if we are on an Orin
    # Look here /datastore/lo/share/calibration/CAMERA_SERIAL_NUMBER/... you can see f# and  focal length in mm
    calibration_folder = os.path.join(getdatastorepath(), "lo", "share", "calibrations", "latest_calibration")
    decoder = SpectralDecoder.from_calibration(calibration_folder)

    # Set up a place to save data to
    data_folder = Path("/datastore/lo/share/data")
    os.makedirs(data_folder.as_posix(), exist_ok=True)
    current_time = timestamp()
    filename = f"decode-and-capture-test-{current_time}.lo"
    filepath = os.path.join(data_folder, filename)
    print(f"Saving to {filepath}")

    # Here we use a 'with' context to open and close the Camera.
    # If we want to do this in simulated mode. we can pass a file using open("something.loraw")
    with LOCamera() as cam:
        with open(filepath, "w", format="lo") as f:
            # We can set some parameters on the stream Note these might depend on your current light levels
            cam.frame_rate = int(5120e3)  # μhz
            cam.exposure = int(1950e3)  # μs
            cam.gain = 0

            for i, frame in enumerate(cam):
                # Decode spectra and decode the scene into 12Bit still bayered format
                processed_frame = decoder(frame, scene_decoder=LORAWtoLOGRAY12, description="Experimental Description")
                f.write(processed_frame)
                if i > 10:
                    # quit after 10 frames
                    break

    with open(filepath, "r") as f:
        # Iterate over each saved frame
        # Each frame in the .lo format is a tuple of (metadata, scene image, spectra array)
        for metadata, scene, spectra in f:
            for v in vars(metadata):
                print(v, getattr(metadata, v))


if __name__ == "__main__":
    main()
