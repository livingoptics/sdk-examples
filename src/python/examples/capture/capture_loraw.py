#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# Capture raw format data from the camera.

This example demonstrates how to connect to the LOCamera, capture data, save it in a raw format and then read it back.

Commandline:
    `python capture_loraw.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : None
    - Output : /datastore/lo/share/data/raw_test.loraw

    To run this example application you should be in {install-dir}/src/python/examples/capture

Installation:
    Installed as part of the sdk.

"""

import os

from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath
from lo.sdk.helpers.time import timestamp


def main():
    # Set up a place to save data to
    data_folder = os.path.join(getdatastorepath(), "lo", "share", "data")
    os.makedirs(data_folder, exist_ok=True)
    current_time = timestamp()
    filename = f"raw-test-{current_time}.loraw"
    filepath = os.path.join(data_folder, filename)
    print(f"Saving to {filepath}")

    # Connect to the camera - the 'with' context will automatically open and close the camera when we enter/exit it.
    with LOCamera() as cam:
        with open(filepath, mode="w", format="loraw") as f:
            # Set camera settings - these will need to change depending on light levels and application
            cam.frame_rate = int(5120e3)  # μhz
            cam.exposure = int(1950e3)  # μs
            cam.gain = 0

            # Iterate over 10 frames and save them to filepath
            for i, frame in enumerate(cam):
                (scene_metadata, encoded_frame), (scene_metadata, scene_frame) = frame
                f.write(frame)

                # Leave the 'with' context after capturing 10 frames.
                if i >= 10:
                    break

    # Read the saved data
    print("reading file {}".format(filepath))
    with open(filepath, mode="r") as f:
        for frame in f:
            # Each frame in the saved file is a tuple of tuples.
            # The first tuple in each frame is the metadata and image date for the encoded view.
            # The second tuple in each frame is the metadata and image data for the scene view.
            (encoded_metadata, encoded_frame), (scene_metadata, scene_frame) = frame

            # Explore some of the metadata saved in the loraw format.
            print(f"{encoded_metadata.vd.timestamp.tv_usec=}")
            print(f"{scene_metadata.vd.timestamp.tv_usec=}")
            print(f"{encoded_frame.mean()=}")
            print(f"{scene_frame.mean()=}")


if __name__ == "__main__":
    main()
