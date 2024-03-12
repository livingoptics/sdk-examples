#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.
"""
# Methods for replaying saved data from file.

This example demonstrates how to replay LO data from both the `.loraw` and `.lo` formats as we as how to use the seek()
method to read specific frames from an LO file.

`lo.sdk.api.acquisition.io.open.open` behaves like python's [open method](
https://docs.python.org/3/library/functions.html#open).

Since the video files can be quite large, image data values are only read when specifically requested via the
read() or iterator methods.

Commandline:
    `python from_file.py`

!!! Note

    Workstation users should run this from inside their Python [virtual environment](../../../install-guide.md#quick-install)


Tips:
    Data locations

    - Input : /datastore/lo/share/samples/replay/raw_test.loraw
    - Input : /datastore/lo/share/samples/replay/decoded_test.loraw
    - Output : None

    [Download]("../../../data-samples-overview.md") the Living Optics sample data folder and unpack it in `/datastore/lo/share/samples`.

    This example will run indefinitely, so press ctrl+c to close it.

    To run this example application you should be in {install-dir}/src/python/examples/replay

Installation:
    Installed as part of the sdk.

"""
import os

from lo.sdk.api.acquisition.io.open import open
from lo.sdk.api.camera.camera import LOCamera
from lo.sdk.helpers.path import getdatastorepath


def main():
    samples_folder = os.path.join(getdatastorepath(), "lo", "share", "samples", "replay")

    print("Raw file playback")
    lo_raw_file = os.path.join(samples_folder, "raw_test.loraw")
    with open(lo_raw_file) as f:
        for (encoded_info, encoded_frame), (scene_info, scene_frame) in f:
            print(encoded_info)

    print("Using the LOCamera mock allows prototyping of live applications")
    with open(lo_raw_file) as f:
        with LOCamera(file=f) as cam:
            for i, ((encoded_info, encoded), (scene_info, scene)) in enumerate(cam):
                print(scene.mean())
                print(cam.frame_rate)  # Prints the frame rate in the file
                # This runs at the frame rate of the file!
                if i > 3:
                    break

    print("Processed data playback")
    lo_processed_file = os.path.join(samples_folder, "decoded_test.lo")
    with open(lo_processed_file) as f:
        for i, (metadata, scene, spectra) in enumerate(f):
            if i > 3:
                break

    print("Files support more fine control over position in the file")
    with open(lo_processed_file) as f:
        print("Frame 8:")
        f.seek(8)  # goto the 8th frame
        frame = f.read()
        print(frame)

        print("Final frame")
        f.seek(len(f) - 1)  # goto the last frame
        frame = f.read()
        print(frame)


if __name__ == "__main__":
    main()
