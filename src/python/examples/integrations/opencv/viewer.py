#!/usr/bin/python3

# This file is subject to the terms and conditions defined in file
# `COPYING.md`, which is part of this source code package.

"""
# OpenCV image viewer.

A small example to show how to view and debayer the LORaw output from the camera.

Installation:
    Installed as part of the sdk.
    Custom installations may require you to install lo-sdk extras:
    - `pip install {wheel}['cv2']`

Example:
    ```python
            with LOCamera(file=None) as cam:
                for frame in cam:
                    try:
                        # The camera returns a packed tuple of frame_info,np.ndarrays
                        (encoded_info, encoded_frame), (scene_info, scene_view) = frame

                        # Convert from the raw camera format to 12bit
                        debayered = cv2.cvtColor(LORAWtoLOGRAY12(scene_view), cv2.COLOR_BayerRGGB2BGR)

                        # Use a simple grey world assumption white balance to correct colour casts.
                        debayered = white_balance(debayered)

                        # Resize image for display
                        debayered = cv2.resize(debayered, (640, 480))

                        # in place normalise so we can cast to 8bit
                        cv2.normalize(debayered, debayered, 0, 255, cv2.NORM_MINMAX)

                        # Push to scene view object
                        cv2.imshow("Scene View", cv2.resize(debayered.astype(np.uint8), (640, 480)))
                        cv2.waitKey(10)
                    except KeyboardInterrupt:
                        break
    ```

"""


import cv2
import numpy as np
from lo.sdk.api.acquisition.data.formats import LORAWtoLOGRAY12
from lo.sdk.api.camera.camera import LOCamera


def white_balance(debayered_frame, threshold=0.8):
    max = np.max(debayered_frame, axis=(2))
    min = np.min(debayered_frame, axis=(2))
    saturation = (max - min) / max

    wts = np.mean(debayered_frame[saturation < threshold], axis=(0))
    wts /= wts.max()

    return (debayered_frame / wts).astype(np.uint16)


def main():
    """Open cv"""

    with LOCamera(file=None) as cam:
        # change some settings using camera level api
        cam.frame_rate = 40000000
        cam.gain = 0
        cam.exposure = 99000

        for frame in cam:
            try:
                # The camera returns a packed tuple of frame_info,np.ndarrays
                (encoded_info, encoded_frame), (scene_info, scene_view) = frame

                # Convert from the raw camera format to 12bit
                debayered = cv2.cvtColor(LORAWtoLOGRAY12(scene_view), cv2.COLOR_BayerRGGB2BGR)

                # Use a simple grey world assumption white balance to correct colour casts.
                debayered = white_balance(debayered)

                # Resize image for display
                debayered = cv2.resize(debayered, (640, 480))

                # in place normalise so we can cast to 8bit
                cv2.normalize(debayered, debayered, 0, 255, cv2.NORM_MINMAX)

                # Push to scene view object
                cv2.imshow("Scene View", cv2.resize(debayered.astype(np.uint8), (640, 480)))
                cv2.waitKey(10)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
